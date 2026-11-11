"""[Reference copy of ingest.py before the 2026-11-09 speed-up (E-FAST); used only by check_ingest_v2.py.] FCX ingestion: turn the raw request of every LLM call into an append-only transition log.

A request is the ordered list of messages an agent sends for one call. Consecutive requests of a trace mostly share their
messages, so the differencer records only what changed: `open` when a message occurrence enters the context, `close` when it
leaves. An occurrence keeps a 64-bit order key, so the context of any call is the set of open occurrences sorted by that key.
The differencer sees message IDs only; it knows nothing about the policy that produced the requests.

  Transition(trace, span, op, occ, msg_id, ord)   op = +1 (open) or -1 (close); `occ` is unique within a trace
  Checkpoint(trace, span, occs, ords, msg_ids)    the open occurrences in order, every `ckpt` calls
"""
import os

import numpy as np, orjson, xxhash

GAP = 1 << 32                                                     # order-key distance between appended occurrences


def lcs_pairs(a, b, budget):
    """Longest-common-subsequence alignment of two ID sequences by Myers' algorithm, giving up after `budget` edits.
    Returns the matched index pairs (i, j) in increasing order, or None when the edit distance exceeds the budget."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return []
    v, trace = {1: 0}, []
    for d in range(min(budget, n + m) + 1):
        trace.append(dict(v))
        for k in range(-d, d + 1, 2):
            x = v[k + 1] if k == -d or (k != d and v[k - 1] < v[k + 1]) else v[k - 1] + 1
            y = x - k
            while x < n and y < m and a[x] == b[y]:
                x += 1; y += 1
            v[k] = x
            if x >= n and y >= m:                                 # backtrack through the saved frontiers
                pairs = []
                for dd in range(d, 0, -1):
                    vv = trace[dd]
                    kk = x - y
                    pk = kk + 1 if kk == -dd or (kk != dd and vv[kk - 1] < vv[kk + 1]) else kk - 1
                    px = vv[pk]; py = px - pk
                    while x > px and y > py and x > 0 and y > 0 and a[x - 1] == b[y - 1]:
                        x -= 1; y -= 1; pairs.append((x, y))
                    x, y = px, py
                while x > 0 and y > 0:                            # the snake of d = 0
                    x -= 1; y -= 1; pairs.append((x, y))
                return pairs[::-1]
    return None


class Differencer:
    """Per-trace state of the open occurrences; `feed` returns the transitions of one request."""

    def __init__(self, budget=64, ckpt=32):
        self.budget, self.ckpt = budget, ckpt
        self.occ = np.empty(0, np.int64); self.ord = np.empty(0, np.uint64); self.ids = np.empty(0, np.int64)
        self.next_occ = self.calls = self.fallbacks = 0

    def feed(self, ids):
        """ids: message IDs of the request, in order. Returns (transitions as a list of (op, occ, msg_id, ord), checkpoint or None)."""
        ids = np.asarray(ids, np.int64)
        old, n, m = self.ids, len(self.ids), len(ids)
        lim = min(n, m)
        neq = np.flatnonzero(old[:lim] != ids[:lim])
        p = int(neq[0]) if neq.size else lim                      # common prefix
        lim = min(n, m) - p
        neq = np.flatnonzero(old[n - lim:][::-1] != ids[m - lim:][::-1]) if lim else np.empty(0, np.int64)
        s = int(neq[0]) if neq.size else lim                      # common suffix of the rest
        a, b = old[p:n - s], ids[p:m - s]
        pairs = lcs_pairs(a.tolist(), b.tolist(), self.budget)
        if pairs is None:                                         # over budget: close the whole middle and reopen it
            pairs = []; self.fallbacks += 1
        kept_old = np.zeros(len(a), bool); kept_new = np.full(len(b), -1)
        for i, j in pairs:
            kept_old[i] = True; kept_new[j] = i
        out = [(-1, int(self.occ[p + i]), int(a[i]), int(self.ord[p + i])) for i in np.flatnonzero(~kept_old)]
        # order keys of the new middle: kept occurrences keep theirs; a run of new ones is spread between its neighbours
        occ_mid, ord_mid = np.empty(len(b), np.int64), np.empty(len(b), np.uint64)
        lo = int(self.ord[p - 1]) if p else 0                     # key just before the middle
        j = 0
        while j < len(b):
            if kept_new[j] >= 0:
                i = p + int(kept_new[j]); occ_mid[j], ord_mid[j] = self.occ[i], self.ord[i]; lo = int(self.ord[i]); j += 1
                continue
            e = j
            while e < len(b) and kept_new[e] < 0:
                e += 1
            if e < len(b):
                hi = int(self.ord[p + int(kept_new[e])])
            elif s:
                hi = int(self.ord[n - s])
            else:
                hi = None                                         # appending at the end
            k = e - j
            if hi is None:
                keys = [lo + GAP * (t + 1) for t in range(k)]
            elif hi - lo > k:
                keys = [lo + (hi - lo) * (t + 1) // (k + 1) for t in range(k)]
            else:                                                 # no room between the neighbours: relabel everything after `lo`
                return self._relabel(ids, p)
            for t in range(k):
                occ_mid[j + t], ord_mid[j + t] = self.next_occ, keys[t]
                out.append((1, self.next_occ, int(b[j + t]), keys[t])); self.next_occ += 1
            lo = keys[-1]; j = e
        self.occ = np.concatenate([self.occ[:p], occ_mid, self.occ[n - s:]])
        self.ord = np.concatenate([self.ord[:p], ord_mid, self.ord[n - s:]])
        self.ids = ids
        return out, self._checkpoint()

    def _relabel(self, ids, p):
        """Close every occurrence from position p on and reopen the new tail with fresh, evenly spaced keys."""
        out = [(-1, int(o), int(i), int(k)) for o, i, k in zip(self.occ[p:], self.ids[p:], self.ord[p:])]
        lo, k = (int(self.ord[p - 1]) if p else 0), len(ids) - p
        keys = [lo + GAP * (t + 1) for t in range(k)]
        occs = list(range(self.next_occ, self.next_occ + k)); self.next_occ += k
        out += [(1, o, int(i), key) for o, i, key in zip(occs, ids[p:], keys)]
        self.occ = np.concatenate([self.occ[:p], np.array(occs, np.int64)])
        self.ord = np.concatenate([self.ord[:p], np.array(keys, np.uint64)])
        self.ids = ids; self.fallbacks += 1
        return out, self._checkpoint()

    def _checkpoint(self):
        self.calls += 1
        if (self.calls - 1) % self.ckpt == 0:
            return self.occ.copy(), self.ord.copy(), self.ids.copy()
        return None


class Dictionary:
    """Message dictionary: serialized message -> dense ID. A message is looked up by its 128-bit content hash; on a hit the
    stored bytes are compared with the new ones, so two different messages never share an ID. The bytes live in an
    append-only file; memory holds one small entry per distinct message."""

    def __init__(self, path):
        self.fd, self.index, self.size = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644), {}, 0
        self.loc = []                                             # message ID -> (offset, length) in the byte file

    def intern(self, b):
        """Returns (message ID, whether the message is new)."""
        h = xxhash.xxh3_128_digest(b)
        e = self.index.get(h)
        if e is None:
            self.index[h] = (len(self.index), self.size, len(b)); self.loc.append((self.size, len(b)))
            os.write(self.fd, b); self.size += len(b)
            return len(self.index) - 1, True
        if e[2] != len(b) or os.pread(self.fd, e[2], e[1]) != b:
            raise ValueError("two different messages share a 128-bit hash")
        return e[0], False

    def read(self, x):
        return os.pread(self.fd, self.loc[x][1], self.loc[x][0])

    def __len__(self):
        return len(self.index)


K, EDIT, MIN_VARIANT = 64, 256, 1024      # window context; longest inserted or deleted run; shortest message recorded as a variant


def splice(a, b):
    """b as a small edit of a: (p, s) such that b = a[:p] + b[p:len(b)-s] + a[len(a)-s:] with at most EDIT bytes deleted from a
    and at most EDIT bytes inserted; None if b is not such an edit."""
    n, m = len(a), len(b)
    if abs(n - m) > EDIT:
        return None
    x, y = np.frombuffer(a, np.uint8), np.frombuffer(b, np.uint8)
    lim = min(n, m)
    d = np.flatnonzero(x[:lim] != y[:lim])
    p = int(d[0]) if d.size else lim
    lim -= p
    d = np.flatnonzero(x[n - lim:][::-1] != y[m - lim:][::-1]) if lim else d[:0]
    s = int(d[0]) if d.size else lim
    return (p, s) if n - p - s <= EDIT and m - p - s <= EDIT else None


def windows(a, b, p, s):
    """The bytes of the variant b and of its base a around the edit, with K bytes of context on each side."""
    return b[max(0, p - K):len(b) - s + K], a[max(0, p - K):len(a) - s + K]


SIG, RESYNC = 64, 8                       # bytes of a message that identify where the previous request is taken up again; messages tried


class Ingester:
    """Online ingestion of raw requests: the dictionary plus one Differencer per active trace. Requests of different traces
    may arrive interleaved; the requests of one trace arrive in call order. Memory is the dictionary index, the open
    contexts of the active traces and, for request_text(), the previous request of every active trace."""

    def __init__(self, dictionary, budget=64, ckpt=32):
        self.dict, self.budget, self.ckpt, self.active, self.fallbacks = dictionary, budget, ckpt, {}, 0
        self.base = {}                                            # variant ID -> ID of the message it is a small edit of
        self.prev = {}                                            # trace -> (request bytes, message IDs, offset of the separator after each message)

    def request(self, trace, messages):
        """messages: the serialized messages of one request, in order.
        Returns (call number within the trace, message IDs, new messages as (id, bytes), variants, transitions, checkpoint or None).
        A variant is a new message that is a small edit of a message this request removes from the context (a cache mark that
        moved, a line that changed): (id, base id, p, s, inserted bytes, variant window, base window)."""
        ids, new = [], []
        for b in messages:
            x, fresh = self.dict.intern(b)
            ids.append(x)
            if fresh:
                new.append((x, b))
        return self._diff(trace, ids, new)

    def request_text(self, trace, text):
        """One request as raw bytes: a JSON array of messages in canonical form (sorted keys, no whitespace).
        The bytes are compared with the previous request of the trace first. A message whose bytes lie inside a run that equals
        the previous request is the same message: it keeps its ID and is neither parsed nor hashed. Where the two requests
        part, the previous request is looked for again further on (messages were dropped, replaced or inserted), and only the
        bytes in between are parsed. Returns what request() returns, then the roles of the new messages and the number of
        bytes that were parsed."""
        n, ids, ends, new, roles, parsed = len(text), [], [], [], [], 0
        if text[:1] != b"[" or text[-1:] != b"]":
            raise ValueError("a request is a JSON array")

        def parse(lo, hi):                                        # the messages in text[lo:hi]; text[hi] is the separator after them
            nonlocal parsed
            seg = text[lo:hi]
            msgs = orjson.loads(b"[" + seg + b"]")
            raw = [orjson.dumps(m, option=orjson.OPT_SORT_KEYS) for m in msgs]
            if b",".join(raw) != seg:
                raise ValueError("the request is not in canonical form")
            parsed += hi - lo
            for m, b in zip(msgs, raw):
                x, fresh = self.dict.intern(b)
                ids.append(x); lo += len(b); ends.append(lo); lo += 1
                if fresh:
                    new.append((x, b)); roles.append(m.get("role", "") if isinstance(m, dict) else "")

        pos, st = 1, self.prev.get(trace)                         # pos: offset of the next message of this request
        if st is not None and n > 2:
            ptext, pids, pends = st
            x, y, k, m = np.frombuffer(text, np.uint8), np.frombuffer(ptext, np.uint8), 0, len(pids)
            start = lambda t: int(pends[t - 1]) + 1 if t else 1   # where message t of the previous request starts
            while pos < n and k < m:
                ps = start(k)
                lim = min(n - pos, len(ptext) - ps)
                diff = np.flatnonzero(x[pos:pos + lim] != y[ps:ps + lim])
                run = int(diff[0]) if diff.size else lim
                j = int(np.searchsorted(pends[k:], ps + run, side="right"))      # whole messages inside the equal run
                if j:
                    ids += pids[k:k + j]; ends += (pends[k:k + j] + (pos - ps)).tolist()
                    pos, k = ends[-1] + 1, k + j
                    continue
                # the two requests part here: find where the previous request is taken up again
                sig = lambda t: ptext[start(t):min(start(t) + SIG, int(pends[t]))]
                hit = None                                        # (offset at which message t continues in this request, t)
                for t in range(k + 1, min(k + 1 + RESYNC, m)):    # messages k..t-1 were dropped
                    if text.startswith(sig(t), pos):
                        hit = (pos, t); break
                if hit is None:                                   # new messages first, then the previous request from message t on
                    for t in range(k, min(k + RESYNC, m)):
                        q = text.find(b"," + sig(t), pos)
                        if q >= 0 and (hit is None or q + 1 < hit[0]):
                            hit = (q + 1, t)
                if hit is None:
                    break
                if hit[0] > pos:
                    try:
                        parse(pos, hit[0] - 1)
                    except orjson.JSONDecodeError:                # the bytes found are not at a message boundary
                        break
                pos, k = hit
        if pos < n and n > 2:
            parse(pos, n - 1)
        out = self._diff(trace, ids, new)
        self.prev[trace] = (text, ids, np.array(ends, np.int64))
        return (*out, roles, parsed)

    def _diff(self, trace, ids, new):
        d = self.active.get(trace)
        if d is None:
            d = self.active[trace] = Differencer(self.budget, self.ckpt)
        trans, ck = d.feed(ids)
        variants = []
        gone = {self.base.get(m, m) for op, _, m, _ in trans if op == -1} if any(len(b) >= MIN_VARIANT for _, b in new) else ()
        for x, b in new:
            best = None
            for r in gone if len(b) >= MIN_VARIANT else ():
                if abs(self.dict.loc[r][1] - len(b)) <= EDIT:
                    a = self.dict.read(r)
                    ps = splice(a, b)
                    if ps and (best is None or sum(ps) > best[1] + best[2]):
                        best = (r, *ps, a)
            if best:
                r, p, s, a = best
                self.base[x] = r
                variants.append((x, r, p, s, b[p:len(b) - s], *windows(a, b, p, s)))
        return d.calls - 1, ids, new, variants, trans, ck

    def end(self, trace):
        """The trace is complete: drop its state."""
        self.fallbacks += self.active.pop(trace).fallbacks
        self.prev.pop(trace, None)
