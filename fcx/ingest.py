"""FCX ingestion: turn the raw request of every LLM call into an append-only transition log.

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
IMPL = "v2"                                                       # E-FAST implementation (list state, C-level comparisons); v1: ingest_v1.py


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


U64 = 1 << 64


def common_prefix(a, b, lim):
    """Length of the common prefix of the lists a and b within their first lim items, by C-level slice comparisons."""
    if a[:lim] == b[:lim]:
        return lim
    lo, hi = 0, lim                                               # a[:lo] == b[:lo] and a[:hi] != b[:hi]
    while hi - lo > 16:
        mid = (lo + hi) >> 1
        if a[lo:mid] == b[lo:mid]:
            lo = mid
        else:
            hi = mid
    while a[lo] == b[lo]:
        lo += 1
    return lo


def common_suffix(a, b, lim):
    """Length of the common suffix of the lists a and b within their last lim items."""
    n, m = len(a), len(b)
    if lim == 0 or a[n - lim:] == b[m - lim:]:
        return lim
    lo, hi = 0, lim                                               # the last lo items agree, the last hi do not
    while hi - lo > 16:
        mid = (lo + hi) >> 1
        if a[n - mid:n - lo] == b[m - mid:m - lo]:
            lo = mid
        else:
            hi = mid
    while a[n - 1 - lo] == b[m - 1 - lo]:
        lo += 1
    return lo


class Differencer:
    """Per-trace state of the open occurrences; `feed` returns the transitions of one request.
    The state is kept in Python lists and compared with C-level list slices; numpy arrays are built only for checkpoints."""

    def __init__(self, budget=64, ckpt=32):
        self.budget, self.ckpt = budget, ckpt
        self.occ, self.ord, self.ids = [], [], []
        self.next_occ = self.calls = self.fallbacks = 0

    def feed(self, ids):
        """ids: message IDs of the request, in order. Returns (transitions as a list of (op, occ, msg_id, ord), checkpoint or None)."""
        ids = ids[:] if isinstance(ids, list) else [int(x) for x in ids]   # own copy: callers may reuse their list
        old, occ, ordk = self.ids, self.occ, self.ord
        n, m = len(old), len(ids)
        p = common_prefix(old, ids, min(n, m))
        s = common_suffix(old[p:], ids[p:], min(n, m) - p)
        a, b = old[p:n - s], ids[p:m - s]
        pairs = lcs_pairs(a, b, self.budget)
        if pairs is None:                                         # over budget: close the whole middle and reopen it
            pairs = []; self.fallbacks += 1
        kept_old = [False] * len(a); kept_new = [-1] * len(b)
        for i, j in pairs:
            kept_old[i] = True; kept_new[j] = i
        out = [(-1, occ[p + i], a[i], ordk[p + i]) for i in range(len(a)) if not kept_old[i]]
        # order keys of the new middle: kept occurrences keep theirs; a run of new ones is spread between its neighbours
        occ_mid, ord_mid = [0] * len(b), [0] * len(b)
        lo = ordk[p - 1] if p else 0                              # key just before the middle
        j = 0
        while j < len(b):
            if kept_new[j] >= 0:
                i = p + kept_new[j]; occ_mid[j], ord_mid[j] = occ[i], ordk[i]; lo = ordk[i]; j += 1
                continue
            e = j
            while e < len(b) and kept_new[e] < 0:
                e += 1
            if e < len(b):
                hi = ordk[p + kept_new[e]]
            elif s:
                hi = ordk[n - s]
            else:
                hi = None                                         # appending at the end
            k = e - j
            if hi is None:
                keys = [lo + GAP * (t + 1) for t in range(k)]
                if keys[-1] >= U64:
                    raise OverflowError("order key beyond 64 bits")
            elif hi - lo > k:
                keys = [lo + (hi - lo) * (t + 1) // (k + 1) for t in range(k)]
            else:                                                 # no room between the neighbours: relabel everything after `lo`
                return self._relabel(ids, p)
            nxt = self.next_occ
            for t in range(k):
                occ_mid[j + t], ord_mid[j + t] = nxt + t, keys[t]
                out.append((1, nxt + t, b[j + t], keys[t]))
            self.next_occ = nxt + k
            lo = keys[-1]; j = e
        self.occ = occ[:p] + occ_mid + occ[n - s:]
        self.ord = ordk[:p] + ord_mid + ordk[n - s:]
        self.ids = ids
        return out, self._checkpoint()

    def _relabel(self, ids, p):
        """Close every occurrence from position p on and reopen the new tail with fresh, evenly spaced keys."""
        out = list(zip([-1] * (len(self.occ) - p), self.occ[p:], self.ids[p:], self.ord[p:]))
        lo, k = (self.ord[p - 1] if p else 0), len(ids) - p
        keys = [lo + GAP * (t + 1) for t in range(k)]
        if keys and keys[-1] >= U64:
            raise OverflowError("order key beyond 64 bits")
        occs = list(range(self.next_occ, self.next_occ + k)); self.next_occ += k
        out += list(zip([1] * k, occs, ids[p:], keys))
        self.occ = self.occ[:p] + occs
        self.ord = self.ord[:p] + keys
        self.ids = ids; self.fallbacks += 1
        return out, self._checkpoint()

    def state_bytes(self):
        """Payload bytes of the open state at 8 bytes per occurrence, key and message ID (as the int64/uint64 arrays used before)."""
        return 8 * (len(self.occ) + len(self.ord) + len(self.ids))

    def _checkpoint(self):
        self.calls += 1
        if (self.calls - 1) % self.ckpt == 0:
            return np.array(self.occ, np.int64), np.array(self.ord, np.uint64), np.array(self.ids, np.int64)
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


def recurring(text, pos, pmv, pends, k, m):
    """The number j of whole messages k..k+j-1 of the previous request that recur byte for byte, in order, at text[pos:].
    pmv: memoryview of the previous request; pends[t]: offset of the separator after its message t. A prefix of messages
    recurs only if every shorter prefix does, so j is found by bisection with C-level comparisons (bytes.startswith)."""
    ps = pends[k - 1] + 1 if k else 1
    if text.startswith(pmv[ps:pends[m - 1]], pos):
        return m - k
    if not text.startswith(pmv[ps:pends[k]], pos):
        return 0
    lo, hi = 1, m - k                                             # the first lo messages recur, the first hi do not
    while hi - lo > 1:
        mid = (lo + hi) >> 1
        if text.startswith(pmv[ps:pends[k + mid - 1]], pos):
            lo = mid
        else:
            hi = mid
    return lo


SIG, RESYNC = 64, 8                       # bytes of a message that identify where the previous request is taken up again; messages tried


class Ingester:
    """Online ingestion of raw requests: the dictionary plus one Differencer per active trace. Requests of different traces
    may arrive interleaved; the requests of one trace arrive in call order. Memory is the dictionary index, the open
    contexts of the active traces and, for request_text(), the previous request of every active trace."""

    def __init__(self, dictionary, budget=64, ckpt=32):
        self.dict, self.budget, self.ckpt, self.active, self.fallbacks = dictionary, budget, ckpt, {}, 0
        self.base = {}                                            # variant ID -> ID of the message it is a small edit of
        self.prev = {}                                            # trace -> (request bytes, message IDs, offsets of the separator after each message; lists)

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
            ptext, pids, pends = st                               # pends: list of separator offsets of the previous request
            pmv, k, m = memoryview(ptext), 0, len(pids)
            start = lambda t: pends[t - 1] + 1 if t else 1        # where message t of the previous request starts
            while pos < n and k < m:
                ps = start(k)
                j = recurring(text, pos, pmv, pends, k, m)        # whole messages that recur unchanged
                if j:
                    off = pos - ps
                    ids += pids[k:k + j]; ends += [e + off for e in pends[k:k + j]]
                    pos, k = ends[-1] + 1, k + j
                    continue
                # the two requests part here: find where the previous request is taken up again
                sig = lambda t: ptext[start(t):min(start(t) + SIG, pends[t])]
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
        self.prev[trace] = (text, ids, ends)
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
