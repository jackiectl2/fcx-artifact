"""Ingestion baselines and ablations on the same raw requests (ICDE supplementary experiment E-ING).

Every variant reads the requests of one manager of one capture directory in the same arrival order (ingest_run.arrivals:
--interleave traces at a time, seeded) and derives its log rows from each request. The timed region of a request covers the
variant's own work: parsing, hashing, dictionary lookup, differencing, and appending the rows it produces to an in-memory log.
Reading the input, writing files and all checks are outside it, so the timings are derivation times, not end-to-end
ingestion times. After a variant's timed pass, a separate untimed pass replays its log against the input and compares the
context of every call with the request byte for byte; the variant's dictionary is then deleted.

Exact variants (expected to reproduce every request)
  fcx_b<B>     FCX as in the paper: delta parsing against the previous request, then the differencer (common prefix and suffix,
               Myers alignment of the rest with edit budget B, 64-bit order keys). B = 64 is the paper's setting.
  fcx_full     the same differencer (B = 64) on fully parsed requests: the ablation of delta parsing.
  nokey        FCX (delta parsing, same alignment, B = 64) with dense positions instead of order keys: a kept occurrence whose
               position changes is closed and reopened at its new position.
  hash_array   direct versioning baseline: every call is a version that stores the ordered array of message IDs (parse and
               hash every message; each distinct message stored once).
  hash_array_delta  the same arrays derived with FCX's delta parser and dictionary (Ingester.request_text): unchanged messages
               keep their IDs without parsing; the request's IDs are stored as the call's array instead of being differenced
               (no occurrence log, no small-edit variants). It isolates the cost of differencing from the cost of parsing.
  pos_diff     positional delta baseline: parse and hash every message and compare the request with the previous one position
               by position; every position whose message changed is closed and reopened.
Lossy diagnostic baselines (not expected to be exact; they show what a content-addressed set difference loses)
  set_diff     set difference of message IDs against the previous request; context = open set ordered by open time.
  mset_lifo    the same with multisets (duplicate counts kept); a removed copy is the most recently opened one.

Per-request slices, from the replayed fcx_b64 log: dup (the request holds a message twice), same_content_reopen (FCX closes
and reopens the same content in this request: a reorder, a budget fallback or a key relabelling), return (a message that left the context earlier in the trace comes back), fallback (the
differencer exceeded its edit budget or relabelled keys). Every variant reports its wrong requests per slice.

Sizes: `records` are rows of the variant's own schema (transition rows; one version record per call for hash_array, whose
array elements are counted in `array_elems`). `payload_bytes` use fixed widths (uncompressed): FCX transition 25 B (trace
u32, span u32, op i8, occ u32, msg u32, ord u64); dense-position row 17 B (trace, span, op, pos u32, msg u32); set/multiset row
13 B (trace, span, op, msg); hash_array 4 B per array element plus 8 B per call (span u32, length u32). Checkpoints are
reported separately in `checkpoint_bytes`: FCX 16 B per element (occ, ord, msg), nokey 4 B (msg; positions implicit).
Dictionary bytes are the same for every variant and reported once.

usage: ingest_ablate.py --data <capture dir> --manager <m> --work <node-local dir> --out <json> [--budgets 0,2,8,64,256]
"""
import argparse, itertools, json, os, random, resource, time
from collections import Counter
from contextlib import contextmanager

import numpy as np, orjson, pyarrow.parquet as pq, xxhash

import ingest
from ingest import Dictionary, Ingester, lcs_pairs
from ingest_run import arrivals

SORT = orjson.OPT_SORT_KEYS
DUP, REOPEN, RETURN, FALLBACK = 1, 2, 4, 8
SLICES = dict(dup=DUP, same_content_reopen=REOPEN, ret=RETURN, fallback=FALLBACK)
W_FCX, W_CKPT, W_POS, W_SET, W_NK_CKPT = 25, 16, 17, 13, 4


def full_parse(text):
    msgs = orjson.loads(text)
    raw = [orjson.dumps(m, option=SORT) for m in msgs]
    if b"[" + b",".join(raw) + b"]" != text:
        raise ValueError("the request is not in canonical form")
    return raw


class NoKeyDifferencer:
    """FCX's alignment (common prefix, common suffix, Myers on the rest with an edit budget) over dense positions. Rows are
    (op, pos, msg_id, pos): closes first, then opens, so a position freed by a close can be reopened in the same request.
    Every `ckpt` calls it copies the context, as FCX's differencer does (message IDs only; positions are implicit)."""

    def __init__(self, budget=64, ckpt=32):
        self.budget, self.ckpt, self.ids, self.calls, self.fallbacks = budget, ckpt, np.empty(0, np.int64), 0, 0

    def feed(self, ids):
        ids = np.asarray(ids, np.int64)
        old, n, m = self.ids, len(self.ids), len(ids)
        lim = min(n, m)
        neq = np.flatnonzero(old[:lim] != ids[:lim])
        p = int(neq[0]) if neq.size else lim
        lim = min(n, m) - p
        neq = np.flatnonzero(old[n - lim:][::-1] != ids[m - lim:][::-1]) if lim else np.empty(0, np.int64)
        s = int(neq[0]) if neq.size else lim
        pairs = lcs_pairs(old[p:n - s].tolist(), ids[p:m - s].tolist(), self.budget)
        if pairs is None:
            pairs = []; self.fallbacks += 1
        new_of_old = np.full(n, -1, np.int64)                     # new position of each kept old occurrence
        new_of_old[:p] = np.arange(p)
        if s:
            new_of_old[n - s:] = np.arange(m - s, m)
        for i, j in pairs:
            new_of_old[p + i] = p + j
        stay = new_of_old == np.arange(n)                         # kept at the same position: no row
        close = np.flatnonzero(~stay)
        stayed_new = np.zeros(m, bool); stayed_new[new_of_old[stay]] = True
        opn = np.flatnonzero(~stayed_new)
        rows = [(-1, int(k), int(old[k]), int(k)) for k in close] + [(1, int(k), int(ids[k]), int(k)) for k in opn]
        self.ids = ids; self.calls += 1
        ck = (ids.copy(),) if (self.calls - 1) % self.ckpt == 0 else None    # dense context: positions are implicit
        return rows, ck


@contextmanager
def differencer(cls):
    saved = ingest.Differencer
    ingest.Differencer = cls
    try:
        yield
    finally:
        ingest.Differencer = saved


class Stream:
    """The request stream of one manager, with the checks that every pass sees the same, complete stream in call order."""

    def __init__(self, a):
        self.a = a
        sp = pq.read_table(f"{a.data}/spans.parquet", columns=["trace_id", "span_seq"])
        self.trace_of, self.seq_of = sp["trace_id"].to_numpy(), sp["span_seq"].to_numpy()
        self.fingerprint = None

    def __iter__(self):
        src = pq.ParquetFile(f"{self.a.data}/{self.a.manager}/wide.parquet")
        h, nxt, n = xxhash.xxh3_64(), {}, 0
        for trace, span, text, last in arrivals(src, self.trace_of, self.a.interleave, random.Random(self.a.seed)):
            k = nxt.get(trace, 0)
            if self.seq_of[span] != k:
                raise AssertionError(f"call order broken: trace {trace} span {span} seq {self.seq_of[span]} expected {k}")
            nxt[trace] = k + 1
            h.update(np.array([trace, span], np.int64).tobytes()); n += 1
            yield trace, span, text, last
            if last:
                del nxt[trace]
        if n != len(self.trace_of) or nxt:
            raise AssertionError(f"incomplete stream: {n} of {len(self.trace_of)} requests, {len(nxt)} traces unfinished")
        fp = h.hexdigest()
        if self.fingerprint is None:
            self.fingerprint = fp
        elif fp != self.fingerprint:
            raise AssertionError("the request stream differs between passes")


def new_dict(a, name):
    path = f"{a.work}/dict-{name}.bin"
    d = Dictionary(path); d.path = path
    return d


def drop_dict(d):
    os.close(d.fd); os.remove(d.path)


def timed_fcx(a, stream, name, budget, delta, nokey=False):
    """One timed pass of FCX or of the no-order-key ablation. Returns stats, the per-request logs and the dictionary."""
    dic = new_dict(a, name)
    ing = Ingester(dic, budget, a.ckpt)
    log, fb_flag = [], []
    st = dict(requests=0, records=0, checkpoint_elems=0, parsed_bytes=0, logical_bytes=0, cpu_s=0.0, wall_s=0.0)
    with differencer(NoKeyDifferencer if nokey else ingest.Differencer):
        for trace, span, text, last in stream:
            fb0 = ing.fallbacks + (ing.active[trace].fallbacks if trace in ing.active else 0)
            c0, w0 = time.process_time(), time.perf_counter()
            if delta:
                _, ids, _, _, trans, ck, _, n_parsed = ing.request_text(trace, text)
            else:
                _, ids, _, _, trans, ck = ing.request(trace, full_parse(text)); n_parsed = len(text)
            log.append((trace, span, trans))
            w1 = time.perf_counter(); c1 = time.process_time()
            st["cpu_s"] += c1 - c0; st["wall_s"] += w1 - w0
            fb_flag.append(ing.active[trace].fallbacks + ing.fallbacks > fb0)
            st["requests"] += 1; st["records"] += len(trans); st["parsed_bytes"] += n_parsed; st["logical_bytes"] += len(text)
            st["checkpoint_elems"] += len(ck[0]) if ck is not None else 0
            if last:
                ing.end(trace)
    if ing.active:
        raise AssertionError("traces left open after the stream")
    st["fallbacks"] = ing.fallbacks
    st["fallback_requests"] = int(sum(fb_flag))
    if nokey:
        st["payload_bytes"], st["checkpoint_bytes"] = st["records"] * W_POS, st["checkpoint_elems"] * W_NK_CKPT
    else:
        st["payload_bytes"], st["checkpoint_bytes"] = st["records"] * W_FCX, st["checkpoint_elems"] * W_CKPT
    return st, log, np.array(fb_flag, bool), dic


class ArrayIngester(Ingester):
    """hash_array_delta: FCX's request_text (delta parsing against the previous request, dictionary lookups), but the message IDs
    of each request become the call's array; nothing is differenced."""

    def __init__(self, dictionary, ckpt=32):
        super().__init__(dictionary, 64, ckpt)
        self.call = {}

    def _diff(self, trace, ids, new):
        c = self.call[trace] = self.call.get(trace, -1) + 1
        return c, ids, new, [], np.array(ids, np.uint32), None

    def end(self, trace):
        self.call.pop(trace, None); self.prev.pop(trace, None)


def timed_array_delta(a, stream):
    dic = new_dict(a, "hash_array_delta")
    ing, log = ArrayIngester(dic, a.ckpt), []
    st = dict(requests=0, records=0, array_elems=0, parsed_bytes=0, logical_bytes=0, cpu_s=0.0, wall_s=0.0)
    for trace, span, text, last in stream:
        c0, w0 = time.process_time(), time.perf_counter()
        _, ids, _, _, arr, _, _, n_parsed = ing.request_text(trace, text)
        log.append((trace, span, arr))
        w1 = time.perf_counter(); c1 = time.process_time()
        st["cpu_s"] += c1 - c0; st["wall_s"] += w1 - w0
        st["requests"] += 1; st["records"] += 1; st["array_elems"] += len(arr); st["parsed_bytes"] += n_parsed; st["logical_bytes"] += len(text)
        if last:
            ing.end(trace)
    if ing.call or ing.prev:
        raise AssertionError("array state left after the stream")
    st["payload_bytes"] = 4 * st["array_elems"] + 8 * st["records"]
    return st, log, dic


def derive(kind, state, trace, ids):
    """Rows of one request under a baseline. `state` holds the per-trace state; the caller drops a trace's entry at its end.
    Returns (rows, count): transition rows for the diff baselines, array elements for hash_array."""
    if kind == "hash_array":
        return np.array(ids, np.uint32), len(ids)
    if kind == "set_diff":
        open_ = state.setdefault(trace, set())
        cur = dict.fromkeys(ids)                                  # distinct IDs in request order
        closes = [m for m in open_ if m not in cur]
        opens = [m for m in cur if m not in open_]
        open_.difference_update(closes); open_.update(opens)
        rows = [(-1, m) for m in closes] + [(1, m) for m in opens]
        return rows, len(rows)
    if kind == "mset_lifo":
        have = state.setdefault(trace, Counter())
        cnt = Counter(ids)
        rows = [(-1, m) for m, c in have.items() for _ in range(c - cnt.get(m, 0))]
        add = Counter()
        for m in ids:                                             # opens in request order
            if have.get(m, 0) + add[m] < cnt[m]:
                add[m] += 1; rows.append((1, m))
        have.subtract(Counter(m for op, m in rows if op == -1)); have.update(add)
        state[trace] = +have                                      # drop zero counts
        return rows, len(rows)
    if kind == "pos_diff":
        prev = state.get(trace, [])
        lim = min(len(prev), len(ids))
        ch = [k for k in range(lim) if prev[k] != ids[k]]
        rows = [(-1, k, prev[k], k) for k in ch] + [(-1, k, prev[k], k) for k in range(lim, len(prev))]
        rows += [(1, k, ids[k], k) for k in ch] + [(1, k, ids[k], k) for k in range(lim, len(ids))]
        state[trace] = list(ids)
        return rows, len(rows)
    raise ValueError(kind)


def timed_baseline(a, stream, kind):
    dic = new_dict(a, kind)
    state, log = {}, []
    st = dict(requests=0, records=0, array_elems=0, logical_bytes=0, cpu_s=0.0, wall_s=0.0)
    for trace, span, text, last in stream:
        c0, w0 = time.process_time(), time.perf_counter()
        ids = [dic.intern(b)[0] for b in full_parse(text)]
        rows, n_ev = derive(kind, state, trace, ids)
        log.append((trace, span, rows))
        w1 = time.perf_counter(); c1 = time.process_time()
        st["cpu_s"] += c1 - c0; st["wall_s"] += w1 - w0
        if kind == "hash_array":                                  # one version record per call holding the ID array
            st["records"] += 1; st["array_elems"] += n_ev
        else:
            st["records"] += n_ev
        st["requests"] += 1; st["logical_bytes"] += len(text)
        if last:
            state.pop(trace, None)
    if state:
        raise AssertionError("baseline state left after the stream")
    if kind == "hash_array":
        st["payload_bytes"] = 4 * st["array_elems"] + 8 * st["records"]
    else:
        st["payload_bytes"] = st["records"] * (W_POS if kind == "pos_diff" else W_SET)
    return st, log, dic


class KeyReplay:
    """Replay of (op, occ, msg_id, key) rows (FCX, nokey, pos_diff); checks op values, double opens, closes of occurrences
    that are not open, and that no two open occurrences share a key. Returns the context sorted by key."""

    def __init__(self):
        self.active = {}

    def apply(self, trace, rows):
        act = self.active.setdefault(trace, {})
        for op, occ, mid, key in rows:
            if op == 1:
                if occ in act:
                    raise AssertionError("occurrence opened twice")
                act[occ] = (key, mid)
            elif op == -1:
                if act.pop(occ, None) != (key, mid):
                    raise AssertionError("close of an occurrence that is not open")
            else:
                raise AssertionError(f"bad op {op}")
        keys = sorted(act.values())
        if len({k for k, _ in keys}) != len(keys):
            raise AssertionError("two open occurrences share an order key")
        return [m for _, m in keys]

    def end(self, trace):
        self.active.pop(trace, None)


class BaselineReplay:
    """Replay of the hash_array, set_diff and mset_lifo logs. The context of set_diff and mset_lifo is the open messages
    ordered by open time; mset_lifo closes the most recently opened copy."""

    def __init__(self, kind):
        self.kind, self.active, self.seq = kind, {}, itertools.count()

    def apply(self, trace, rows):
        if self.kind == "hash_array":
            return rows.tolist()
        o = self.active.setdefault(trace, {})
        for op, m in rows:
            if self.kind == "set_diff":
                if op == -1:
                    del o[m]
                elif m in o:
                    raise AssertionError("set opened twice")
                else:
                    o[m] = next(self.seq)
            elif op == -1:
                o[m].pop()
                if not o[m]:
                    del o[m]
            else:
                o.setdefault(m, []).append(next(self.seq))
        if self.kind == "set_diff":
            return sorted(o, key=o.get)
        return [m for _, m in sorted((s, m) for m, ss in o.items() for s in ss)]

    def end(self, trace):
        self.active.pop(trace, None)


def replayer(name):
    if name == "hash_array_delta":
        return BaselineReplay("hash_array")
    return BaselineReplay(name) if name in ("hash_array", "set_diff", "mset_lifo") else KeyReplay()


def validate(a, stream, name, log, dic, flags=None, fb_flag=None):
    """Untimed: replay the log, compare every call with its request byte for byte, and count wrong requests per slice.
    With fb_flag given (the fcx_b64 pass), also derive the per-request slice flags from the replayed contexts."""
    rep = replayer(name)
    derive_flags = fb_flag is not None
    out_flags = np.zeros(len(log), np.uint8) if derive_flags else None
    prev, hist = {}, {}
    wrong = np.zeros(len(log), bool)
    i = -1
    end = object()
    for i, (req, entry) in enumerate(itertools.zip_longest(stream, log, fillvalue=end)):
        if req is end or entry is end:
            raise AssertionError("log and stream have different lengths")
        (trace, span, text, last), (t2, s2, rows) = req, entry
        if (t2, s2) != (trace, span):
            raise AssertionError("log and stream out of step")
        ctx = rep.apply(trace, rows)
        wrong[i] = (b"[" + b",".join(dic.read(x) for x in ctx) + b"]") != text
        if derive_flags:
            if wrong[i]:
                raise AssertionError("fcx_b64 does not reproduce a request; slices cannot be derived")
            f = DUP if len(set(ctx)) != len(ctx) else 0
            closed = Counter(m for op, _, m, _ in rows if op == -1)
            opened = Counter(m for op, _, m, _ in rows if op == 1)
            f |= REOPEN if closed & opened else 0
            cur, p = set(ctx), prev.get(trace, set())
            h = hist.setdefault(trace, set())
            f |= RETURN if (cur - p) & h else 0
            f |= FALLBACK if fb_flag[i] else 0
            out_flags[i] = f
            h |= cur; prev[trace] = cur
        if last:
            rep.end(trace); prev.pop(trace, None); hist.pop(trace, None)
    if rep.active:
        raise AssertionError("replay state left after the stream")
    fl = out_flags if derive_flags else flags
    res = dict(wrong=int(wrong.sum()))
    res["by_slice"] = {k: dict(requests=int(((fl & b) > 0).sum()), wrong=int((wrong & ((fl & b) > 0)).sum())) for k, b in SLICES.items()}
    res["by_slice"]["none"] = dict(requests=int((fl == 0).sum()), wrong=int((wrong & (fl == 0)).sum()))
    return res, out_flags



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--manager", required=True)
    ap.add_argument("--work", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--budgets", default="0,2,8,64,256"); ap.add_argument("--ckpt", type=int, default=32)
    ap.add_argument("--interleave", type=int, default=64); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--array-delta", type=int, default=1, help="1: also time hash_array_delta (E-FAST), after the other variants")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    budgets = [int(b) for b in a.budgets.split(",")]
    if budgets[0] != 64:
        budgets = [64] + [b for b in budgets if b != 64]          # fcx_b64 first: it defines the slices
    stream = Stream(a)
    out = dict(data=a.data, manager=a.manager, interleave=a.interleave, seed=a.seed, ckpt=a.ckpt, variants={}, ingest_impl=getattr(ingest, "IMPL", "v1"))
    flags, t0 = None, time.perf_counter()

    def record(name, st, res):
        st.update(res)
        st["must_be_exact"] = name not in ("set_diff", "mset_lifo")
        st["records_per_request"] = round(st["records"] / st["requests"], 3)
        for k in ("cpu_s", "wall_s"):
            st[k] = round(st[k], 3)
        out["variants"][name] = st
        print(name, json.dumps(st), flush=True)

    plan = [(f"fcx_b{b}", dict(budget=b, delta=1)) for b in budgets] + [("fcx_full", dict(budget=64, delta=0)), ("nokey", dict(budget=64, delta=1, nokey=True))]
    for name, kw in plan:
        st, log, fb, dic = timed_fcx(a, stream, name, **kw)
        if name == "fcx_b64":
            out["dictionary_bytes"], out["distinct_messages"] = dic.size, len(dic)
            res, flags = validate(a, stream, name, log, dic, fb_flag=fb)
        else:
            res, _ = validate(a, stream, name, log, dic, flags=flags)
        drop_dict(dic); del log
        record(name, st, res)
    for kind in ("hash_array", "pos_diff", "set_diff", "mset_lifo"):
        st, log, dic = timed_baseline(a, stream, kind)
        res, _ = validate(a, stream, kind, log, dic, flags=flags)
        drop_dict(dic); del log
        record(kind, st, res)
    if a.array_delta:
        st, log, dic = timed_array_delta(a, stream)
        res, _ = validate(a, stream, "hash_array_delta", log, dic, flags=flags)
        drop_dict(dic); del log
        record("hash_array_delta", st, res)
    out["slice_requests"] = {k: int(((flags & b) > 0).sum()) for k, b in SLICES.items()}
    out["stream_fingerprint"] = stream.fingerprint
    out["requests"] = int(len(flags))
    out["total_wall_s"] = round(time.perf_counter() - t0, 1)
    out["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    tmp = a.out + ".tmp"
    json.dump(out, open(tmp, "w"), indent=1)
    bad = {k: v["wrong"] for k, v in out["variants"].items() if v["must_be_exact"] and v["wrong"]}
    if bad:
        raise SystemExit(f"exact variants failed to reproduce requests: {bad} (partial result kept in {tmp})")
    os.replace(tmp, a.out)
    print("ABLATE_OK", json.dumps({k: (v["records_per_request"], v["cpu_s"], v["wrong"]) for k, v in out["variants"].items()}))


if __name__ == "__main__":
    main()
