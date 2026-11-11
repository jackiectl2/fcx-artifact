"""FCX benchmark driver: load the ingested transition log into ClickHouse, build the access paths, check and time context queries.

The transition log D (with checkpoints) comes from the differencer (convert.py); it is the source of truth. Per policy <p>:
  D  d_<p>     Transition(trace_id, span_id, op, occ, msg_id, ord)        append-only log; ckpt_<p> holds the checkpoints
  I  i_<p>     Interval(msg_id, trace_id, occ, ord, span_from, span_to)   one row per message occurrence
  A  a_<p>     SpanCtx(span_id, msg_ids)                                  the context of every span, in order
  B, a_bf, a_tx                                                           baselines: bitmap postings, arrays with host indexes
  N  nf_<p>, ni_<p>   ContextItem(trace_id, span_id, pos, msg_id)          baseline: conventional normalized membership
I, B and A are built from D inside ClickHouse; N is built from the ingested requests. The arrays rebuilt from the log are
compared, span by span, with the requests that were ingested (arrays.parquet), every plan's span-ID set is compared with a
reference, and a mismatch stops the run. The SQL of every plan comes from the operator API in query.py.
Latencies are server-side elapsed times in milliseconds.
"""
import argparse, base64, ctypes, hashlib, json, math, os, platform, statistics, time, urllib.error, urllib.parse, urllib.request

import numpy as np

from query import ctx_any, ctx_count, ctx_items, ctx_sum, ctx_text, expand, matched, matched_variants, runs, slots

INT = "CODEC(Delta, ZSTD(1))"
ARR = f"span_id UInt32 {INT}, msg_ids Array(UInt32) {INT}"
DDL = {  # kind -> (columns, order by)
    "d": (f"trace_id UInt32 {INT}, span_id UInt32 {INT}, op Int8 CODEC(ZSTD(1)), occ UInt32 {INT}, msg_id UInt32 {INT}, ord UInt64 {INT}", "trace_id, span_id, occ"),
    "ckpt": (f"span_id UInt32 {INT}, occs Array(UInt32) {INT}, ords Array(UInt64) {INT}, msg_ids Array(UInt32) {INT}", "span_id"),
    "i": (f"msg_id UInt32 {INT}, trace_id UInt32 {INT}, occ UInt32 {INT}, ord UInt64 {INT}, span_from UInt32 {INT}, span_to UInt32 {INT}", "msg_id, span_from"),
    "b": (f"msg_id UInt32 {INT}, spans AggregateFunction(groupBitmap, UInt32) CODEC(ZSTD(1))", "msg_id"),
    "a": (ARR, "span_id"),
    "nf": (f"trace_id UInt32 {INT}, span_id UInt32 {INT}, pos UInt32 {INT}, msg_id UInt32 {INT}", "span_id, pos"),
    "ni": (f"trace_id UInt32 {INT}, span_id UInt32 {INT}, pos UInt32 {INT}, msg_id UInt32 {INT}", "msg_id, span_id"),
    "a_bf": (ARR + ", INDEX bf msg_ids TYPE bloom_filter(0.01) GRANULARITY 1", "span_id"),
    "a_tx": (ARR + ", INDEX tx arrayMap(x -> toString(x), msg_ids) TYPE text(tokenizer = array)", "span_id"),
    "w": ("span_id UInt32, input String CODEC(ZSTD(3)), INDEX tx input TYPE text(tokenizer = splitByNonAlpha)", "span_id"),
    "msg_s": ("msg_id UInt32, text String CODEC(ZSTD(3)), INDEX tx text TYPE text(tokenizer = splitByNonAlpha)", "msg_id"),
    "wx": ("span_id UInt32, input String CODEC(ZSTD(3))", "span_id"),          # index stage: the text index is added afterwards
    "wl": ("span_id UInt32, messages Array(LowCardinality(String)) CODEC(ZSTD(3))", "span_id"),     # the engine's own dictionary encoding of the messages
    "mx": ("msg_id UInt32, text String CODEC(ZSTD(3))", "msg_id"),
    "mxf": ("msg_id UInt32, first_span UInt32 CODEC(Delta, ZSTD(1)), text String CODEC(ZSTD(3))", "first_span, msg_id"),   # --dict-order first
}
INDEXES = {  # index stage: the same index kinds on the wide context column and on the message dictionary
    "none": None,
    "token": "text(tokenizer = splitByNonAlpha)",
    "ngrams3": "text(tokenizer = ngrams(3))",
    "ngrams4": "text(tokenizer = ngrams(4))",
    "sparsegrams": "text(tokenizer = sparseGrams)",
    "bloom4": "ngrambf_v1(4, 65536, 3, 0) GRANULARITY 1",
}
REGEX = {  # no quotes or backslashes: message text is canonical JSON
    "calib": ["[A-Za-z]+Error: [a-z]+", "line [0-9]+, in [a-z_]+", "[0-9]+ passed"],
    "eval": ["Ran [0-9]+ tests? in [0-9.]+s", "def test_[a-z0-9_]+", "import [a-z]+ as [a-z]+", "git (diff|status|log)",
             "[0-9]+ failed, [0-9]+ passed", "No such file or directory: [^ ]+"],
}
GRAN = {"d": 8192, "ckpt": 256, "i": 8192, "b": 8192, "a": 256, "a_bf": 256, "a_tx": 256, "nf": 8192, "ni": 8192}   # index_granularity; the calibration stage replaces these
CAND = {"d": [256, 1024, 8192], "i": [64, 256, 1024, 8192], "b": [16, 64, 256, 1024, 8192], "a": [16, 64, 256, 1024],
        "a_bf": [16, 64, 256, 1024], "a_tx": [16, 64, 256, 1024], "ckpt": [16, 64, 256]}
FILES = {  # converter output -> column types (stated explicitly: inferred Parquet columns would be Nullable)
    "messages": "msg_id UInt32, role String, n_chars UInt32, text String",
    "traces": "trace_id UInt32, span_lo UInt32, span_hi UInt32, n_msgs UInt32, repo String, exit_status String, resolved Int8",
    "spans": "span_id UInt32, trace_id UInt32, span_seq UInt32",
    "arrays": "span_id UInt32, msg_ids Array(UInt32)",
    "transitions": "trace_id UInt32, span_id UInt32, op Int8, occ UInt32, msg_id UInt32, ord UInt64",
    "checkpoints": "span_id UInt32, occs Array(UInt32), ords Array(UInt64), msg_ids Array(UInt32)",
    "wide": "span_id UInt32, input String",
    "variants": "msg_id UInt32, base_id UInt32, p UInt32, s UInt32, ins String, wv String, wb String",
}
SELECTIVITY = {"0.1": 1, "1": 10, "10": 100}                      # label (% of distinct messages) -> per mille
NEEDLES = {"t:traceback": "Traceback (most recent call last)", "t:nosuchfile": "No such file or directory", "t:assertionerror": "AssertionError",
           "t:importnumpy": "import numpy", "t:deftest": "def test_", "t:gitdiff": "git diff"}
TOKENS = ["Traceback", "AssertionError", "ModuleNotFoundError"]
AGG_NEEDLE, COUNT_TOKEN = "t:traceback", "Error"                  # the needle of the aggregate checks; the string ctx_sum[n_err] counts
# --workload wiki (E-WIKI, Simple English Wikipedia blocks), fixed before any Wikipedia sample was converted. The needles and the
# counted string serve the exactness checks only; the timed content predicates of the index stage are substrings cut from block
# text (sample_predicates), with no fixed regular expressions or needles. Stages that use SWE-specific predicates are refused.
WIKI_NEEDLES = {"w:citeweb": "{{cite web", "w:category": "[[Category:", "w:infobox": "{{Infobox", "w:ref": "<ref>", "w:file": "[[File:",
                "w:reflist": "{{reflist"}
WIKI_COUNT_TOKEN = "[["
WIKI_STAGES = {"base", "calib", "build", "any", "point", "agg", "bytes", "selfcheck", "life", "index"}
# --workload chat (E-CHAT, WildChat-1M conversations), fixed before any WildChat sample was converted. As for wiki: the needles and
# the counted string serve the exactness checks only, and the timed content predicates are substrings cut from message content
# (the "content" field of the canonical message JSON), with no fixed regular expressions or needles.
# Prefix "h:": matched-set names that start with "c" are calibration sets (stage_calib, stage_any, stage_route).
CHAT_NEEDLES = {"h:code": "```", "h:python": "python", "h:write": "Write", "h:please": "please", "h:continue": "continue", "h:story": "story"}
CHAT_COUNT_TOKEN = "the"
TEXT_FIELD = {"wiki": "text", "chat": "content"}                  # data-driven workloads: the JSON field substrings are cut from
EVICT = ["SYSTEM DROP MARK CACHE", "SYSTEM DROP UNCOMPRESSED CACHE", "SYSTEM DROP INDEX MARK CACHE", "SYSTEM DROP INDEX UNCOMPRESSED CACHE",
         "SYSTEM DROP TEXT INDEX CACHES", "SYSTEM DROP QUERY CONDITION CACHE", "SYSTEM UNLOAD PRIMARY KEY"]
TIMEOUT = 30                                                      # seconds; a plan this slow is out of the comparison


class CH:
    def __init__(self, port, threads, read_thresholds="zero"):
        self.url = f"http://127.0.0.1:{port}/?"
        self.hdr = {"X-ClickHouse-User": "default", "X-ClickHouse-Key": os.environ["FCX_CH_PW"]}
        self.base = {"max_threads": threads, "use_query_condition_cache": 0, "use_query_cache": 0, "wait_end_of_query": 1}
        # By default the server reads a table with one thread unless the read covers 163,840 rows or 240 MB. A wide table of
        # requests is above that at any size; a message dictionary or an interval table of a few thousand trajectories is not
        # (measured by stage diag: 108 ms with the defaults, as with one thread, against 32 ms with the thresholds at zero).
        # Every query of every side runs with both thresholds at zero, so that all tables are read with the same threads.
        if read_thresholds == "zero":
            self.base.update(merge_tree_min_rows_for_concurrent_read=0, merge_tree_min_bytes_for_concurrent_read=0)

    def q(self, sql, **settings):
        """Run one statement; return (text result, server-side summary, client wall seconds)."""
        req = urllib.request.Request(self.url + urllib.parse.urlencode({**self.base, **settings}), data=sql.encode(), headers=self.hdr)
        t = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=4 * 3600) as r:
                body, summ = r.read().decode().strip(), json.loads(r.headers.get("X-ClickHouse-Summary", "{}"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(e.read().decode()[:1500]) from None
        return body, {k: int(v) for k, v in summ.items()}, time.perf_counter() - t

    def val(self, sql):
        return self.q(sql)[0]


libc = ctypes.CDLL("libc.so.6", use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]


def src(name, policy=None):
    return f"file('{policy + '/' if policy else ''}{name}.parquet', Parquet, '{FILES[name]}')"


def data_files(store):
    """Open descriptors of the server's table files; the server may delete merged-away parts at any time."""
    for r, _, fs in os.walk(os.path.join(store, "store")):
        for f in fs:
            try:
                fd = os.open(os.path.join(r, f), os.O_RDONLY)
            except FileNotFoundError:
                continue
            yield fd
            os.close(fd)


def cached_fraction(store):
    """Fraction of the server's table files that sits in the OS page cache (mincore)."""
    hit = tot = 0
    for fd in data_files(store):
        size = os.fstat(fd).st_size
        if not size:
            continue
        addr = libc.mmap(None, size, 1, 1, fd, 0)                 # PROT_READ, MAP_SHARED
        if addr in (None, ctypes.c_void_p(-1).value):
            continue
        vec = np.zeros((size + 4095) // 4096, np.uint8)
        if libc.mincore(addr, size, vec.ctypes.data) == 0:
            hit += int((vec & 1).sum()) * 4096; tot += size
        libc.munmap(addr, size)
    return round(hit / tot, 4) if tot else None


def evict(ch, store):
    """Cold start: drop the server's caches and the OS page cache of its table files."""
    for s in list(EVICT):
        try:
            ch.q(s)
        except RuntimeError:
            EVICT.remove(s)                                       # statement unknown to this server version
    for fd in data_files(store):
        os.fsync(fd); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)


def over_limit(e):
    """A query stopped by the time or memory limit is a result (the plan is out); any other server error is a bug."""
    if "TIMEOUT_EXCEEDED" in str(e) or "MEMORY_LIMIT_EXCEEDED" in str(e):
        return str(e)[:300]
    raise e


def timed(ch, sql, store, cold, warm):
    out = {"cold_ms": [], "warm_ms": []}
    try:
        for key, n in (("cold_ms", cold), ("warm_ms", warm)):
            for _ in range(n):
                if key == "cold_ms":
                    evict(ch, store)
                val, s, wall = ch.q(sql, max_execution_time=TIMEOUT)
                out[key].append(round(s.get("elapsed_ns", wall * 1e9) / 1e6, 3))
        out.update(value=val, read_rows=s.get("read_rows"), read_bytes=s.get("read_bytes"), memory=s.get("memory_usage"), wall_ms=round(wall * 1e3, 3))
    except RuntimeError as e:
        out["error"] = over_limit(e)
    return out


def server_cpu(store):
    """CPU seconds (user + system) used so far by this job's ClickHouse server, found by its --config-file next to the data path."""
    mark, ticks = b"--config-file=" + os.path.dirname(store.rstrip("/")).encode(), 0
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            if mark in open(f"/proc/{pid}/cmdline", "rb").read():
                f = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
                ticks += int(f[11]) + int(f[12])                  # utime, stime
        except OSError:
            pass
    return ticks / os.sysconf("SC_CLK_TCK")


def save(res, path):
    with open(path + ".tmp", "w") as f:
        json.dump(res, f, indent=1)
    os.replace(path + ".tmp", path)


def load(ch, store, name, kind, select, gran=8192, optimize=True):
    """Create table `name` with the DDL of `kind`, fill it from a cold cache, merge it into one part, report size and time."""
    cols, order = DDL[kind]
    ch.q(f"DROP TABLE IF EXISTS {name} SYNC")
    ch.q(f"CREATE TABLE {name} ({cols}) ENGINE = MergeTree ORDER BY ({order}) SETTINGS index_granularity = {gran}, min_bytes_for_wide_part = 0")
    evict(ch, store)
    cpu = server_cpu(store)
    t_ins = ch.q(f"INSERT INTO {name} {select}", max_insert_threads=max(1, ch.base["max_threads"] // 2))[2]
    t_opt = ch.q(f"OPTIMIZE TABLE {name} FINAL")[2] if optimize else 0.0
    cpu = server_cpu(store) - cpu
    rows, comp, raw, disk, idx = map(int, ch.val(
        f"SELECT sum(rows), sum(data_compressed_bytes), sum(data_uncompressed_bytes), sum(bytes_on_disk), sum(secondary_indices_compressed_bytes) "
        f"FROM system.parts WHERE active AND database = currentDatabase() AND table = '{name}'").split("\t"))
    cols = dict(l.split("\t") for l in ch.val(
        f"SELECT column, sum(column_data_compressed_bytes) FROM system.parts_columns WHERE active AND database = currentDatabase() AND table = '{name}' GROUP BY column").splitlines())
    return dict(rows=rows, granularity=gran, compressed_bytes=comp, uncompressed_bytes=raw, bytes_on_disk=disk, index_bytes=idx,
                column_bytes={k: int(v) for k, v in cols.items()}, insert_s=round(t_ins, 2), optimize_s=round(t_opt, 2), cpu_s=round(cpu, 2))


def build_sql(kind, p, d):
    """SELECT that fills a table of `kind` for policy p: D and the checkpoints from the differencer's output, the rest from D."""
    if kind == "d":
        return f"SELECT trace_id, span_id, op, occ, msg_id, ord FROM {src('transitions', p)}"
    if kind == "ckpt":
        return f"SELECT span_id, occs, ords, msg_ids FROM {src('checkpoints', p)}"
    if kind == "i":
        return f"SELECT msg_id, trace_id, occ, ord, span_from, span_to FROM {runs(d)}"
    if kind == "b":
        return f"SELECT msg_id, {expand('span_from', 'span_to')} FROM {runs(d)} GROUP BY msg_id"
    if kind in ("nf", "ni"):                                      # one row per (request, message): what a normalized store ingests
        return (f"SELECT joinGet('span_trace', 'trace_id', span_id) AS trace_id, span_id, toUInt32(pos - 1) AS pos, msg_id "
                f"FROM {src('arrays', p)} ARRAY JOIN msg_ids AS msg_id, arrayEnumerate(msg_ids) AS pos")
    return (f"SELECT span_id, arrayMap(x -> x.2, arraySort(groupArray((ord, msg_id)))) FROM (SELECT toUInt32(arrayJoin(range(span_from, toUInt32(span_to + 1)))) AS span_id, "
            f"ord, msg_id FROM {runs(d)}) GROUP BY span_id")


def fail(res, a, why):
    save(res, a.out)
    raise SystemExit("MISMATCH: " + why)


def run_plans(ch, P, ref, a, tag, res):
    """Check every plan's span-ID set against the plan named `ref`, then time the cardinality of its set.
    A plan that returns a different set stops the run; a plan that exceeds the time limit is recorded and skipped."""
    ref_sql = f"SELECT {P[ref][0]} FROM {P[ref][1]}" if ref in P else ref     # a plan name, or a query returning the reference bitmap
    ch.q(f"CREATE OR REPLACE TABLE ref ENGINE = Memory AS SELECT ({ref_sql}) AS bm")
    card = int(ch.val("SELECT bitmapCardinality(bm) FROM ref"))
    recs = []
    for name, (agg, frm) in P.items():
        rec = dict(tag, plan=name, cardinality=card)
        try:
            diff = int(ch.q(f"WITH (SELECT {agg} FROM {frm}) AS x SELECT bitmapXorCardinality(x, (SELECT bm FROM ref))", max_execution_time=a.check_timeout)[0])
        except RuntimeError as e:
            diff, rec["error"] = 0, over_limit(e)
        if "error" not in rec:
            rec.update(set_equal=diff == 0, **(timed(ch, f"SELECT bitmapCardinality({agg}) FROM {frm}", a.store, a.cold, a.warm) if diff == 0 else {}))
        recs.append(rec)
        print(json.dumps(rec), flush=True)
        if diff or ("error" not in rec and rec["value"] != str(card)):
            res["mismatch"] = rec
            fail(res, a, json.dumps(rec))
    return recs


def targets(ch, n, seed):
    ch.q(f"CREATE OR REPLACE TABLE targets ENGINE = MergeTree ORDER BY span_id AS SELECT span_id, trace_id, toUInt32(span_id - span_seq % {CKPT}) AS ck "
         f"FROM spans ORDER BY cityHash64(span_id, {seed}) LIMIT {n}")


def stage_base(ch, a, res):
    ch.q("CREATE OR REPLACE TABLE msg (msg_id UInt32, role LowCardinality(String), n_chars UInt32, text String CODEC(ZSTD(3))) ENGINE = MergeTree ORDER BY msg_id")
    cpu = server_cpu(a.store)
    t = ch.q(f"INSERT INTO msg SELECT msg_id, role, n_chars, text FROM {src('messages')}")[2]
    res["load"]["msg"] = dict(rows=int(ch.val("SELECT count() FROM msg")), insert_s=round(t, 2), cpu_s=round(server_cpu(a.store) - cpu, 2))
    ch.q("CREATE OR REPLACE TABLE traces ENGINE = MergeTree ORDER BY trace_id AS SELECT * FROM " + src("traces"))
    ch.q("CREATE OR REPLACE TABLE spans ENGINE = MergeTree ORDER BY span_id AS SELECT * FROM " + src("spans"))
    ch.q("CREATE OR REPLACE TABLE span_trace ENGINE = Join(ANY, LEFT, span_id) AS SELECT span_id, trace_id FROM spans")
    res["spans"], res["traces"] = int(ch.val("SELECT count() FROM spans")), int(ch.val("SELECT count() FROM traces"))

    # Matched sets: independent random samples of distinct messages (three seeds per selectivity for the evaluation,
    # a separate one for the calibration) and substring predicates on the message text.
    ch.q("CREATE OR REPLACE TABLE matched (name String, msg_id UInt32) ENGINE = MergeTree ORDER BY (name, msg_id)")
    for k, (sel, per_mille) in enumerate(SELECTIVITY.items()):
        for name, seed in [(f"r{sel}-s{s}", 10 * k + s) for s in (1, 2, 3)] + [(f"c{sel}", 90 + k)]:
            ch.q(f"INSERT INTO matched SELECT '{name}', msg_id FROM msg WHERE cityHash64(msg_id, {seed}) % 1000 < {per_mille}")
    for name, needle in ({} if "planlab" in a.stages else NEEDLES).items():      # the needles are held out; a calibration-only job does not touch them
        res["msg_scan"][name] = timed(ch, f"SELECT count() FROM msg WHERE position(text, '{needle}') > 0", a.store, 1, a.warm)
        ch.q(f"INSERT INTO matched SELECT '{name}', msg_id FROM msg WHERE position(text, '{needle}') > 0")
    ch.q("OPTIMIZE TABLE matched FINAL")
    res["matched"] = {l.split("\t")[0]: int(l.split("\t")[1]) for l in ch.val("SELECT name, count() FROM matched GROUP BY name ORDER BY name").splitlines()}

    repo = ch.val(f"SELECT repo FROM traces GROUP BY repo ORDER BY abs(count() - {res['traces']} / 100), repo LIMIT 1")
    trace = ch.val(f"SELECT trace_id FROM traces ORDER BY span_hi - span_lo, trace_id LIMIT 1 OFFSET {res['traces'] // 2}")
    res["filters"] = {"none": dict(pred=None), "repo": dict(pred="repo = '%s'" % repo.replace("'", "\\'")), "trace": dict(pred=f"trace_id = {trace}")}
    for v in res["filters"].values():
        if v["pred"]:
            v["spans"], v["traces"] = map(int, ch.val(f"SELECT sum(span_hi - span_lo + 1), count() FROM traces WHERE {v['pred']}").split("\t"))


def stage_calib(ch, a, res):
    """Pick each layout's index granularity for this policy by one rule fixed in advance: the lowest geometric mean, over the
    calibration regimes, of the layout's best plan, where a plan costs sqrt(cold median * warm median). The calibration matched
    sets are drawn with seeds the evaluation never uses. A regime in which the layout finishes at no granularity is dropped for
    that layout; a granularity that fails a regime another one finishes is invalid."""
    assert len(a.policies) == 1, "calibrate one policy per run"
    p, C = a.policies[0], res["calib"]
    regimes = [(name, f["pred"]) for name in res["matched"] if name.startswith("c") for f in res["filters"].values()]
    cost = lambda t: math.sqrt(statistics.median(t["cold_ms"]) * statistics.median(t["warm_ms"]))

    def choose(kind, costs):                                      # costs: granularity -> per-regime cost or None
        usable = [r for r in range(len(next(iter(costs.values())))) if any(c[r] is not None for c in costs.values())]
        score = {g: round(math.exp(statistics.fmean(math.log(c[r]) for r in usable)), 3) for g, c in costs.items() if usable and all(c[r] is not None for r in usable)}
        C[kind]["score_ms"], C[kind]["regimes_used"] = score, len(usable)
        save(res, a.out)
        if not score:
            raise SystemExit(f"calibration: no granularity of {kind} finishes any regime")
        GRAN[kind] = min(score, key=score.get)
        print("calib", kind, score, "->", GRAN[kind], flush=True)

    for kind in ("d", "i", "b", "a", "a_bf", "a_tx"):
        C[kind], costs = {"errors": [], "bytes_on_disk": {}, "build_s": {}}, {}
        for g in CAND[kind]:
            name = "calib_d" if kind == "d" else "calib_t"
            st = load(ch, a.store, name, kind, build_sql(kind, p, "calib_d"), g)
            C[kind]["bytes_on_disk"][g], C[kind]["build_s"][g] = st["bytes_on_disk"], round(st["insert_s"] + st["optimize_s"], 2)
            costs[g] = []
            for mname, pred in regimes:
                P = ctx_any({kind: name}, f"(SELECT msg_id FROM matched WHERE name = '{mname}')", pred, res["matched"][mname])
                ts = {k: timed(ch, f"SELECT bitmapCardinality({agg}) FROM {frm}", a.store, 2, 3) for k, (agg, frm) in P.items()}
                C[kind]["errors"] += [f"{g} {mname} {k}: {t['error']}" for k, t in ts.items() if "error" in t][:2]
                costs[g].append(min((cost(t) for t in ts.values() if "error" not in t), default=None))
        choose(kind, costs)
        if kind == "d":                                           # the projections below are built from the log at its chosen granularity
            load(ch, a.store, "calib_d", "d", build_sql("d", p, None), GRAN["d"])
    C["ckpt"], costs = {"errors": []}, {}
    for g in CAND["ckpt"]:                                        # checkpoints: single-span reconstruction by log replay
        load(ch, a.store, "calib_t", "ckpt", build_sql("ckpt", p, None), g)
        costs[g] = []
        for seed in range(200, 205):
            targets(ch, 1, seed)
            t = timed(ch, ctx_items(dict(d="calib_d", ckpt="calib_t"))["D"], a.store, 2, 3)
            C["ckpt"]["errors"] += [t["error"]] if "error" in t else []
            costs[g].append(None if "error" in t else cost(t))
    choose("ckpt", costs)
    ch.q("DROP TABLE calib_d SYNC"); ch.q("DROP TABLE calib_t SYNC")
    C["chosen"] = dict(GRAN)


def stage_build(ch, a, res):
    L, K = res["load"], res["checks"]
    for p in a.policies:
        for kind in ("d", "ckpt", "i", "b", "a", "a_bf", "a_tx", "nf", "ni"):
            L[f"{kind}_{p}"] = load(ch, a.store, f"{kind}_{p}", kind, build_sql(kind, p, f"d_{p}"), GRAN[kind])
            save(res, a.out)
        print(p, json.dumps({k: (v["rows"], v["bytes_on_disk"], v["insert_s"] + v["optimize_s"]) for k, v in L.items() if k.endswith("_" + p)}), flush=True)
        # The arrays rebuilt from the log must equal the ingested requests, span by span and in order; every occurrence has one run.
        # Sparse contract: a span whose context is empty (possible in E-WIKI: a blanked page) has no occurrence and no rebuilt row,
        # so the rebuilt arrays are compared with the non-empty ingested arrays, and the ingested arrays must cover every span.
        K[f"a_{p}_spans_differ_from_requests"] = int(ch.val(
            f"SELECT countIf(a.msg_ids != f.msg_ids) + abs(toInt64(count()) - (SELECT count() FROM {src('arrays', p)} WHERE notEmpty(msg_ids))) "
            f"FROM a_{p} AS a INNER JOIN {src('arrays', p)} AS f ON f.span_id = a.span_id"))
        K[f"arrays_{p}_rows_not_one_per_span"] = int(ch.val(           # duplicates, plus span IDs in exactly one of arrays and spans
            f"SELECT (SELECT count() - uniqExact(span_id) FROM {src('arrays', p)}) + (SELECT count() FROM {src('arrays', p)} WHERE span_id NOT IN (SELECT span_id FROM spans)) "
            f"+ (SELECT count() FROM spans WHERE span_id NOT IN (SELECT span_id FROM {src('arrays', p)}))"))
        K[f"i_{p}_rows_minus_opens"] = L[f"i_{p}"]["rows"] - res["convert"]["policies"][p]["opens"]
        K[f"n_{p}_rows_minus_array_elems"] = L[f"ni_{p}"]["rows"] - res["convert"]["policies"][p]["array_elems"]
        res["matched_stats"][p] = {l.split("\t")[0]: dict(zip(("runs", "span_slots", "run_len_quantiles"), l.split("\t")[1:])) for l in ch.val(
            f"SELECT m.name, count(), sum(i.span_to - i.span_from + 1), toString(quantiles(0.5, 0.9, 0.99)(i.span_to - i.span_from + 1)) "
            f"FROM i_{p} AS i INNER JOIN matched AS m ON m.msg_id = i.msg_id GROUP BY m.name ORDER BY m.name").splitlines()}
        print("checks", K, flush=True)
        if any(K.values()):
            fail(res, a, json.dumps(K))


def stage_any(ch, a, res):
    for p in a.policies:
        T = {k: f"{k}_{p}" for k in ("a", "a_bf", "a_tx", "i", "b", "d", "nf", "ni")}
        for name, n in res["matched"].items():
            if name.startswith("c"):                              # calibration sets stay out of the evaluation
                continue
            for fname, fv in res["filters"].items():
                if a.any_filters and fname not in a.any_filters:
                    continue
                P = ctx_any(T, f"(SELECT msg_id FROM matched WHERE name = '{name}')", fv["pred"], n)
                if a.any_plans:                                   # a recheck of some plans where they apply (figures/tab2_exact.py checks the keys it needs)
                    P = {k: v for k, v in P.items() if k == "A.exists" or k in a.any_plans}
                res["any"] += run_plans(ch, P, "A.exists", a, dict(policy=p, matched=name, n_matched=n, filter=fname), res)
                save(res, a.out)


def stage_point(ch, a, res):
    """Reconstruct the ordered context of random spans and compare it, row by row, with the converter's arrays."""
    for p in a.policies:
        T = {k: f"{k}_{p}" for k in ("a", "i", "d", "ckpt", "nf")}
        for n, seeds in ((1000, [7]), (1, range(100, 110))):
            for seed in seeds:
                targets(ch, n, seed)
                truth = ch.val(f"SELECT span_id, msg_ids FROM {src('arrays', p)} WHERE span_id IN (SELECT span_id FROM targets) AND notEmpty(msg_ids) ORDER BY span_id")   # sparse contract
                for name, sql in ctx_items(T).items():
                    rec = dict(policy=p, n=n, seed=seed, plan=name, **timed(ch, sql, a.store, a.cold, a.warm))
                    rec["equal"] = rec.pop("value", None) == truth         # a plan without a value (time limit) fails the check
                    res["point"].append(rec)
                    print(json.dumps(rec), flush=True)
                    if not rec["equal"]:
                        fail(res, a, json.dumps(rec))
        save(res, a.out)


def stage_wide(ch, a, res):
    """Content queries on the wide span table against message-level predicates plus a mapping, on the wide subset.
    The wide text comes from the converter's simulation of the agent loop, so it is the independent reference here."""
    for p in a.wide_policies:
        L = res["load"]
        S = int(ch.val(f"SELECT max(span_id) + 1 FROM {src('wide', p)}"))
        L[f"w_{p}"] = load(ch, a.store, f"w_{p}", "w", f"SELECT span_id, input FROM {src('wide', p)}")
        L[f"w_{p}"]["logical_bytes"] = int(ch.val(f"SELECT sum(length(input)) FROM w_{p}"))
        L[f"msg_s_{p}"] = load(ch, a.store, f"msg_s_{p}", "msg_s", f"SELECT msg_id, text FROM msg WHERE msg_id IN (SELECT msg_id FROM i_{p} WHERE span_from < {S})")
        L[f"a_s_{p}"] = load(ch, a.store, f"a_s_{p}", "a", f"SELECT * FROM a_{p} WHERE span_id < {S}", GRAN["a"])
        L[f"i_s_{p}"] = load(ch, a.store, f"i_s_{p}", "i", f"SELECT * FROM i_{p} WHERE span_from < {S}", GRAN["i"])
        L[f"b_s_{p}"] = load(ch, a.store, f"b_s_{p}", "b", f"SELECT msg_id, {expand('span_from', 'span_to')} FROM i_s_{p} GROUP BY msg_id", GRAN["b"])
        preds = [(f"token:{t}", f"hasToken({{col}}, '{t}')") for t in TOKENS] + [(f"substr:{k[2:]}", f"position({{col}}, '{v}') > 0") for k, v in NEEDLES.items()]
        for label, pred in preds:
            P = {"W": ("groupBitmapState(span_id)", f"w_{p} WHERE {pred.format(col='input')}")}
            M = f"(SELECT msg_id FROM msg_s_{p} WHERE {pred.format(col='text')})"
            P.update({"F+" + k: v for k, v in ctx_any(dict(a=f"a_s_{p}", i=f"i_s_{p}", b=f"b_s_{p}"), M, None, 1 << 30).items()})
            res["wide"] += run_plans(ch, P, "W", a, dict(policy=p, pred=label, spans=S), res)
            res["wide"].append(dict(policy=p, pred=label, plan="msg_scan", **timed(ch, f"SELECT count() FROM msg_s_{p} WHERE {pred.format(col='text')}", a.store, a.cold, a.warm)))
            save(res, a.out)


VALUES = {"n_chars": "n_chars", "n_bytes": "toUInt32(length(text))", "n_err": "toUInt32(countSubstrings(text, 'Error'))"}   # per-message values for ctx_sum
                                                                  # (--workload wiki counts WIKI_COUNT_TOKEN instead; set in main)


def val(f):
    return lambda col: f"joinGet('msg_val', '{f}', {col})"


def msg_values(ch):
    ch.q("CREATE OR REPLACE TABLE msg_val ENGINE = Join(ANY, LEFT, msg_id) AS SELECT msg_id, " + ", ".join(f"{e} AS {k}" for k, e in VALUES.items()) + " FROM msg")


def rel_diff(ch, ref, sql, limit=0):
    """Compare two (span_id, v) relations, zero values dropped: the rows that are in exactly one of them, and both row counts.
    Under the sparse contract (an omitted span has value 0) this equals comparing the value of every span. limit: seconds
    (0: none); a comparison over the limit raises RuntimeError."""
    x, y = (f"(SELECT span_id, toInt64(v) AS v FROM ({q}) WHERE v != 0)" for q in (ref, sql))
    only, nx, ny = map(int, ch.q(f"SELECT (SELECT count() FROM (SELECT span_id, v FROM {x} EXCEPT SELECT span_id, v FROM {y})) + "
                                   f"(SELECT count() FROM (SELECT span_id, v FROM {y} EXCEPT SELECT span_id, v FROM {x})), "
                                   f"(SELECT count() FROM {x}), (SELECT count() FROM {y})", max_execution_time=limit)[0].split("\t"))
    return dict(rows_differ=only + abs(nx - ny), rows_ref=nx, rows_plan=ny)


def stage_agg(ch, a, res):
    """ctx_count and ctx_sum (three per-message values): the (span, value) relation of every plan against the one computed
    from the arrays. A plan whose relation differs stops the run."""
    msg_values(ch)
    for p in a.policies:
        T = {k: f"{k}_{p}" for k in ("a", "i", "d", "nf", "ni")}
        cases = {}
        for name in ("r1-s1", AGG_NEEDLE):
            M = f"(SELECT msg_id FROM matched WHERE name = '{name}')"
            cases[f"ctx_count/{name}"] = ctx_count(T, M)
            cases.update({f"ctx_sum[{f}]/{name}": ctx_sum(T, val(f), M) for f in VALUES})
        cases.update({f"ctx_sum[{f}]/all": ctx_sum(T, val(f)) for f in VALUES})          # e.g. the size of every span's context
        for case, Q in cases.items():
            ref = Q.pop("A")
            rec = dict(policy=p, case=case, plan="A (reference)", **timed(ch, f"SELECT sum(v), count() FROM ({ref})", a.store, a.cold, a.warm))
            res["agg"].append(rec); print(json.dumps(rec), flush=True)
            for name, sql in Q.items():
                d = rel_diff(ch, ref, sql)
                rec = dict(policy=p, case=case, plan=name, **d, **(timed(ch, f"SELECT sum(v), count() FROM ({sql})", a.store, a.cold, a.warm) if d["rows_differ"] == 0 else {}))
                res["agg"].append(rec); print(json.dumps(rec), flush=True)
                if d["rows_differ"]:
                    fail(res, a, json.dumps(rec))
        save(res, a.out)


def not_reproduced(ch, p, S, text="text"):
    """Rebuild the request of every span below S from its array and the dictionary (column expression `text`) and compare it
    with the logged request. Returns (spans compared, spans whose bytes differ)."""
    ch.q(f"CREATE OR REPLACE TABLE msg_text ENGINE = Join(ANY, LEFT, msg_id) AS SELECT msg_id, {text} AS text FROM msg WHERE msg_id IN (SELECT msg_id FROM i_{p} WHERE span_from < {S})")
    ch.q(f"CREATE OR REPLACE TABLE ctx_j ENGINE = Join(ANY, LEFT, span_id) AS SELECT span_id, msg_ids FROM a_{p} WHERE span_id < {S}")
    rebuilt = ctx_text("joinGet('ctx_j', 'msg_ids', span_id)", lambda c: f"joinGet('msg_text', 'text', {c})")
    n, bad = map(int, ch.val(f"SELECT count(), countIf(input != {rebuilt}) FROM {src('wide', p)}").split("\t"))
    ch.q("DROP TABLE msg_text"); ch.q("DROP TABLE ctx_j")
    return n, bad


def stage_bytes(ch, a, res):
    """Factored results against the logged requests themselves (the spans whose serialized request was kept, wide.parquet):
    the request rebuilt from the array and the dictionary equals the logged bytes; ctx_any, ctx_sum and ctx_count equal the
    values computed from the request text. Any difference stops the run. Nothing here is timed."""
    msg_values(ch)
    for p in a.wide_policies:
        W = src("wide", p)
        S = int(ch.val(f"SELECT max(span_id) + 1 FROM {W}"))
        sub = f"(SELECT * FROM i_{p} WHERE span_from < {S})"
        n, bad = not_reproduced(ch, p, S)
        r = res["bytes"][p] = dict(spans=n, logical_bytes=int(ch.val(f"SELECT sum(length(input)) FROM {W}")), requests_not_reproduced=bad + abs(n - S), checks={})
        K = r["checks"]
        for name, needle in NEEDLES.items():                      # ctx_any against a scan of the request text
            agg, frm = ctx_any(dict(i=sub), matched(f"position(text, '{needle}') > 0"), None, 0)["I.expand"]
            K[f"ctx_any/{name}"] = dict(rows_differ=int(ch.val(
                f"SELECT bitmapXorCardinality((SELECT {agg} FROM {frm}), (SELECT groupBitmapState(span_id) FROM {W} WHERE position(input, '{needle}') > 0))")))
        # ctx_sum over the whole context. A request is '[' + messages joined by ',' + ']': its length is the sum of the message
        # lengths plus one byte per message plus one. Occurrences of a string that cannot contain a message boundary add up.
        size = (f"SELECT span_id, v + c + 1 AS v FROM (SELECT span_id, toInt64(sum(n)) AS v, toInt64(count()) AS c "
                f"FROM ({slots(sub, ', ' + val('n_bytes')('msg_id') + ' AS n')}) GROUP BY span_id)")
        K["ctx_sum[n_bytes]/all"] = rel_diff(ch, f"SELECT span_id, length(input) AS v FROM {W} WHERE length(input) > 2", size)
        K["ctx_sum[n_err]/all"] = rel_diff(ch, f"SELECT span_id, countSubstrings(input, '{COUNT_TOKEN}') AS v FROM {W}", ctx_sum(dict(i=sub), val("n_err"))["I.expand"])
        # ctx_count on a 1% sample of the spans: split the request into its messages and test each one
        needle, pick = NEEDLES[AGG_NEEDLE], "cityHash64(span_id, 9) % 100 = 0"
        counted = ctx_count(dict(i=sub), matched(f"position(text, '{needle}') > 0"))["I.expand"]
        K[f"ctx_count/{AGG_NEEDLE} (1% of spans)"] = rel_diff(
            ch, f"SELECT span_id, arrayCount(m -> position(m, '{needle}') > 0, JSONExtractArrayRaw(input)) AS v FROM {W} WHERE {pick}",
            f"SELECT span_id, v FROM ({counted}) WHERE {pick}")
        print("bytes", p, json.dumps(r), flush=True)
        save(res, a.out)
        if r["requests_not_reproduced"] or any(k["rows_differ"] for k in K.values()):
            fail(res, a, json.dumps(r))


def stage_selfcheck(ch, a, res):
    """The new checks must be able to fail: damage one input of each and require that the check reports it."""
    p = a.wide_policies[0]
    msg_values(ch)
    ref = ctx_sum(dict(a=f"a_{p}"), val("n_bytes"))["A"]
    span = int(ch.val(f"SELECT span_id FROM a_{p} WHERE length(msg_ids) > 0 ORDER BY cityHash64(span_id) LIMIT 1"))
    S = int(ch.val(f"SELECT max(span_id) + 1 FROM {src('wide', p)}"))
    victim = int(ch.val(f"SELECT msg_id FROM i_{p} WHERE span_from < {S} ORDER BY cityHash64(msg_id) LIMIT 1"))
    out = res["selfcheck"] = {
        "aggregate relation, one span removed from the plan: rows that differ": rel_diff(ch, ref, f"SELECT * FROM ({ref}) WHERE span_id != {span}")["rows_differ"],
        "aggregate relation, one value off by one: rows that differ": rel_diff(ch, ref, f"SELECT span_id, v + (span_id = {span}) AS v FROM ({ref})")["rows_differ"],
        "byte check, one stored message with one byte appended: requests not reproduced": not_reproduced(ch, p, S, f"if(msg_id = {victim}, concat(text, ' '), text)")[1],
        "byte check, untouched dictionary: requests not reproduced": not_reproduced(ch, p, S)[1]}
    print("selfcheck", json.dumps(out), flush=True)
    vals = list(out.values())
    if not (vals[0] > 0 and vals[1] > 0 and vals[2] > 0 and vals[3] == 0):
        fail(res, a, "a damaged input went unnoticed: " + json.dumps(out))


def presence(frm, cols):
    """Maximal runs of consecutive spans in which a message is present in a trace, from rows (trace_id, msg_id, lo, hi, tie):
    intervals that overlap or touch are merged."""
    w = "PARTITION BY trace_id, msg_id ORDER BY lo, hi, tie"
    return (f"(SELECT trace_id, msg_id, min(lo) AS span_from, max(hi) AS span_to FROM (SELECT trace_id, msg_id, lo, hi, sum(is_new) OVER ({w} ROWS UNBOUNDED PRECEDING) AS grp "
            f"FROM (SELECT trace_id, msg_id, lo, hi, tie, lo > 1 + max(hi) OVER ({w} ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS is_new "
            f"FROM (SELECT {cols} FROM {frm}))) GROUP BY trace_id, msg_id, grp)")


def stage_life(ch, a, res):
    """Lifecycle queries: when messages enter and leave contexts. Interval rows hold the runs; the normalized membership table
    holds one row per (span, message) and has to find the runs first. Both sides must return the same rows."""
    ch.q("CREATE OR REPLACE TABLE msg_role ENGINE = Join(ANY, LEFT, msg_id) AS SELECT msg_id, role, n_chars FROM msg")
    ch.q("CREATE OR REPLACE TABLE trace_hi ENGINE = Join(ANY, LEFT, trace_id) AS SELECT trace_id, span_hi FROM traces")
    role, chars = "joinGet('msg_role', 'role', msg_id)", "joinGet('msg_role', 'n_chars', msg_id)"
    ended, length = "span_to < joinGet('trace_hi', 'span_hi', trace_id)", "toUInt64(span_to - span_from + 1)"
    for p in a.policies:
        I, NI, NF = f"i_{p}", f"ni_{p}", f"nf_{p}"
        runs_i = presence("{I}", "trace_id, msg_id, span_from AS lo, span_to AS hi, occ AS tie")
        runs_n = presence("{NI}", "trace_id, msg_id, span_id AS lo, span_id AS hi, pos AS tie")
        Q = {  # query -> (FCX on interval rows, baseline on normalized rows); {I}, {NI}, {NF} are filled in below
            "L1 context volume by role (characters x calls)": (
                f"SELECT {role} AS r, sum({length} * {chars}) AS v, sum({length}) AS n FROM {{I}} GROUP BY r ORDER BY r",
                f"SELECT {role} AS r, sum(toUInt64({chars})) AS v, count() AS n FROM {{NF}} GROUP BY r ORDER BY r"),
            "L2 evictions by role (runs that end before their trace does)": tuple(
                f"SELECT {role} AS r, count() AS n, sum({length}) AS v FROM {R} WHERE {ended} GROUP BY r ORDER BY r" for R in (runs_i, runs_n)),
            "L3 re-entries (messages present in more than one run of a trace)": tuple(
                f"SELECT count() AS n, sum(c) AS v FROM (SELECT trace_id, msg_id, count() AS c FROM {R} GROUP BY trace_id, msg_id HAVING c > 1)" for R in (runs_i, runs_n)),
            "L4 residence by role (run length in calls: mean, max)": tuple(
                f"SELECT {role} AS r, count() AS n, sum({length}) AS v, max({length}) AS m FROM {R} GROUP BY r ORDER BY r" for R in (runs_i, runs_n)),
        }
        full = dict(I=I, NI=NI, NF=NF)
        small = {k: f"(SELECT * FROM {t} WHERE trace_id < 2000)" for k, t in full.items()}
        for name, sqls in Q.items():
            recs = [dict(policy=p, query=name, plan=plan, **timed(ch, sql.format(**full), a.store, a.cold, a.warm)) for plan, sql in zip(("I", "N"), sqls)]
            rows = [rec.pop("value", None) for rec in recs]
            if None in rows:                                      # a side ran out of time: compare the two on the first 2,000 traces
                rows = [ch.val(sql.format(**small)) for sql in sqls]
                equal = dict(equal_on_2000_traces=rows[0] == rows[1])
            else:
                equal = dict(equal=rows[0] == rows[1])
            for rec in recs:
                rec.update(equal)
                res["life"].append(rec); print(json.dumps(rec), flush=True)
            if rows[0] != rows[1]:
                fail(res, a, json.dumps(recs))
        save(res, a.out)


EXAMPLE_K = (3, 10, 30)                                          # residence thresholds of the running example, in calls
EXAMPLE_REF_TIMEOUT = 1800                                        # seconds for the reference and the equality checks (not timed)


def stage_example(ch, a, res):
    """The paper's running example: the calls whose context holds a message containing a Python traceback that has also been in
    the context of each of the K - 1 preceding calls of its trace. Three plans, on the spans that have a logged request:
      I  the predicate runs once on the dictionary; presence runs of the matched messages come from interval rows;
      N  the same runs found from the normalized membership table;
      W  the wide table: the calls whose request contains the needle, their messages that contain it (identified by their
         serialized bytes), and runs of consecutive calls per (trace, message).
    A call qualifies when it lies at least K - 1 calls after the start of a run. Every plan's span set is checked against W
    before it is timed and a difference stops the run; a plan over the time limit is recorded and skipped, as in run_plans.
    The trace lookup span_tr is built once, outside the timed queries; the wide table's text index, if the server uses it for
    position(), is recorded with EXPLAIN."""
    needle = NEEDLES["t:traceback"]
    if not set(a.wide_policies) <= set(a.policies):
        fail(res, a, f"example: wide policies {a.wide_policies} must also be built (policies {a.policies})")
    ch.q(f"CREATE OR REPLACE TABLE span_tr ENGINE = Join(ANY, LEFT, span_id) AS SELECT span_id, trace_id FROM {src('spans')}")
    M = matched(f"position(text, '{needle}') > 0")
    res.setdefault("example", [])
    res.setdefault("example_meta", {})
    for p in a.wide_policies:
        for tab in (f"i_{p}", f"ni_{p}"):
            if not int(ch.val(f"EXISTS TABLE {tab}")):
                fail(res, a, f"example: table {tab} is missing (run the build stage with policy {p})")
        if not int(ch.val(f"EXISTS TABLE w_{p}")):
            res["load"][f"w_{p}"] = load(ch, a.store, f"w_{p}", "w", f"SELECT span_id, input FROM {src('wide', p)}")
        n, u, lo, hi = map(int, ch.val(f"SELECT count(), uniqExact(span_id), min(span_id), max(span_id) FROM w_{p}").split("\t"))
        if n == 0 or u != n or lo != 0 or n != hi + 1:
            fail(res, a, f"example: w_{p} is not a dense prefix of the spans (count {n}, distinct {u}, min {lo}, max {hi})")
        S = hi + 1
        runs_i = presence(f"(SELECT * FROM i_{p} WHERE msg_id IN {M} AND span_from < {S})",
                          f"trace_id, msg_id, span_from AS lo, least(span_to, toUInt32({S - 1})) AS hi, occ AS tie")
        runs_n = presence(f"(SELECT * FROM ni_{p} WHERE msg_id IN {M} AND span_id < {S})", "trace_id, msg_id, span_id AS lo, span_id AS hi, pos AS tie")
        cand = f"(SELECT span_id, input FROM w_{p} PREWHERE position(input, '{needle}') > 0)"      # filter requests before unnesting
        hits = (f"(SELECT joinGet('span_tr', 'trace_id', span_id) AS trace_id, m, span_id FROM {cand} ARRAY JOIN JSONExtractArrayRaw(input) AS m "
                f"WHERE position(m, '{needle}') > 0 GROUP BY trace_id, m, span_id)")
        runs_w = (f"(SELECT span_id, span_id - min(span_id) OVER (PARTITION BY trace_id, m, grp) AS age FROM "
                  f"(SELECT trace_id, m, span_id, span_id - row_number() OVER (PARTITION BY trace_id, m ORDER BY span_id) AS grp FROM {hits}))")
        res["example_meta"][p] = dict(spans=S, wide_indexes=ch.val(
            f"SELECT name, type_full FROM system.data_skipping_indices WHERE database = currentDatabase() AND table = 'w_{p}'"))
        for K in EXAMPLE_K:
            P = {"I": f"SELECT {expand(f'span_from + {K - 1}', 'span_to')} FROM {runs_i} WHERE span_to >= span_from + {K - 1}",
                 "N": f"SELECT {expand(f'span_from + {K - 1}', 'span_to')} FROM {runs_n} WHERE span_to >= span_from + {K - 1}",
                 "W": f"SELECT groupBitmapState(toUInt32(span_id)) FROM {runs_w} WHERE age >= {K - 1}"}
            res["example_meta"][p][f"explain_W_K{K}"] = ch.val(f"EXPLAIN indexes = 1 SELECT bitmapCardinality(({P['W']}))")
            try:
                ch.q(f"CREATE OR REPLACE TABLE ex_ref ENGINE = Memory AS SELECT ({P['W']}) AS bm", max_execution_time=EXAMPLE_REF_TIMEOUT)
            except RuntimeError as e:
                res["mismatch"] = dict(policy=p, K=K, plan="W", reference_failed=str(e)[:300])
                fail(res, a, json.dumps(res["mismatch"]))
            card = int(ch.val("SELECT bitmapCardinality(bm) FROM ex_ref"))
            for plan, sql in P.items():
                rec = dict(policy=p, K=K, plan=plan, spans=S, cardinality=card)
                try:
                    diff = int(ch.q(f"SELECT bitmapXorCardinality(({sql}), (SELECT bm FROM ex_ref))", max_execution_time=EXAMPLE_REF_TIMEOUT)[0])
                except RuntimeError as e:                         # over the time or memory limit: recorded and skipped, as in run_plans
                    rec["error"] = over_limit(e)
                    res["example"].append(rec); print(json.dumps(rec), flush=True)
                    continue
                if diff:
                    res["mismatch"] = dict(rec, xor=diff)
                    fail(res, a, json.dumps(res["mismatch"]))
                rec.update(set_equal=True, **timed(ch, f"SELECT bitmapCardinality(({sql}))", a.store, a.cold, a.warm))
                res["example"].append(rec); print(json.dumps(rec), flush=True)
                if "error" not in rec and rec["value"] != str(card):
                    res["mismatch"] = rec
                    fail(res, a, json.dumps(rec))
        save(res, a.out)


def stage_mixed(ch, a, res):
    """The running example and ctx_count over the held-out predicates of the index stage instead of one needle. The predicates
    are drawn by sample_predicates from the same dictionary rows with the same seed as stage_index, so they are the index
    stage's predicates (recorded for comparison). For every held-out predicate:
      residence, K in EXAMPLE_K: the calls whose context holds a matching message that has also been in the context of each of
        the K - 1 preceding calls of its trace; plans I, N, W as in stage_example, the predicate in place of the needle;
      ctx_count: per call, the number of matching message occurrences in its context; plans I.expand, N.inv, and W, which
        splits each request into its messages and counts the matching ones.
    Residence also has W.distinct: the distinct matching elements of each request by arrayDistinct(arrayFilter(...)) instead
    of ARRAY JOIN and GROUP BY (a second wide formulation; W stays the reference). The wide table is the index-free layout of
    stage_index (wx, as inserted); the index metadata of both text tables and EXPLAIN indexes = 1 of each W plan are recorded.
    W evaluates the predicate on each message of a request. It first keeps only the requests on which the predicate holds: a
    request that holds a matching message also contains the substring, or a match of an unanchored regular expression (an
    anchored one would make W split every request; REGEX has none). No text index on either side. An equality check over
    EXAMPLE_REF_TIMEOUT is recorded as unvalidated and the plan is not timed. Every plan is compared with W before it is
    timed and a difference stops the run; a plan over the time limit is recorded and skipped."""
    if not set(a.wide_policies) <= set(a.policies):
        fail(res, a, f"mixed: wide policies {a.wide_policies} must also be built (policies {a.policies})")
    ch.q(f"CREATE OR REPLACE TABLE span_tr ENGINE = Join(ANY, LEFT, span_id) AS SELECT span_id, trace_id FROM {src('spans')}")
    out = res.setdefault("mixed", {})
    for p in a.wide_policies:
        for tab in (f"i_{p}", f"ni_{p}"):
            if not int(ch.val(f"EXISTS TABLE {tab}")):
                fail(res, a, f"mixed: table {tab} is missing (run the build stage with policy {p})")
        WT = f"wm_{p}"
        res["load"][WT] = load(ch, a.store, WT, "wx", f"SELECT span_id, input FROM {src('wide', p)}", a.mixed_wide_gran, optimize=a.mixed_merge)
        n, u, lo, hi = map(int, ch.val(f"SELECT count(), uniqExact(span_id), min(span_id), max(span_id) FROM {WT}").split("\t"))
        if n == 0 or u != n or lo != 0 or n != hi + 1:
            fail(res, a, f"mixed: {WT} is not a dense prefix of the spans (count {n}, distinct {u}, min {lo}, max {hi})")
        S = hi + 1
        M = f"mm_{p}"                                             # the rows stage_index samples its predicates from
        res["load"][M] = load(ch, a.store, M, "mx", f"SELECT msg_id, text FROM msg WHERE msg_id IN (SELECT msg_id FROM i_{p} WHERE span_from < {S})", VARIANT_KINDS.get(a.mixed_index, 8192), optimize=a.mixed_merge)
        r_parts = {t: ch.val(f"SELECT sum(rows), sum(marks), count() FROM system.parts WHERE active AND database = currentDatabase() AND table = '{t}'") for t in (M, WT)}
        built = {t: set_index(ch, a.store, t, c, INDEXES[a.mixed_index]) for t, c in ((M, "text"), (WT, "input"))} if a.mixed_index else {}   # the same kind on both sides
        state = lambda: {t: ch.val(f"SELECT groupArray(name), sum(rows), sum(marks), sum(bytes_on_disk) FROM (SELECT name, rows, marks, bytes_on_disk FROM system.parts WHERE active AND database = currentDatabase() AND table = '{t}' ORDER BY name)") for t in (M, WT)}
        if a.mixed_merge:                                         # one active part each, nothing merging or mutating, before anything is timed
            busy = {t: ch.val(f"SELECT (SELECT count() FROM system.parts WHERE active AND database = currentDatabase() AND table = '{t}'), (SELECT count() FROM system.merges WHERE database = currentDatabase() AND table = '{t}'), (SELECT count() FROM system.mutations WHERE database = currentDatabase() AND table = '{t}' AND NOT is_done)") for t in (M, WT)}
            if any(v != "1\t0\t0" for v in busy.values()):
                fail(res, a, f"mixed: merge state not settled: {busy}")
        merge_state = dict(mixed_merge=a.mixed_merge, before=state())
        preds, _ = sample_predicates(ch, a, M)
        preds = {k: v for k, v in preds.items() if k.startswith("eval")}
        if len(preds) != a.mixed_preds:
            fail(res, a, f"mixed: {len(preds)} held-out predicates, expected {a.mixed_preds}")
        ix = {t: ch.val(f"SELECT name, type_full FROM system.data_skipping_indices WHERE database = currentDatabase() AND table = '{t}'") for t in (WT, M)}
        r = out[p] = dict(spans=S, mixed_index=a.mixed_index, index_build=built, parts=r_parts, merge_state=merge_state, predicates={k: list(v) for k, v in preds.items()}, text_indexes=ix, explain={}, records=[])
        # intervals clipped to the calls with a logged request, so that no plan expands calls it then discards
        Isub = f"(SELECT * REPLACE (least(span_to, toUInt32({S - 1})) AS span_to) FROM i_{p} WHERE span_from < {S})"
        NIsub = f"(SELECT * FROM ni_{p} WHERE span_id < {S})"
        for name, (kind, s) in preds.items():
            cond = (lambda col: like(col, s)) if kind == "like" else (lambda col: f"match({col}, '{s}')")
            Mq = f"(SELECT msg_id FROM {M} WHERE {cond('text')})"
            anchored = kind == "match" and (s.startswith("^") or s.endswith("$"))
            cand = WT if anchored else f"(SELECT span_id, input FROM {WT} PREWHERE {cond('input')})"
            tag = dict(policy=p, pred=name, matched_messages=int(ch.val(f"SELECT count() FROM {Mq}")))
            runs_i = presence(f"(SELECT * FROM {Isub} WHERE msg_id IN {Mq})", "trace_id, msg_id, span_from AS lo, span_to AS hi, occ AS tie")
            runs_n = presence(f"(SELECT * FROM {NIsub} WHERE msg_id IN {Mq})", "trace_id, msg_id, span_id AS lo, span_id AS hi, pos AS tie")
            hits = (f"(SELECT joinGet('span_tr', 'trace_id', span_id) AS trace_id, m, span_id FROM {cand} ARRAY JOIN JSONExtractArrayRaw(input) AS m "
                    f"WHERE {cond('m')} GROUP BY trace_id, m, span_id)")
            hits_d = (f"(SELECT joinGet('span_tr', 'trace_id', span_id) AS trace_id, arrayJoin(arrayDistinct(arrayFilter(x -> {cond('x')}, "
                      f"JSONExtractArrayRaw(input)))) AS m, span_id FROM {cand})")
            age = lambda h: (f"(SELECT span_id, span_id - min(span_id) OVER (PARTITION BY trace_id, m, grp) AS age FROM "
                             f"(SELECT trace_id, m, span_id, span_id - row_number() OVER (PARTITION BY trace_id, m ORDER BY span_id) AS grp FROM {h}))")
            runs_w, runs_wd = age(hits), age(hits_d)
            for K in a.mixed_ks:
                P = {"W": f"SELECT groupBitmapState(toUInt32(span_id)) FROM {runs_w} WHERE age >= {K - 1}",
                     "W.distinct": f"SELECT groupBitmapState(toUInt32(span_id)) FROM {runs_wd} WHERE age >= {K - 1}",
                     "I": f"SELECT {expand(f'span_from + {K - 1}', 'span_to')} FROM {runs_i} WHERE span_to >= span_from + {K - 1}",
                     "N": f"SELECT {expand(f'span_from + {K - 1}', 'span_to')} FROM {runs_n} WHERE span_to >= span_from + {K - 1}"}
                try:
                    ch.q(f"CREATE OR REPLACE TABLE ex_ref ENGINE = Memory AS SELECT ({P['W']}) AS bm", max_execution_time=EXAMPLE_REF_TIMEOUT, **NO_INDEX)
                except RuntimeError as e:
                    res["mismatch"] = dict(tag, query="residence", K=K, plan="W", reference_failed=str(e)[:300])
                    fail(res, a, json.dumps(res["mismatch"]))
                card = int(ch.val("SELECT bitmapCardinality(bm) FROM ex_ref"))
                if K == a.mixed_ks[0]:
                    r["explain"][name] = {k: ch.val(f"EXPLAIN indexes = 1 SELECT bitmapCardinality(({P[k]}))") for k in P if a.mixed_index or k.startswith("W")}
                for plan, sql in P.items():
                    rec = dict(tag, query="residence", K=K, plan=plan, cardinality=card)
                    try:
                        diff = int(ch.q(f"SELECT bitmapXorCardinality(({sql}), (SELECT bm FROM ex_ref))", max_execution_time=EXAMPLE_REF_TIMEOUT)[0])
                    except RuntimeError as e:                     # over the time or memory limit: unvalidated, not timed
                        rec["unvalidated"] = over_limit(e)
                        r["records"].append(rec); print(json.dumps(rec), flush=True)
                        continue
                    if diff:
                        res["mismatch"] = dict(rec, xor=diff)
                        fail(res, a, json.dumps(res["mismatch"]))
                    rec.update(set_equal=True, **timed(ch, f"SELECT bitmapCardinality(({sql}))", a.store, a.cold, a.warm))
                    r["records"].append(rec); print(json.dumps(rec), flush=True)
                    if "error" not in rec and rec["value"] != str(card):
                        res["mismatch"] = rec
                        fail(res, a, json.dumps(rec))
            Q = {"W": f"SELECT span_id, toInt64(arrayCount(m -> {cond('m')}, JSONExtractArrayRaw(input))) AS v FROM {cand} WHERE v > 0",
                 "I.expand": f"SELECT span_id, v FROM ({ctx_count(dict(i=Isub), Mq)['I.expand']}) WHERE span_id < {S}",
                 "N.inv": ctx_count(dict(ni=NIsub), Mq)["N.inv"]}
            if a.mixed_index:
                r["explain"][name].update({f"count {k}": ch.val(f"EXPLAIN indexes = 1 SELECT count(), sum(v) FROM ({v})") for k, v in Q.items()})
            try:
                ch.q(f"CREATE OR REPLACE TABLE cnt_ref ENGINE = Memory AS {Q['W']}", max_execution_time=EXAMPLE_REF_TIMEOUT, **NO_INDEX)
            except RuntimeError as e:
                res["mismatch"] = dict(tag, query="ctx_count", plan="W", reference_failed=str(e)[:300])
                fail(res, a, json.dumps(res["mismatch"]))
            rows, total = map(int, ch.val("SELECT count(), sum(v) FROM cnt_ref").split("\t"))
            for plan, sql in Q.items():
                rec = dict(tag, query="ctx_count", plan=plan, rows=rows, total=total)
                try:
                    rec.update(rel_diff(ch, "SELECT span_id, v FROM cnt_ref", sql, EXAMPLE_REF_TIMEOUT))
                except RuntimeError as e:                         # over the time or memory limit: unvalidated, not timed
                    rec["unvalidated"] = over_limit(e)
                    r["records"].append(rec); print(json.dumps(rec), flush=True)
                    continue
                if rec["rows_differ"]:
                    res["mismatch"] = rec
                    fail(res, a, json.dumps(rec))
                rec.update(timed(ch, f"SELECT count(), sum(v) FROM ({sql})", a.store, a.cold, a.warm))
                r["records"].append(rec); print(json.dumps(rec), flush=True)
                if "error" not in rec and rec["value"] != f"{rows}\t{total}":
                    res["mismatch"] = rec
                    fail(res, a, json.dumps(rec))
            save(res, a.out)
        merge_state["after"] = state()
        save(res, a.out)


def elapsed_ms(result):
    """Server-side milliseconds of a finished statement; a missing measurement is an error, not zero."""
    ns = result[1].get("elapsed_ns")
    if not ns:
        raise SystemExit("no elapsed_ns in the server's summary")
    return round(int(ns) / 1e6, 3)


def dense_spans(ch, a, res, table):
    """The wide table must hold every call below its maximum exactly once; returns the number of calls."""
    n, u, lo, hi = map(int, ch.val(f"SELECT count(), uniqExact(span_id), min(span_id), max(span_id) FROM {table}").split("\t"))
    if n == 0 or u != n or lo != 0 or n != hi + 1:
        fail(res, a, f"{table} is not a dense prefix of the spans (count {n}, distinct {u}, min {lo}, max {hi})")
    return hi + 1


def timed_two_step(ch, pre, sql, warm):
    """Build, then query, in that order on every run, within one time limit per run. The cost of a run is the sum of the two."""
    out = dict(pre_ms=[], query_ms=[], warm_ms=[])
    try:
        for _ in range(warm):
            t = time.perf_counter()
            p = elapsed_ms(ch.q(pre, max_execution_time=TIMEOUT))
            left = max(1, int(TIMEOUT - (time.perf_counter() - t)))
            q = ch.q(sql, max_execution_time=left)
            out["pre_ms"].append(p); out["query_ms"].append(elapsed_ms(q)); out["warm_ms"].append(round(p + out["query_ms"][-1], 3))
        out["value"] = q[0]
    except RuntimeError as e:
        out["error"] = over_limit(e)
    return out


def stage_planlab(ch, a, res):
    """Plan variants for the running example and ctx_count, on the calibration predicates only (never the held-out ones):
    where a query's time goes, and whether other plans lower it, for K = 3 and warm runs. Every variant is compared with W; a
    difference stops the run. M: the predicate alone on the dictionary. W, I: the plans of stage_mixed.
      I.2p: the predicate first, into a Set table; then the intervals of that set (each run does both; its time is the sum).
      I.fold: as I, merging a message's intervals with arrayFold instead of window functions. I.2pfold: both."""
    if not set(a.wide_policies) <= set(a.policies):
        fail(res, a, f"planlab: wide policies {a.wide_policies} must also be built (policies {a.policies})")
    ch.q(f"CREATE OR REPLACE TABLE span_tr ENGINE = Join(ANY, LEFT, span_id) AS SELECT span_id, trace_id FROM {src('spans')}")
    out = res.setdefault("planlab", {})
    K = EXAMPLE_K[0]
    for p in a.wide_policies:
        WT, M = f"wm_{p}", f"mm_{p}"
        res["load"][WT] = load(ch, a.store, WT, "wx", f"SELECT span_id, input FROM {src('wide', p)}", a.lab_wide_gran, optimize=False)
        S = dense_spans(ch, a, res, WT)
        res["load"][M] = load(ch, a.store, M, "mx", f"SELECT msg_id, text FROM msg WHERE msg_id IN (SELECT msg_id FROM i_{p} WHERE span_from < {S})", VARIANT_KINDS.get(a.lab_index, 8192), optimize=False)
        preds, _ = sample_predicates(ch, a, M)
        preds = {k: v for k, v in preds.items() if k.startswith("calib")}
        ix = {}
        if a.lab_index:                                           # the same index kind on both sides, as in stage_index
            ix = {t: set_index(ch, a.store, t, c, INDEXES[a.lab_index]) for t, c in ((M, "text"), (WT, "input"))}
        Isub = f"(SELECT * REPLACE (least(span_to, toUInt32({S - 1})) AS span_to) FROM i_{p} WHERE span_from < {S})"
        r = out[p] = dict(spans=S, wide_gran=a.lab_wide_gran, parts={t: ch.val(f"SELECT sum(rows), sum(marks), count() FROM system.parts WHERE active AND database = currentDatabase() AND table = '{t}'") for t in (M, WT)}, lab_index=a.lab_index, index_build=ix, protocol=dict(cold=0, warm=a.warm, limit_s=TIMEOUT), predicates={k: list(v) for k, v in preds.items()}, records=[])
        fold = ("arrayJoin(arrayFold((acc, x) -> if(length(acc) > 0 AND x.1 <= acc[-1].2 + 1, arrayPushBack(arrayPopBack(acc), (acc[-1].1, greatest(acc[-1].2, x.2))), "
                "arrayPushBack(acc, x)), arraySort(groupArray((span_from, span_to))), CAST([], 'Array(Tuple(UInt32, UInt32))')))")
        for name, (kind, s) in preds.items():
            cond = (lambda col: like(col, s)) if kind == "like" else (lambda col: f"match({col}, '{s}')")
            Mq = f"(SELECT msg_id FROM {M} WHERE {cond('text')})"
            tag = dict(policy=p, pred=name, matched_messages=int(ch.val(f"SELECT count() FROM {Mq}")))
            cand = f"(SELECT span_id, input FROM {WT} PREWHERE {cond('input')})"
            hits = (f"(SELECT joinGet('span_tr', 'trace_id', span_id) AS trace_id, m, span_id FROM {cand} ARRAY JOIN JSONExtractArrayRaw(input) AS m "
                    f"WHERE {cond('m')} GROUP BY trace_id, m, span_id)")
            runs_w = (f"(SELECT span_id, span_id - min(span_id) OVER (PARTITION BY trace_id, m, grp) AS age FROM "
                      f"(SELECT trace_id, m, span_id, span_id - row_number() OVER (PARTITION BY trace_id, m ORDER BY span_id) AS grp FROM {hits}))")
            W = f"SELECT groupBitmapState(toUInt32(span_id)) FROM {runs_w} WHERE age >= {K - 1}"
            ch.q(f"CREATE OR REPLACE TABLE ex_ref ENGINE = Memory AS SELECT ({W}) AS bm", max_execution_time=EXAMPLE_REF_TIMEOUT, **NO_INDEX)
            card = int(ch.val("SELECT bitmapCardinality(bm) FROM ex_ref"))
            ch.q(f"CREATE OR REPLACE TABLE mset ENGINE = Set AS {Mq}")
            def res_plan(Mset, merged):
                if merged:
                    runs = f"(SELECT r.1 AS span_from, r.2 AS span_to FROM (SELECT {fold} AS r FROM {Isub} WHERE msg_id IN {Mset} GROUP BY trace_id, msg_id))"
                else:
                    runs = presence(f"(SELECT * FROM {Isub} WHERE msg_id IN {Mset})", "trace_id, msg_id, span_from AS lo, span_to AS hi, occ AS tie")
                return f"SELECT {expand(f'span_from + {K - 1}', 'span_to')} FROM {runs} WHERE span_to >= span_from + {K - 1}"
            hits_d = (f"(SELECT joinGet('span_tr', 'trace_id', span_id) AS trace_id, arrayJoin(arrayDistinct(arrayFilter(x -> {cond('x')}, "
                      f"JSONExtractArrayRaw(input)))) AS m, span_id FROM {cand})")
            runs_d = (f"(SELECT span_id, span_id - min(span_id) OVER (PARTITION BY trace_id, m, grp) AS age FROM "
                      f"(SELECT trace_id, m, span_id, span_id - row_number() OVER (PARTITION BY trace_id, m ORDER BY span_id) AS grp FROM {hits_d}))")
            P = {"W": (W, None), "W.distinct": (f"SELECT groupBitmapState(toUInt32(span_id)) FROM {runs_d} WHERE age >= {K - 1}", None), "I": (res_plan(Mq, False), None), "I.fold": (res_plan(Mq, True), None),
                 "I.2p": (res_plan("mset", False), f"CREATE OR REPLACE TABLE mset ENGINE = Set AS {Mq}"),
                 "I.2pfold": (res_plan("mset", True), f"CREATE OR REPLACE TABLE mset ENGINE = Set AS {Mq}")}
            rec0 = dict(tag, query="M", plan="M", **timed(ch, f"SELECT count() FROM {Mq}", a.store, 0, a.warm))
            r["records"].append(rec0); print(json.dumps(rec0), flush=True)
            for plan, (sql, pre) in P.items():
                rec = dict(tag, query="residence", K=K, plan=plan, cardinality=card)
                diff = int(ch.q(f"SELECT bitmapXorCardinality(({sql}), (SELECT bm FROM ex_ref))", max_execution_time=EXAMPLE_REF_TIMEOUT)[0])
                if diff:
                    res["mismatch"] = dict(rec, xor=diff)
                    fail(res, a, json.dumps(res["mismatch"]))
                count = f"SELECT bitmapCardinality(({sql}))"
                rec.update(set_equal=True, **(timed_two_step(ch, pre, count, a.warm) if pre else timed(ch, count, a.store, 0, a.warm)))
                r["records"].append(rec); print(json.dumps(rec), flush=True)
            Q = {"W": f"SELECT span_id, toInt64(arrayCount(m -> {cond('m')}, JSONExtractArrayRaw(input))) AS v FROM {cand} WHERE v > 0",
                 "I.expand": f"SELECT span_id, v FROM ({ctx_count(dict(i=Isub), Mq)['I.expand']}) WHERE span_id < {S}",
                 "I.expand.2p": f"SELECT span_id, v FROM ({ctx_count(dict(i=Isub), 'mset')['I.expand']}) WHERE span_id < {S}",
                 "A": ctx_count(dict(a=f"(SELECT * FROM a_{p} WHERE span_id < {S})"), Mq)["A"]}
            ch.q(f"CREATE OR REPLACE TABLE cnt_ref ENGINE = Memory AS {Q['W']}", max_execution_time=EXAMPLE_REF_TIMEOUT, **NO_INDEX)
            for plan, sql in Q.items():
                rec = dict(tag, query="ctx_count", plan=plan)
                rec.update(rel_diff(ch, "SELECT span_id, v FROM cnt_ref", sql, EXAMPLE_REF_TIMEOUT))
                if rec["rows_differ"]:
                    res["mismatch"] = rec
                    fail(res, a, json.dumps(rec))
                count = f"SELECT count(), sum(v) FROM ({sql})"
                rec.update(timed_two_step(ch, f"CREATE OR REPLACE TABLE mset ENGINE = Set AS {Mq}", count, a.warm) if plan.endswith(".2p") else timed(ch, count, a.store, 0, a.warm))
                r["records"].append(rec); print(json.dumps(rec), flush=True)
        save(res, a.out)


NO_INDEX = dict(use_skip_indexes=0)                                # a reference that no text index can affect
FALLBACK_LIMIT = 900                                              # seconds per execution of a fallback plan
FALLBACK_LENS = (8, 16, 32)


def stage_fallback(ch, a, res):
    """Substring predicates that cross a message boundary, which message-level evaluation cannot answer. Needles are cut around
    the separator of two adjacent messages of hash-chosen calls, 4 per length. Three plans, compared with W; a difference stops the run:
      W  the wide table; R  rebuild every call's request and search it;
      F  rebuild and search only the candidate calls: a match lies inside a message, or crosses a first boundary, and then the
         part of the needle before that boundary ends a message; so the candidates are the calls holding a message that contains
         the needle or ends with such a part. This reasoning needs a non-empty part: a needle that begins with '[' or ',' has
         none, so such a needle would go to R; none is sampled, so the results describe needles that start inside a message.
      P  no rebuilding, for a non-empty needle (lengths are in characters): a match lies inside a message, or crosses one boundary, at some comma of the needle: the part before it
         ends a message and the part after it begins the next one; both tests run on the dictionary and the adjacency on the
         call's message IDs. A match that crosses two boundaries contains a whole message between two commas of the needle, so
         P applies only if no message equals such a piece, and a needle that ends with ']' goes to F (recorded as p_applies).
    Warm runs on tables built beforehand; the join of message texts that R and F share is built outside the timing."""
    out = res.setdefault("fallback", {})
    for p in a.wide_policies:
        WT, M = f"wm_{p}", f"mm_{p}"
        res["load"][WT] = load(ch, a.store, WT, "wx", f"SELECT span_id, input FROM {src('wide', p)}", optimize=False)
        S = dense_spans(ch, a, res, WT)
        res["load"][M] = load(ch, a.store, M, "mx", f"SELECT msg_id, text FROM msg WHERE msg_id IN (SELECT msg_id FROM i_{p} WHERE span_from < {S})", optimize=False)
        t0 = time.perf_counter()
        ch.q(f"CREATE OR REPLACE TABLE msg_text ENGINE = Join(ANY, LEFT, msg_id) AS SELECT msg_id, text FROM {M}")
        r = out[p] = dict(spans=S, protocol=dict(cold=0, warm=a.warm, limit_s=FALLBACK_LIMIT), join_build_s=round(time.perf_counter() - t0, 2), records=[], needles={})
        rows = ch.val(f"SELECT span_id, msg_ids FROM a_{p} WHERE span_id < {S} AND length(msg_ids) >= 2 ORDER BY cityHash64(span_id, {a.seed} + 500) LIMIT 2000").splitlines()
        rebuilt = ctx_text("msg_ids", lambda c: f"joinGet('msg_text', 'text', {c})")
        needles, it = [], iter(rows)
        for L in FALLBACK_LENS:
            got = 0
            for line in it:
                span, ids = line.split("\t")
                ids = json.loads(ids)
                h = int(span) * 2654435761 + a.seed
                i = h % (len(ids) - 1)
                pair = ch.val(f"SELECT msg_id, base64Encode(text) FROM {M} WHERE msg_id IN ({ids[i]}, {ids[i + 1]})").splitlines()
                txt = {int(x.split("\t")[0]): base64.b64decode(x.split("\t")[1]).decode() for x in pair}
                if len(txt) < 2 and ids[i] != ids[i + 1]:
                    continue
                k = 1 + (h >> 8) % (L - 2)
                n = txt[ids[i]][-k:] + "," + txt[ids[i + 1]][:L - k - 1]
                if len(n) == L and n[0] not in "[," and n not in needles:
                    needles.append(n); got += 1
                if got == 4:
                    break
        if len(needles) != 4 * len(FALLBACK_LENS):
            fail(res, a, f"fallback: {len(needles)} needles, expected {4 * len(FALLBACK_LENS)}")
        for idx, n in enumerate(needles):
            b = lambda t: "base64Decode('" + base64.b64encode(t.encode()).decode() + "')"
            N = b(n)
            parts = [n[:i] for i, c in enumerate(n) if c == "," and i > 0] + ([n[:-1]] if n.endswith("]") else [])
            cond = " OR ".join([f"position(text, {N}) > 0"] + [f"endsWith(text, {b(x)})" for x in parts])
            Ma = f"(SELECT msg_id FROM {M} WHERE {cond})"
            cand = f"(SELECT span_id, msg_ids FROM a_{p} WHERE span_id < {S} AND arrayExists(x -> x IN {Ma}, msg_ids))"
            commas = [i for i, c in enumerate(n) if c == ","]
            middles = [n[i + 1:j] for i in commas for j in commas if j > i]
            whole, guard_ms = 0, 0.0                              # the guard is part of P's cost; the report adds guard_ms to P's time
            if middles:
                g = ch.q(f"SELECT count() FROM {M} WHERE text IN ({', '.join(b(x) for x in middles)})")
                whole, guard_ms = int(g[0]), elapsed_ms(g)
            p_applies = whole == 0 and not n.endswith("]")
            tests = [f"arrayExists(x -> x IN (SELECT msg_id FROM {M} WHERE position(text, {N}) > 0), msg_ids)"]
            for i in commas:
                if i > 0:
                    tests.append(f"arrayExists((x, y) -> x IN (SELECT msg_id FROM {M} WHERE endsWith(text, {b(n[:i])})) AND y IN (SELECT msg_id FROM {M} WHERE startsWith(text, {b(n[i + 1:])})), "
                                 f"arrayPopBack(msg_ids), arrayPopFront(msg_ids))")
            P = {"W": f"SELECT groupBitmapState(toUInt32(span_id)) FROM {WT} PREWHERE position(input, {N}) > 0",
                 "R": f"SELECT groupBitmapState(toUInt32(span_id)) FROM a_{p} WHERE span_id < {S} AND position({rebuilt}, {N}) > 0",
                 "F": f"SELECT groupBitmapState(toUInt32(span_id)) FROM {cand} WHERE position({rebuilt}, {N}) > 0"}
            if p_applies:
                P["P"] = f"SELECT groupBitmapState(toUInt32(span_id)) FROM a_{p} WHERE span_id < {S} AND ({' OR '.join(tests)})"
            ch.q(f"CREATE OR REPLACE TABLE ex_ref ENGINE = Memory AS SELECT ({P['W']}) AS bm", max_execution_time=FALLBACK_LIMIT)
            card = int(ch.val("SELECT bitmapCardinality(bm) FROM ex_ref"))
            ncand = int(ch.val(f"SELECT count() FROM {cand}"))
            r["needles"][str(idx)] = n
            for plan, sql in P.items():
                rec = dict(policy=p, needle=idx, length=len(n), plan=plan, cardinality=card, candidates=ncand, parts=len(parts), p_applies=p_applies, guard_ms=guard_ms)
                try:
                    diff = int(ch.q(f"SELECT bitmapXorCardinality(({sql}), (SELECT bm FROM ex_ref))", max_execution_time=FALLBACK_LIMIT)[0])
                    if diff:
                        res["mismatch"] = dict(rec, xor=diff)
                        fail(res, a, json.dumps(res["mismatch"]))
                    rec["set_equal"] = True
                    rec["warm_ms"] = [elapsed_ms(ch.q(f"SELECT bitmapCardinality(({sql}))", max_execution_time=FALLBACK_LIMIT)) for _ in range(a.warm)]
                except RuntimeError as e:
                    rec["error"] = over_limit(e)
                r["records"].append(rec); print(json.dumps(rec), flush=True)
            save(res, a.out)
        ch.q("DROP TABLE msg_text")


ROUTE_TRACES = [None, 1, 10, 100, 1000, 10000]                    # candidate sets: all spans, or this many traces chosen by hash


def stage_route(ch, a, res):
    """Latency of the two access paths over a grid of trace filters and matched sets: the test data of the routing rule (route.py).
    The paths must agree on the number of matching spans."""
    n_traces = res["traces"]
    for p in a.policies:
        T = dict(a=f"a_{p}", i=f"i_{p}")
        for k in ROUTE_TRACES:
            if k is not None and k >= n_traces:
                continue
            pred = c = None
            if k is not None:
                pred = "trace_id IN (%s)" % ch.val(f"SELECT arrayStringConcat(groupArray(toString(trace_id)), ',') FROM (SELECT trace_id FROM traces ORDER BY cityHash64(trace_id, 5) LIMIT {k})")
                c = int(ch.val(f"SELECT sum(span_hi - span_lo + 1) FROM traces WHERE {pred}"))
            for name, n in res["matched"].items():
                if name.startswith("c"):
                    continue
                P = ctx_any(T, f"(SELECT msg_id FROM matched WHERE name = '{name}')", pred, n)
                vals = set()
                for plan in ("A.exists", "A.unnest", "I.expand", "I.pruned"):
                    if plan in P:
                        agg, frm = P[plan]
                        rec = dict(policy=p, filter="all" if k is None else f"t{k}", traces=k, C=c, matched=name, n_matched=n, plan=plan,
                                   **timed(ch, f"SELECT bitmapCardinality({agg}) FROM {frm}", a.store, 0, 3))
                        res["route"].append(rec); vals.add(rec.get("value", "error"))
                if len(vals - {"error"}) > 1:
                    fail(res, a, f"route {p} {k} {name}: the paths disagree: {vals}")
            save(res, a.out)
            print("route", p, k, c, flush=True)


CODECS = {  # wide-table and dictionary text column: codec, and MergeTree compression-block settings
    "zstd3": ("ZSTD(3)", ""),
    "zstd3-64m": ("ZSTD(3)", ", min_compress_block_size = 16777216, max_compress_block_size = 67108864"),
    "zstd3-long27-64m": ("ZSTD(3, 27)", ", min_compress_block_size = 16777216, max_compress_block_size = 67108864"),
}


def stage_wcodec(ch, a, res):
    """Storage tuning of the wide table: consecutive requests of a trace share most of their text, so a compression block that
    spans many rows hides the redundancy on disk. The same codecs are applied to the message dictionary. For every variant:
    size, load time, substring scans, and reads of single rows (a large block must be decompressed to return one row)."""
    for p in a.wide_policies:
        S = int(ch.val(f"SELECT max(span_id) + 1 FROM {src('wide', p)}"))
        msg_sel = f"SELECT msg_id, text FROM msg WHERE msg_id IN (SELECT msg_id FROM i_{p} WHERE span_from < {S})"
        for v, (codec, settings) in CODECS.items():
            for side, table, cols, order, sel, col in (("wide", "wc", f"span_id UInt32, input String CODEC({codec})", "span_id", f"SELECT span_id, input FROM {src('wide', p)}", "input"),
                                                       ("dictionary", "mc", f"msg_id UInt32, text String CODEC({codec})", "msg_id", msg_sel, "text")):
                ch.q(f"DROP TABLE IF EXISTS {table} SYNC")
                ch.q(f"CREATE TABLE {table} ({cols}) ENGINE = MergeTree ORDER BY {order} SETTINGS min_bytes_for_wide_part = 0{settings}")
                evict(ch, a.store)
                t_ins = ch.q(f"INSERT INTO {table} {sel}", max_insert_threads=max(1, ch.base["max_threads"] // 2))[2]
                t_opt = ch.q(f"OPTIMIZE TABLE {table} FINAL")[2]
                rows, comp, raw = map(int, ch.val(f"SELECT sum(rows), sum(data_compressed_bytes), sum(data_uncompressed_bytes) FROM system.parts "
                                                  f"WHERE active AND database = currentDatabase() AND table = '{table}'").split("\t"))
                rec = dict(policy=p, codec=v, side=side, rows=rows, compressed_bytes=comp, uncompressed_bytes=raw, insert_s=round(t_ins, 2), optimize_s=round(t_opt, 2))
                key = "span_id" if side == "wide" else "msg_id"
                rec["scan"] = {k: timed(ch, f"SELECT count() FROM {table} WHERE position({col}, '{n}') > 0", a.store, a.cold, a.warm) for k, n in list(NEEDLES.items())[:3]}
                ids = ch.val(f"SELECT {key} FROM {table} ORDER BY cityHash64({key}, 3) LIMIT 10").split()
                rec["point"] = [timed(ch, f"SELECT length({col}) FROM {table} WHERE {key} = {x}", a.store, 1, 2) for x in ids]
                res["wcodec"].append(rec)
                print(json.dumps({k: rec[k] for k in ("policy", "codec", "side", "compressed_bytes", "uncompressed_bytes", "insert_s", "optimize_s")}), flush=True)
                save(res, a.out)
        ch.q("DROP TABLE wc SYNC"); ch.q("DROP TABLE mc SYNC")


def like(col, needle):
    esc = needle.replace("\\", "\\\\\\\\").replace("%", "\\\\%").replace("_", "\\\\_").replace("'", "\\'")
    return f"{col} LIKE '%{esc}%'"


def set_index(ch, store, table, col, spec):
    """Replace the single index `ix` of `table` by `spec` (None: no index); report build time and size."""
    ch.q(f"ALTER TABLE {table} DROP INDEX IF EXISTS ix", mutations_sync=2)
    if spec is None:
        return dict(build_s=0.0, index_bytes=0)
    evict(ch, store)
    cpu = server_cpu(store)
    t = ch.q(f"ALTER TABLE {table} ADD INDEX ix {col} TYPE {spec}")[2] + ch.q(f"ALTER TABLE {table} MATERIALIZE INDEX ix", mutations_sync=2)[2]
    size = int(ch.val(f"SELECT sum(data_compressed_bytes) FROM system.data_skipping_indices WHERE database = currentDatabase() AND table = '{table}' AND name = 'ix'"))
    return dict(build_s=round(t, 2), build_cpu_s=round(server_cpu(store) - cpu, 2), index_bytes=size)


BLOOM_BYTES = (4096, 16384, 65536, 262144)                        # n-gram Bloom filter sizes per granule, calibrated per side


def index_specs(v):
    """Index definitions to calibrate for index kind v (one, except for the Bloom filter)."""
    if v == "bloom4":
        return {b: f"ngrambf_v1(4, {b}, 3, 0) GRANULARITY 1" for b in BLOOM_BYTES}
    return {0: INDEXES[v]}


def sample_predicates(ch, a, M):
    """Substring predicates cut from the messages of dictionary table M, regular expressions and fixed needles, split into
    calibration and held-out ("eval") ones. Returns (name -> (kind, string), name -> source message of a sampled substring)."""
    preds, sources = {}, {}
    wiki = a.workload in TEXT_FIELD                               # wiki or chat: data-driven substrings only
    if a.preds_from and not wiki:                                 # E-DORD on SWE samples: the exact 15 + 30 predicates of an earlier run
        src = json.load(open(a.preds_from))["index"]
        if len(src) != 1:
            raise SystemExit("--preds-from: needs a run with exactly one index policy")
        rec = next(iter(src.values()))["predicates"]
        want = ({f"calib:sub{L}:{k}" for L in (8, 16, 32) for k in range(4)} | {f"eval:sub{L}:{k}" for L in (8, 16, 32) for k in range(6)}
                | {f"{sp}:re:{k}" for sp in ("calib", "eval") for k in range(len(REGEX[sp]))} | {f"eval:{k}" for k in NEEDLES})
        if set(rec) != want or not all(isinstance(v, str) and v for v in rec.values()):
            raise SystemExit("--preds-from: the run does not hold exactly the SWE calibration and held-out predicates")
        if any(rec[f"{sp}:re:{k}"] != x for sp in ("calib", "eval") for k, x in enumerate(REGEX[sp])) or any(rec[f"eval:{k}"] != v for k, v in NEEDLES.items()):
            raise SystemExit("--preds-from: the regular expressions or needles differ from this driver's")
        return {k: ("match" if ":re:" in k else "like", rec[k]) for k in sorted(want)}, {}
    if a.preds_from:                                              # E-SCALE: the exact 12 + 30 substrings of an earlier index run
        src = json.load(open(a.preds_from))["index"]
        if not (wiki and len(src) == 1):
            raise SystemExit("--preds-from: needs --workload wiki|chat and a run with exactly one index policy")
        rec = next(iter(src.values()))["predicates"]
        want = {f"calib:sub{L}:{k}" for L in (8, 16, 32) for k in range(4)} | {f"eval:sub{L}:{k}" for L in (8, 16, 32) for k in range(10)}
        if set(rec) != want or not all(isinstance(v, str) and v for v in rec.values()):
            raise SystemExit("--preds-from: the run does not hold exactly the 12 calibration and 30 held-out substrings")
        return {k: ("like", rec[k]) for k in sorted(want)}, {}
    for split, cls, per_len in (("calib", 0, 4), ("eval", 1, 10 if wiki else 6)):   # source messages: class 0 for calibration, class 1 held out
        for L in (8, 16, 32):
            seed = 100 * a.seed + L + cls
            if wiki:                                              # cut from the block text, never from the JSON envelope; 30 held out
                # chat: the envelope {"content":...,"role":"assistant"} contains the quote-free 8-character windows "assistan" and
                # "ssistant"; a candidate that is a substring of "assistant" could match there, so it is excluded before the LIMIT
                envelope = " AND position('assistant', s) = 0" if a.workload == "chat" else ""
                sql = (f"SELECT s, src FROM (SELECT substring(b, 1 + cityHash64(msg_id, {seed}) % (length(b) - {L}), {L}) AS s, min(msg_id) AS src "
                       f"FROM (SELECT msg_id, JSONExtractString(text, '{TEXT_FIELD[a.workload]}') AS b FROM {M} WHERE cityHash64(msg_id, {a.seed} + 77) % 50 = {cls}) "
                       f"WHERE length(b) > {L} + 2 GROUP BY s) WHERE match(s, '^[A-Za-z0-9 .:=()/-]+$'){envelope} ORDER BY cityHash64(s, {seed}) LIMIT {4 * per_len}")   # pool fixed in advance
            else:
                sql = (f"SELECT s, src FROM (SELECT substring(text, 1 + cityHash64(msg_id, {seed}) % (length(text) - {L}), {L}) AS s, min(msg_id) AS src FROM {M} "
                       f"WHERE length(text) > {L} + 2 AND cityHash64(msg_id, {a.seed} + 77) % 50 = {cls} GROUP BY s) WHERE match(s, '^[A-Za-z0-9 .:=()/-]+$') "
                       f"ORDER BY cityHash64(s, {seed}) LIMIT {2 * per_len}")
            rows = [l.split("\t") for l in ch.val(sql).splitlines()]
            rows = [x for x in rows if all(x[0] not in v[1] and v[1] not in x[0] for v in preds.values())][:per_len]     # no string shared between predicates
            preds.update({f"{split}:sub{L}:{k}": ("like", x[0]) for k, x in enumerate(rows)})
            sources.update({f"{split}:sub{L}:{k}": int(x[1]) for k, x in enumerate(rows)})
        if not wiki:
            preds.update({f"{split}:re:{k}": ("match", x) for k, x in enumerate(REGEX[split])})
    if not wiki:
        preds.update({f"eval:{k}": ("like", v) for k, v in NEEDLES.items()})
    else:                                                         # exactly 12 calibration and 30 held-out substrings, or stop
        want = {f"calib:sub{L}:{k}" for L in (8, 16, 32) for k in range(4)} | {f"eval:sub{L}:{k}" for L in (8, 16, 32) for k in range(10)}
        if set(preds) != want:
            raise SystemExit(f"--workload {a.workload}: predicate sample incomplete: missing {sorted(want - set(preds))}")
    return preds, sources


VARIANT_KINDS = {"none": 512, "ngrams3": 64}                      # index kind -> dictionary granularity (what the index stage's calibration chose in every run so far)


def stage_variants(ch, a, res):
    """Small-edit variants, recorded by ingest_run.py (a cache mark that moved, a line that changed): a dictionary that stores
    such a message as a splice of the message it replaced, against the dictionary that stores every message in full. For every
    held-out substring predicate the two must return the same messages, and the spans must equal a scan of the logged
    requests; a difference stops the run. Timed: the predicate on the dictionary alone, and the query end to end."""
    for p in a.wide_policies:
        r = res["variants"][p] = dict(load={}, index={}, records=[])
        W = src("wide", p)
        S = int(ch.val(f"SELECT max(span_id) + 1 FROM {W}"))
        used = f"(SELECT msg_id FROM i_{p} WHERE span_from < {S})"
        r["load"]["i"] = load(ch, a.store, "i_v", "i", f"SELECT * FROM i_{p} WHERE span_from < {S}", GRAN["i"])
        ch.q(f"CREATE OR REPLACE TABLE var ENGINE = MergeTree ORDER BY msg_id AS SELECT * FROM {src('variants')} WHERE msg_id IN {used} AND base_id IN {used}")
        ch.q("CREATE OR REPLACE TABLE vb ENGINE = Join(ANY, LEFT, base_id) AS SELECT base_id, groupArray((msg_id, p, s, ins, wb, wv)) AS vars FROM var GROUP BY base_id")
        full = f"SELECT msg_id, text FROM msg WHERE msg_id IN {used}"
        r["var"] = dict(zip(("variants", "bases_with_variants", "window_bytes", "inserted_bytes"),
                            map(int, ch.val("SELECT count(), uniqExact(base_id), sum(length(wv) + length(wb)), sum(length(ins)) FROM var").split("\t"))))
        r["var"]["variant_text_bytes"] = int(ch.val("SELECT sum(length(text)) FROM msg WHERE msg_id IN (SELECT msg_id FROM var)"))
        r["var"]["table_bytes"] = int(ch.val("SELECT sum(bytes_on_disk) FROM system.parts WHERE active AND database = currentDatabase() AND table = 'var'"))
        print("variants", p, json.dumps(r["var"]), flush=True)
        preds = None
        for v, g in VARIANT_KINDS.items():
            r["load"][f"full/{v}"] = load(ch, a.store, "mf", "mx", full, g, optimize=False)
            r["load"][f"bases/{v}"] = load(ch, a.store, "mb", "mx", full + " AND msg_id NOT IN (SELECT msg_id FROM var)", g, optimize=False)
            r["index"][v] = dict(full=set_index(ch, a.store, "mf", "text", INDEXES[v]), bases=set_index(ch, a.store, "mb", "text", INDEXES[v]))
            if preds is None:
                preds, r["predicate_sources"] = sample_predicates(ch, a, "mf")
                preds = r["predicates"] = {k: x[1] for k, x in preds.items() if k.startswith("eval") and x[0] == "like"}
                ch.q("CREATE OR REPLACE TABLE truth (pred String, bm AggregateFunction(groupBitmap, UInt32)) ENGINE = MergeTree ORDER BY pred")
                for name, needle in preds.items():                # ground truth: the logged requests
                    ch.q(f"INSERT INTO truth SELECT '{name}', groupBitmapState(span_id) FROM {W} WHERE {like('input', needle)}")
            for name, needle in preds.items():
                pred = lambda col: like(col, needle)
                M = {"full": matched(pred("text"), "mf"), "variants": matched_variants(pred, "mb", "var", "vb")}
                diff, n = map(int, ch.val(f"WITH (SELECT groupBitmapState(msg_id) FROM {M['full']}) AS x, (SELECT groupBitmapState(msg_id) FROM {M['variants']}) AS y "
                                          f"SELECT bitmapXorCardinality(x, y), bitmapCardinality(x)").split("\t"))
                tag = dict(policy=p, index=v, pred=name, matched_messages=n)
                if diff:
                    res["mismatch"] = dict(tag, messages_differ=diff)
                    fail(res, a, json.dumps(res["mismatch"]))
                for k, m in M.items():
                    r["records"].append(dict(tag, plan=f"dictionary:{k}", **timed(ch, f"SELECT uniqExact(msg_id) FROM {m}", a.store, a.cold, a.warm)))
                P = {f"F:{k}": ctx_any(dict(i="i_v"), m, None, 0)["I.expand"] for k, m in M.items()}
                r["records"] += run_plans(ch, P, f"SELECT bm FROM truth WHERE pred = '{name}'", a, tag, res)
            save(res, a.out)


def dict_load(p, S, order):
    """(DDL kind, SELECT) of the index stage's dictionary table for policy p and the spans below S. id: sorted by message ID, as
    in the paper's first runs. first: sorted by the first span (call) in which the message is open, ties by message ID, so that
    the messages a trace logs, e.g. the versions of an edited message, lie together, as the wide table keeps a trace's calls
    together; the relation (message ID, text) is the same."""
    if order == "first":
        return "mxf", (f"SELECT m.msg_id AS msg_id, f.fs AS first_span, m.text AS text FROM msg AS m INNER JOIN (SELECT msg_id, toUInt32(min(span_from)) AS fs "
                       f"FROM i_{p} WHERE span_from < {S} GROUP BY msg_id) AS f ON f.msg_id = m.msg_id")
    return "mx", f"SELECT msg_id, text FROM msg WHERE msg_id IN (SELECT msg_id FROM i_{p} WHERE span_from < {S})"


def dict_checks(ch, M, p, S, order):
    """The dictionary table M holds every message open in a span below S exactly once, with the text of the base dictionary;
    with order first, each row's first_span is the first span of its message's intervals. Returns the counts and ok."""
    ids = f"(SELECT DISTINCT msg_id FROM i_{p} WHERE span_from < {S})"
    q = {"rows": f"SELECT count() FROM {M}", "distinct_ids": f"SELECT uniqExact(msg_id) FROM {M}", "source_ids": f"SELECT count() FROM {ids}",
         "missing": f"SELECT count() FROM {ids} WHERE msg_id NOT IN (SELECT msg_id FROM {M})", "extra": f"SELECT count() FROM {M} WHERE msg_id NOT IN {ids}",
         "msg_duplicate_ids": "SELECT count() - uniqExact(msg_id) FROM msg",
         "text_differs": f"SELECT count() FROM {M} AS t INNER JOIN msg AS m ON m.msg_id = t.msg_id WHERE m.text != t.text"}
    if order == "first":
        q["wrong_first_span"] = (f"SELECT count() FROM {M} AS t INNER JOIN (SELECT msg_id, toUInt32(min(span_from)) AS fs FROM i_{p} WHERE span_from < {S} "
                                 f"GROUP BY msg_id) AS f ON f.msg_id = t.msg_id WHERE t.first_span != f.fs")
    out = {k: int(ch.val(v)) for k, v in q.items()}
    out["ok"] = (out["rows"] == out["distinct_ids"] == out["source_ids"] and out["missing"] == out["extra"] == out["msg_duplicate_ids"] == out["text_differs"] == 0
                 and out.get("wrong_first_span", 0) == 0)
    return out


def stage_index(ch, a, res):
    """Substring and regular-expression predicates with the same index kind on both sides: the wide context column and the
    message dictionary. Per side and per index kind, the table granularity (dictionary) and the Bloom-filter size (both sides)
    are chosen on calibration predicates; held-out predicates are cut from a disjoint class of messages, and the source message
    of every predicate is recorded. The wide table keeps adaptive granules (about 10 MB of text each): granularities of 1 and 8
    rows exceeded the time limit on the calibration predicates in the first run. Every result is compared with a scan of the
    wide table that uses no index. Two more plans answer "is this normalization?": F+N.inv maps the matched messages through
    the normalized membership table, and R.reconstruct rebuilds every request from the array and the dictionary (held in
    memory) before it evaluates the predicate, as a store that deduplicates messages but queries span inputs does."""
    R = res["index"]
    cost = lambda t: math.sqrt(statistics.median(t["cold_ms"]) * statistics.median(t["warm_ms"]))

    def score(query, names):                                      # geometric mean over the calibration predicates; None if one fails
        ts = [timed(ch, query(n), a.store, 2, 3) for n in names]
        return None if any("error" in t for t in ts) else round(math.exp(statistics.fmean(math.log(cost(t)) for t in ts)), 3)

    for p in a.index_policies:
        r = R[p] = dict(load={}, calib={}, variants={}, records=[])
        S = int(ch.val(f"SELECT max(span_id) + 1 FROM {src('wide', p)}"))
        W, M = f"wx_{p}", f"mx_{p}"
        msg_sel = f"SELECT msg_id, text FROM msg WHERE msg_id IN (SELECT msg_id FROM i_{p} WHERE span_from < {S})"
        # --dict-order first: the dictionary is stored sorted by the first call that logged each message (the wide table is
        # sorted by call); this tends to co-locate nearby first appearances and the versions of a message edited within a trace.
        # Message IDs, the rows and every query are unchanged. id: sorted by message ID, as in the paper's runs.
        mkind, msel = dict_load(p, S, a.dict_order)
        # the two text tables are left as inserted (several parts): an index is then materialized on the parts in parallel
        r["load"][W] = load(ch, a.store, W, "wx", f"SELECT span_id, input FROM {src('wide', p)}", optimize=False)
        r["load"][M] = load(ch, a.store, M, mkind, msel, optimize=False)
        r["dict_checks"] = {"initial": dict_checks(ch, M, p, S, a.dict_order)}
        if not r["dict_checks"]["initial"]["ok"]:
            fail(res, a, "dictionary table check: " + json.dumps(r["dict_checks"]))
        r["load"]["a"] = load(ch, a.store, f"a_x_{p}", "a", f"SELECT * FROM a_{p} WHERE span_id < {S}", GRAN["a"])
        r["load"]["i"] = load(ch, a.store, f"i_x_{p}", "i", f"SELECT * FROM i_{p} WHERE span_from < {S}", GRAN["i"])
        r["load"]["ni"] = load(ch, a.store, f"ni_x_{p}", "ni", f"SELECT * FROM ni_{p} WHERE span_id < {S}", GRAN["ni"])
        ch.q(f"CREATE OR REPLACE TABLE msg_text ENGINE = Join(ANY, LEFT, msg_id) AS {msg_sel}")
        T = dict(a=f"a_x_{p}", i=f"i_x_{p}", ni=f"ni_x_{p}")
        rebuilt = "(SELECT span_id, " + ctx_text("msg_ids", lambda c: f"joinGet('msg_text', 'text', {c})") + f" AS input FROM {T['a']} SETTINGS max_block_size = 256)"

        preds, r["predicate_sources"] = sample_predicates(ch, a, M)
        cond = lambda name, col: like(col, preds[name][1]) if preds[name][0] == "like" else f"match({col}, '{preds[name][1]}')"
        r["predicates"] = {k: v[1] for k, v in preds.items()}

        ch.q("CREATE OR REPLACE TABLE truth (pred String, bm AggregateFunction(groupBitmap, UInt32)) ENGINE = MergeTree ORDER BY pred")
        for name in preds:                                        # ground truth: scan of the wide table without any index
            ch.q(f"INSERT INTO truth SELECT '{name}', groupBitmapState(span_id) FROM {W} WHERE {cond(name, 'input')}")
        r["truth_spans"] = {l.split("\t")[0]: int(l.split("\t")[1]) for l in ch.val("SELECT pred, bitmapCardinality(bm) FROM truth ORDER BY pred").splitlines()}
        r["matched_messages"] = {name: int(ch.val(f"SELECT count() FROM {M} WHERE {cond(name, 'text')}")) for name in preds}
        calib = [n for n in preds if n.startswith("calib")]
        fact = lambda name: f"SELECT bitmapCardinality({expand('span_from', 'span_to')}) FROM {T['i']} WHERE msg_id IN (SELECT msg_id FROM {M} WHERE {cond(name, 'text')})"
        wide = lambda name: f"SELECT bitmapCardinality(groupBitmapState(span_id)) FROM {W} WHERE {cond(name, 'input')}"

        for v in a.indexes:
            c = r["calib"][v] = dict(wide={}, factored={})
            # wide side: only the index parameters vary
            for key, spec in index_specs(v).items():
                c["wide"][key] = dict(set_index(ch, a.store, W, "input", spec), score=score(wide, calib))
            wkey = min((k for k in c["wide"] if c["wide"][k]["score"] is not None), key=lambda k: c["wide"][k]["score"])
            if len(c["wide"]) > 1:
                set_index(ch, a.store, W, "input", index_specs(v)[wkey])
            # dictionary side: granularity and index parameters
            for g in (64, 512, 8192):
                load(ch, a.store, M, mkind, msel, g, optimize=False)
                for key, spec in index_specs(v).items():
                    c["factored"][f"{g}/{key}"] = dict(set_index(ch, a.store, M, "text", spec), score=score(fact, calib))
            fkey = min((k for k in c["factored"] if c["factored"][k]["score"] is not None), key=lambda k: c["factored"][k]["score"])
            g, key = (int(x) for x in fkey.split("/"))
            r["load"][f"{M}/{v}"] = load(ch, a.store, M, mkind, msel, g, optimize=False)
            r["dict_checks"][v] = dict_checks(ch, M, p, S, a.dict_order)
            if not r["dict_checks"][v]["ok"]:
                fail(res, a, "dictionary table check: " + json.dumps(r["dict_checks"][v]))
            r["variants"][v] = dict(wide=dict(c["wide"][wkey], param=wkey), factored=dict(set_index(ch, a.store, M, "text", index_specs(v)[key]), granularity=g, param=key))
            print("index variant", p, v, json.dumps(r["variants"][v]), flush=True)
            for name in preds:
                Mq = f"(SELECT msg_id FROM {M} WHERE {cond(name, 'text')})"
                P = {"W": ("groupBitmapState(span_id)", f"{W} WHERE {cond(name, 'input')}")}
                P.update({"F+" + k: x for k, x in ctx_any(T, Mq, None, 1 << 30).items() if k in ("I.expand", "A.unnest", "N.inv")})
                if v == "none":                                   # independent of the index: the dictionary is read through the in-memory table
                    P["R.reconstruct"] = ("groupBitmapState(span_id)", f"{rebuilt} WHERE {cond(name, 'input')}")
                tag = dict(policy=p, index=v, pred=name, matched_messages=r["matched_messages"][name])
                r["records"] += run_plans(ch, P, f"SELECT bm FROM truth WHERE pred = '{name}'", a, tag, res)
                r["records"].append(dict(tag, plan="msg_scan", **timed(ch, f"SELECT count() FROM {M} WHERE {cond(name, 'text')}", a.store, a.cold, a.warm)))
            save(res, a.out)


def stage_lc(ch, a, res):
    """Engine-native deduplication as a baseline: the request stored as an array of dictionary-encoded messages,
    Array(LowCardinality(String)), in the span table itself. The engine then keeps each distinct message of a part's dictionary
    once and evaluates a predicate on the distinct messages of every block it reads. No index. The held-out predicates of the
    index stage are run on the wide text column (W), on this table (W.lc) and on the factored form; every result is compared
    with a scan of the wide text column."""
    for p in a.index_policies:
        r = res["lc"][p] = dict(load={}, records=[])
        S = int(ch.val(f"SELECT max(span_id) + 1 FROM {src('wide', p)}"))
        W, WL, M = f"wx_{p}", f"wl_{p}", f"mx_{p}"
        r["load"][W] = load(ch, a.store, W, "wx", f"SELECT span_id, input FROM {src('wide', p)}", optimize=False)
        r["load"][WL] = load(ch, a.store, WL, "wl", f"SELECT span_id, JSONExtractArrayRaw(input) FROM {src('wide', p)}", optimize=False)
        r["load"][M] = load(ch, a.store, M, "mx", f"SELECT msg_id, text FROM msg WHERE msg_id IN (SELECT msg_id FROM i_{p} WHERE span_from < {S})", 512, optimize=False)
        r["load"]["i"] = load(ch, a.store, f"i_x_{p}", "i", f"SELECT * FROM i_{p} WHERE span_from < {S}", GRAN["i"])
        preds, r["predicate_sources"] = sample_predicates(ch, a, M)
        preds = r["predicates"] = {k: v for k, v in preds.items() if k.startswith("eval")}
        cond = lambda name, col: like(col, preds[name][1]) if preds[name][0] == "like" else f"match({col}, '{preds[name][1]}')"
        ch.q("CREATE OR REPLACE TABLE truth (pred String, bm AggregateFunction(groupBitmap, UInt32)) ENGINE = MergeTree ORDER BY pred")
        for name in preds:
            ch.q(f"INSERT INTO truth SELECT '{name}', groupBitmapState(span_id) FROM {W} WHERE {cond(name, 'input')}")
        for name in preds:
            Mq = f"(SELECT msg_id FROM {M} WHERE {cond(name, 'text')})"
            P = {"W": ("groupBitmapState(span_id)", f"{W} WHERE {cond(name, 'input')}"),
                 "W.lc": ("groupBitmapState(span_id)", f"{WL} WHERE arrayExists(x -> {cond(name, 'x')}, messages)"),
                 "F+I.expand": ctx_any(dict(i=f"i_x_{p}"), Mq, None, 0)["I.expand"]}
            r["records"] += run_plans(ch, P, f"SELECT bm FROM truth WHERE pred = '{name}'", a, dict(policy=p, index="none", pred=name), res)
        save(res, a.out)


def stage_diag(ch, a, res):
    """Diagnostic: does the server read a small dictionary with all its threads? The same substring scans on the dictionary
    (three granularities) and on the wide table, with the default thresholds for concurrent reads and with both set to zero."""
    p = a.index_policies[0]
    S = int(ch.val(f"SELECT max(span_id) + 1 FROM {src('wide', p)}"))
    load(ch, a.store, "wd", "wx", f"SELECT span_id, input FROM {src('wide', p)}", optimize=False)
    zero = dict(merge_tree_min_rows_for_concurrent_read=0, merge_tree_min_bytes_for_concurrent_read=0)
    assert a.read_thresholds == "default", "run this stage with --read-thresholds default"
    def run(table, col, extra):
        out = []
        for needle in list(NEEDLES.values())[:3]:
            ts = []
            for _ in range(5):
                _, summ, wall = ch.q(f"SELECT count() FROM {table} WHERE position({col}, '{needle}') > 0", **extra)
                ts.append(round(summ.get("elapsed_ns", wall * 1e9) / 1e6, 2))
            out.append(statistics.median(ts))
        return out
    for g in (64, 512, 8192):
        st = load(ch, a.store, "md", "mx", f"SELECT msg_id, text FROM msg WHERE msg_id IN (SELECT msg_id FROM i_{p} WHERE span_from < {S})", g, optimize=False)
        parts = int(ch.val("SELECT count() FROM system.parts WHERE active AND database = currentDatabase() AND table = 'md'"))
        res["diag"].append(dict(table="dictionary", granularity=g, rows=st["rows"], text_bytes=st["uncompressed_bytes"], parts=parts, default_ms=run("md", "text", {}), zero_thresholds_ms=run("md", "text", zero),
                                one_thread_ms=run("md", "text", dict(max_threads=1))))
        print(json.dumps(res["diag"][-1]), flush=True)
    res["diag"].append(dict(table="wide", default_ms=run("wd", "input", {}), zero_thresholds_ms=run("wd", "input", zero), one_thread_ms=run("wd", "input", dict(max_threads=1))))
    print(json.dumps(res["diag"][-1]), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--store", required=True, help="the server's data path (to evict its files from the page cache)")
    ap.add_argument("--data", required=True, help="converter output directory (the server's user_files_path)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--policies", default="prefix,window32,amortized32,elide5")
    ap.add_argument("--wide-policies", default="prefix,window32")
    ap.add_argument("--index-policies", default="", help="policies of the index stage (default: the wide policies)")
    ap.add_argument("--indexes", default=",".join(INDEXES), help="index kinds of the index stage")
    ap.add_argument("--stages", default="base,calib,build,any,point,wide")
    ap.add_argument("--ckpt", type=int, default=32, help="checkpoint interval in spans")
    ap.add_argument("--gran", default="", help="granularities frozen by an earlier calibration of this policy, as d=8192:i=8192:... (then omit the calib stage)")
    ap.add_argument("--seed", type=int, default=0, help="index stage: seed of the sampled substring predicates")
    ap.add_argument("--read-thresholds", default="zero", choices=["zero", "default"], help="the server's thresholds for reading a table with several threads")
    ap.add_argument("--mixed-preds", type=int, default=30, help="mixed stage: the number of held-out predicates the sample must yield")
    ap.add_argument("--check-timeout", type=int, default=TIMEOUT, help="any stage: seconds for a plan's equality check (timing keeps the time limit)")
    ap.add_argument("--mixed-index", default="", help="mixed stage: this index kind on the dictionary and on the wide table")
    ap.add_argument("--mixed-reduced", action="store_true", help="mixed stage: K = 3 only")
    ap.add_argument("--mixed-wide-gran", type=int, default=8192, help="mixed stage: granularity of the wide table")
    ap.add_argument("--mixed-merge", action="store_true", help="mixed stage: OPTIMIZE FINAL both text tables before their indexes are built")
    ap.add_argument("--lab-wide-gran", type=int, default=8192, help="planlab stage: granularity of the wide table")
    ap.add_argument("--lab-index", default="", help="planlab stage: this index kind on the dictionary and on the wide table")
    ap.add_argument("--any-filters", default="", help="any stage: only these filters")
    ap.add_argument("--any-plans", default="", help="any stage: only these plans, with the reference plan")
    ap.add_argument("--dict-order", default="id", choices=["id", "first"], help="index stage: physical order of the dictionary table (first: by the first call that logged the message)")
    ap.add_argument("--preds-from", default="", help="index stage (wiki, chat): use the substrings recorded by this earlier index run instead of sampling (E-SCALE)")
    ap.add_argument("--workload", default="swe", choices=["swe", "wiki", "chat"], help="wiki, chat: E-WIKI / E-CHAT needles and data-driven predicates only (see WIKI_NEEDLES, CHAT_NEEDLES)")
    ap.add_argument("--cold", type=int, default=2)
    ap.add_argument("--warm", type=int, default=5)
    a = ap.parse_args()
    split = lambda x: [p for p in x.replace(":", ",").split(",") if p]     # ':' also separates: sbatch --export splits on commas
    a.policies, a.wide_policies, CKPT = split(a.policies), split(a.wide_policies), a.ckpt
    a.index_policies, a.indexes = split(a.index_policies) or a.wide_policies, split(a.indexes)
    a.any_filters, a.any_plans = split(a.any_filters), split(a.any_plans)
    a.mixed_ks = EXAMPLE_K[:1] if a.mixed_reduced else EXAMPLE_K
    GRAN.update({k: int(v) for k, v in (kv.split("=") for kv in a.gran.split(":") if kv)})
    if a.workload in TEXT_FIELD:                                  # the chat workload allows the same stages as wiki
        refused = [st for st in a.stages.replace(":", ",").split(",") if st not in WIKI_STAGES]
        if refused:
            raise SystemExit(f"--workload {a.workload}: stages {refused} use SWE-specific predicates")
        needles, token = (WIKI_NEEDLES, WIKI_COUNT_TOKEN) if a.workload == "wiki" else (CHAT_NEEDLES, CHAT_COUNT_TOKEN)
        NEEDLES.clear(); NEEDLES.update(needles)
        AGG_NEEDLE, COUNT_TOKEN = next(iter(needles)), token
        VALUES["n_err"] = f"toUInt32(countSubstrings(text, '{token}'))"
    ch = CH(a.port, a.threads, a.read_thresholds)
    for _ in range(120):                                          # wait for the server
        try:
            version = ch.val("SELECT version()"); break
        except (OSError, RuntimeError):
            time.sleep(1)
    else:
        raise SystemExit("ClickHouse server did not come up; see its error log")
    res = dict(meta=dict(clickhouse=version, host=platform.node(), threads=a.threads, ckpt=a.ckpt, cold=a.cold, warm=a.warm, timeout_s=TIMEOUT, check_timeout_s=a.check_timeout, settings=ch.base,
                         data=os.path.basename(os.path.normpath(a.data)), workload=a.workload, policies=a.policies, wide_policies=a.wide_policies,
                         index_policies=a.index_policies, indexes=a.indexes, stages=a.stages.replace(":", ",").split(","), seed=a.seed, dict_order=a.dict_order, pair_id=os.environ.get("FCX_PAIR_ID", ""), pair_order=os.environ.get("FCX_PAIR_ORDER", ""),
                         **({"preds_from": os.path.basename(a.preds_from), "preds_from_md5": hashlib.md5(open(a.preds_from, "rb").read()).hexdigest()} if a.preds_from else {})),   # run identity
               convert=json.load(open(f"{a.data}/convert_stats.json")), load={}, msg_scan={}, matched={}, filters={}, calib={}, checks={}, matched_stats={},
               any=[], point=[], agg=[], bytes={}, selfcheck={}, life=[], variants={}, lc={}, diag=[], route=[], wide=[], wcodec=[], index={}, cache_check={})
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    for stage in a.stages.replace(":", ",").split(","):           # ':' also separates: sbatch --export splits on commas
        t0 = time.time()
        globals()["stage_" + stage](ch, a, res)
        if stage == "build":                                      # show that the cold protocol empties the page cache and a query refills it
            c = res["cache_check"]
            c["before_evict"] = cached_fraction(a.store); evict(ch, a.store); c["after_evict"] = cached_fraction(a.store)
            ch.q(f"SELECT count() FROM a_{a.policies[0]} WHERE has(msg_ids, 0)"); c["after_scan"] = cached_fraction(a.store)
            c["evict_statements"] = list(EVICT)
        res["meta"][stage + "_s"] = round(time.time() - t0, 1)
        res["meta"]["granularity"] = dict(GRAN)
        save(res, a.out)
        print(f"stage {stage} done in {res['meta'][stage + '_s']}s", flush=True)
