"""E-DUCK: the content-query comparison of the paper on a second engine, DuckDB, without text indexes (DuckDB has none for
substrings). Same capture sample, same stores, same 30 held-out predicates as the paper's ClickHouse run of that sample.

Tables, built in one DuckDB database file on node-local disk (load times and file growth are recorded):
  wide  (span_id, input)          the logged requests, from <data>/<manager>/wide.parquet
  iv    (msg_id, span_from, span_to, occ)   occurrence intervals replayed from <data>/<manager>/transitions.parquet as
        fcx/query.py runs() does: an open paired with its close; a run without a close ends with its trace (traces.parquet);
        sorted by msg_id
  msg   (msg_id, text)            the dictionary messages that occur in this manager's log, sorted by msg_id
Plans (fixed in advance):
  W         SELECT count(*) FROM wide WHERE <pred on input>                                    (span_id is unique in wide)
  F.expand  count(DISTINCT span) over unnest(range(span_from, span_to + 1)) of the intervals of the matching messages
            (the paper's fixed plan, intervals expanded to a span set)
  F.union   the covered length of the union of those intervals (a sort-based sweep with a total order), no expansion
A substring predicate is contains(col, s); a regular expression is regexp_matches(col, re) (RE2, partial match, as ClickHouse's
match()). Every plan's count must equal the ClickHouse truth count recorded for that predicate (a scan of the wide table in the
paper's run) and the W count of this run; a difference stops the run. Each query runs --cold times cold (connection closed, the
OS page cache of the database file evicted with posix_fadvise DONTNEED, connection reopened; only the query is timed) and
--warm times warm on one connection. A query over --timeout seconds is interrupted and recorded as over the limit.

usage: duck_bench.py --data <capture dir> --manager prefix --preds results/duck/preds_<sample>.json --work <node-local dir>
                     --threads 16 --mem-gb 60 --out <json> [--cold 2 --warm 3 --timeout 120]
"""
import argparse, json, os, statistics, threading, time

import duckdb


def connect(a, path):
    con = duckdb.connect(path)
    con.execute(f"SET threads = {a.threads}; SET memory_limit = '{a.mem_gb}GB'; SET temp_directory = '{a.work}/duck_tmp'")
    return con


def evict(path):
    for p in (path, path + ".wal"):
        if os.path.exists(p):
            fd = os.open(p, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)


def run_one(con, sql, args, timeout):
    timer = threading.Timer(timeout, con.interrupt)
    t0 = time.perf_counter(); timer.start()
    try:
        v = con.execute(sql, args).fetchone()[0]
    except (duckdb.InterruptException, duckdb.Error) as e:
        if isinstance(e, duckdb.InterruptException) or "INTERRUPT" in str(e).upper():
            return None, None
        raise
    finally:
        timer.cancel(); timer.join()                              # a callback already running finishes before the next query
    return int(v), (time.perf_counter() - t0) * 1000


def plans(kind):
    p = "contains({c}, ?)" if kind == "like" else "regexp_matches({c}, ?)"
    m = f"(SELECT msg_id FROM msg WHERE {p.format(c='text')})"
    return {
        "W": f"SELECT count(*) FROM wide WHERE {p.format(c='input')}",
        "F.expand": f"SELECT count(DISTINCT s) FROM (SELECT unnest(range(span_from, span_to + 1)) AS s FROM iv WHERE msg_id IN {m})",
        "F.union": ("SELECT coalesce(sum(e - b + 1), 0) FROM (SELECT min(span_from) AS b, max(span_to) AS e FROM ("
                    "SELECT span_from, span_to, sum(is_new) OVER (ORDER BY span_from, span_to, occ ROWS UNBOUNDED PRECEDING) AS grp FROM ("
                    "SELECT span_from, span_to, occ, CASE WHEN span_from > 1 + coalesce(max(span_to) OVER (ORDER BY span_from, span_to, occ "
                    f"ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), -2) THEN 1 ELSE 0 END AS is_new FROM iv WHERE msg_id IN {m})) GROUP BY grp)"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--manager", required=True); ap.add_argument("--preds", required=True)
    ap.add_argument("--work", required=True); ap.add_argument("--threads", type=int, required=True); ap.add_argument("--mem-gb", type=int, required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--cold", type=int, default=2); ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=120)
    a = ap.parse_args()
    P = json.load(open(a.preds))
    if P["manager"] != a.manager or os.path.basename(os.path.normpath(a.data)) != P["sample"]:
        raise SystemExit(f"predicate file is for {P['sample']}/{P['manager']}")
    os.makedirs(f"{a.work}/duck_tmp", exist_ok=True)
    path = f"{a.work}/fcx.duckdb"
    if os.path.exists(path):
        raise SystemExit(f"refusing to reuse {path}")
    D, m = a.data, a.manager
    con = connect(a, path)
    res = dict(meta=dict(duckdb=duckdb.__version__, sample=P["sample"], manager=m, threads=a.threads, mem_gb=a.mem_gb, cold=a.cold, warm=a.warm,
                         timeout_s=a.timeout, preds_source=P["source"], preds_source_md5=P["source_md5"]), load={}, records=[])
    build = {
        "wide": f"CREATE TABLE wide AS SELECT span_id, input FROM read_parquet('{D}/{m}/wide.parquet')",
        "iv": (f"CREATE TABLE iv AS SELECT g.m AS msg_id, g.f::BIGINT AS span_from, (CASE WHEN g.nc > 0 THEN g.c::BIGINT - 1 ELSE tr.span_hi::BIGINT END) AS span_to, g.occ "
               f"FROM (SELECT trace_id, occ, any_value(msg_id) AS m, min(span_id) FILTER (WHERE op = 1) AS f, min(span_id) FILTER (WHERE op = -1) AS c, "
               f"count(*) FILTER (WHERE op = -1) AS nc FROM read_parquet('{D}/{m}/transitions.parquet') GROUP BY trace_id, occ) AS g "
               f"JOIN read_parquet('{D}/traces.parquet') AS tr ON tr.trace_id = g.trace_id ORDER BY msg_id, span_from"),
        "msg": f"CREATE TABLE msg AS SELECT msg_id, text FROM read_parquet('{D}/messages.parquet') WHERE msg_id IN (SELECT msg_id FROM iv) ORDER BY msg_id",
    }
    for name, sql in build.items():
        s0 = os.path.getsize(path); t0 = time.perf_counter()
        con.execute(sql); con.execute("CHECKPOINT")
        res["load"][name] = dict(seconds=time.perf_counter() - t0, file_growth_bytes=os.path.getsize(path) - s0,
                                 rows=con.execute(f"SELECT count(*) FROM {name}").fetchone()[0])
    opens = con.execute(f"SELECT count(*) FROM read_parquet('{D}/{m}/transitions.parquet') WHERE op = 1").fetchone()[0]
    spans = con.execute("SELECT count(*), count(DISTINCT span_id) FROM wide").fetchone()
    bad = con.execute("SELECT count(*) FROM iv WHERE span_from IS NULL OR span_to < span_from").fetchone()[0]
    res["checks"] = dict(iv_rows_minus_opens=res["load"]["iv"]["rows"] - opens, wide_rows_minus_distinct_spans=spans[0] - spans[1], bad_intervals=bad)
    if any(res["checks"].values()):
        raise SystemExit(f"store checks failed: {res['checks']}")
    print("loaded", json.dumps(res["load"]), flush=True)
    for name, pr in P["predicates"].items():
        for plan, sql in plans(pr["kind"]).items():
            rec = dict(pred=name, kind=pr["kind"], plan=plan, cold_ms=[], warm_ms=[], truth_spans=pr["truth_spans"])

            def check(v):                                         # every successful repetition is checked at once
                if v != pr["truth_spans"]:
                    res["mismatch"] = dict(rec, count=v)
                    json.dump(res, open(a.out, "w"), indent=1)
                    raise SystemExit(f"mismatch: {name} {plan} count {v} vs ClickHouse truth {pr['truth_spans']}")
            for _ in range(a.cold):
                con.close(); evict(path); con = connect(a, path)
                v, ms = run_one(con, sql, [pr["s"]], a.timeout)
                if v is None:
                    rec["over_limit"] = True; break
                check(v); rec["cold_ms"].append(ms)
            if not rec.get("over_limit"):
                for _ in range(a.warm):
                    v, ms = run_one(con, sql, [pr["s"]], a.timeout)
                    if v is None:
                        rec["over_limit"] = True; break
                    check(v); rec["warm_ms"].append(ms)
            rec["count"] = pr["truth_spans"] if rec["cold_ms"] or rec["warm_ms"] else None
            rec["equal"] = bool(rec["cold_ms"] or rec["warm_ms"])
            res["records"].append(rec)
            print(name, plan, rec["count"], rec.get("over_limit", False), [round(x) for x in rec["warm_ms"]], flush=True)
    con.close()
    res["db_file_bytes"] = os.path.getsize(path)
    sp = {}
    for plan in ("F.expand", "F.union"):
        w = {r["pred"]: r for r in res["records"] if r["plan"] == "W"}
        f = {r["pred"]: r for r in res["records"] if r["plan"] == plan}
        ok = [p for p in w if not w[p].get("over_limit") and not f[p].get("over_limit")]
        for mode in ("warm_ms", "cold_ms"):
            s = [statistics.median(w[p][mode]) / statistics.median(f[p][mode]) for p in ok]
            sp[f"{plan}/{mode[:4]}"] = dict(n=len(s), median=statistics.median(s) if s else None, min=min(s, default=None), max=max(s, default=None),
                                            wide_faster=sum(x < 1 for x in s))
    res["speedup"] = sp
    os.remove(path)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(sp, indent=1)); print("DUCK_BENCH_OK")


if __name__ == "__main__":
    main()
