"""E-ONLINE and E-DUCK summaries; reviewed before their output is used. Every number the paper may quote is computed here.

E-ONLINE inputs: results/online_v1/<data>__<manager>__k<K>.json for every line "<old|new> <data> <manager> <K>" of
cluster/tasks/online.txt (fcx/ingest_online.py). Checks per file: identity (data, manager, interleave = K), seed 1, budget 64,
checkpoint 32, the replay check string, requests equal to the stream's request count from an independent artifact (E-ING:
results/ing_s1_v1/<data>__<manager>.json; E-WIKI: results/wiki/wiki_s1/convert_stats.json), finite nonnegative latencies with
p50 <= p90 <= p99 <= p99.9 <= max, max_active_traces an integer in 1..min(K, 3,000) (completed traces are dropped before the
state is sampled, so at large K the maximum can stay below K), and finite nonnegative RSS and distinct-message counts.
Reported: at K = 64 over the 17 seed-1 streams, min-max ranges of the latency percentiles (microseconds), of the derivation
throughput (MB/s of request text, request_text only) and of the active-trace payload (mean per active trace, and the maximum
total); per sweep stream and K, the same quantities.

E-DUCK inputs: results/duck_v1/<sample>__prefix.json and results/duck/preds_<sample>.json for the samples present among
cap_swea_s1, cap_smith_s1, cap_oh_s1, cap_gym_s1 (fcx/duck_bench.py). Checks per run: meta (DuckDB 1.5.6, 16 threads, 2 cold, 3
warm, timeout 120 s, the predicate file's source and md5), no mismatch, the store checks all zero, exactly one record per
(predicate, plan) for the 30 predicates and the plans W, F.expand, F.union, each either over the limit or with 2 cold and 3 warm
finite positive times, equal, and its count equal to the predicate file's ClickHouse truth; an over-limit record (over_limit is
True) holds a prefix of that schedule (cold first) of finite positive times, and its completed repetitions agree with the truth.
The predicate file must name the sample and "prefix" and hold exactly 30 predicates of kind like or match with string patterns
and nonnegative integer truth counts; meta.mem_gb must be 52; the table growth bytes must be positive integers. Speedup per predicate = median W /
median F (warm, cold); per run and plan: median, min, max over the predicates where neither side is over the limit, and the
number where W is faster; the over-limit records are counted and listed (an over-limit W with an F result is a predicate where
the wide table needed more than 120 s). Also the database bytes of the wide table against the dictionary and intervals.
E-DUCK2 (refine-logs/E-DUCK-PROTOCOL.md): the same checks per run for wiki_s1-3 (manager wiki) and cap_chat_s1-3 (manager prefix),
inputs results/duck_v2/<sample>__<manager>.json and results/duck/preds_<sample>__<manager>.json; the same per-run statistics,
and ranges over the three samples per workload (wiki, chat).
A problem gives status "blocked" for that experiment, with no numbers for it.
usage: .venv/bin/python figures/online_duck_summary.py [--out results/online_duck_summary.json]
"""
import argparse, json, math, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
PCTS = ["p50", "p90", "p99", "p99.9", "max"]
DUCK_SAMPLES = ["cap_swea_s1", "cap_smith_s1", "cap_oh_s1", "cap_gym_s1"]
DUCK_RUNS = [(x, "prefix", RES / "duck_v1" / f"{x}__prefix.json", RES / "duck" / f"preds_{x}.json") for x in DUCK_SAMPLES]
DUCK2_RUNS = ([(f"wiki_s{k}", "wiki", RES / "duck_v2" / f"wiki_s{k}__wiki.json", RES / "duck" / f"preds_wiki_s{k}__wiki.json") for k in (1, 2, 3)]
              + [(f"cap_chat_s{k}", "prefix", RES / "duck_v2" / f"cap_chat_s{k}__prefix.json", RES / "duck" / f"preds_cap_chat_s{k}__prefix.json")
                 for k in (1, 2, 3)])
PLANS = ["W", "F.expand", "F.union"]
num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
rng = lambda xs: [min(xs), max(xs)]


def tasks():
    out = []
    for line in (ROOT / "cluster" / "tasks" / "online.txt").read_text().splitlines():
        where, data, m, k = line.split()
        out.append((data, m, int(k)))
    return out


def ref_requests(data, m):
    if data.startswith("wiki_"):
        return json.load(open(RES / "wiki" / data / "convert_stats.json"))["spans"]
    return json.load(open(RES / "ing_s1_v1" / f"{data}__{m}.json"))["requests"]


def online(tag="online_v1", impl=None):
    """tag: result directory under results/; impl: if given, every file must record this ingest implementation (E-FAST: v2)."""
    probs, rows = [], {}
    for data, m, k in tasks():
        p = RES / tag / f"{data}__{m}__k{k}.json"
        try:
            d = json.load(open(p))
        except (OSError, ValueError) as e:
            return dict(status="incomplete", problems=[f"{p.relative_to(ROOT)}: {type(e).__name__}"])
        if impl is not None and d.get("ingest_impl") != impl:
            probs.append(f"{p.name}: ingest_impl {d.get('ingest_impl')!r}, expected {impl!r}")
            continue
        lat = d.get("latency_us", {})
        mat = d.get("max_active_traces")
        ok = (d.get("data") == data and d.get("manager") == m and d.get("interleave") == k and d.get("seed") == 1 and d.get("budget") == 64
              and d.get("ckpt") == 32 and d.get("replay_check") == "all requests equal, call order and completeness checked"
              and d.get("requests") == ref_requests(data, m) and lat.get("n") == d.get("requests")
              and all(num(lat.get(q)) and lat[q] >= 0 for q in PCTS + ["mean"]) and all(lat[a] <= lat[b] for a, b in zip(PCTS, PCTS[1:]))
              and isinstance(mat, int) and not isinstance(mat, bool) and 1 <= mat <= min(k, 3000)
              and all(num(d.get(x)) and d[x] > 0 for x in ("mb_per_s", "mean_payload_bytes_per_active_trace", "max_active_payload_bytes", "wall_s",
                                                           "harness_peak_rss_kib", "distinct_messages")))
        if not ok:
            probs.append(f"{p.name}: identity, parameters or values")
            continue
        rows[(data, m, k)] = dict(p50_us=lat["p50"], p99_us=lat["p99"], p999_us=lat["p99.9"], max_us=lat["max"], mean_us=lat["mean"], mb_per_s=d["mb_per_s"],
                                  payload_kib_per_active_trace=d["mean_payload_bytes_per_active_trace"] / 1024,
                                  max_active_payload_mib=d["max_active_payload_bytes"] / 2 ** 20, max_active_traces=d["max_active_traces"],
                                  harness_peak_rss_mib=d["harness_peak_rss_kib"] / 1024, distinct_messages=d["distinct_messages"])
    if probs:
        return dict(status="blocked", problems=probs)
    k64 = [v for (data, m, k), v in rows.items() if k == 64 and data.startswith("cap_")]
    if len(k64) != 17:
        return dict(status="blocked", problems=[f"{len(k64)} seed-1 streams at K = 64, expected 17"])
    sweep = {}
    for (data, m, k), v in sorted(rows.items(), key=lambda x: (x[0][0], x[0][2])):
        if (data, m) in {("cap_gym_s1", "prefix"), ("cap_swea_s1", "prefix"), ("wiki_s1", "wiki")}:
            sweep.setdefault(f"{data}/{m}", {})[str(k)] = v
    pts = [x for v in sweep.values() for x in v.values()]
    sweep_ranges = dict(p50_us=rng([x["p50_us"] for x in pts]), p99_us=rng([x["p99_us"] for x in pts]), mb_per_s=rng([x["mb_per_s"] for x in pts]),
                        per_stream={s: dict(max_active_traces=max(x["max_active_traces"] for x in v.values()),
                                            max_active_payload_mib=max(x["max_active_payload_mib"] for x in v.values()),
                                            payload_kib_per_active_trace=rng([x["payload_kib_per_active_trace"] for x in v.values()]),
                                            p50_us=rng([x["p50_us"] for x in v.values()]), p99_us=rng([x["p99_us"] for x in v.values()])) for s, v in sweep.items()})
    return dict(status="ok", k64_ranges={key: rng([v[key] for v in k64]) for key in k64[0]}, sweep=sweep, sweep_ranges=sweep_ranges,
                per_stream={f"{d}/{m}/k{k}": v for (d, m, k), v in rows.items()})


def duck(spec=DUCK_RUNS, complete_status="ok"):
    probs, runs = [], {}
    for s, manager, p, pf in spec:
        if not p.exists():
            continue
        d, P = json.load(open(p)), json.load(open(pf))
        pp = P.get("predicates", {})
        if not (P.get("sample") == s and P.get("manager") == manager and len(pp) == 30 and all(
                isinstance(v, dict) and v.get("kind") in ("like", "match") and isinstance(v.get("s"), str) and v["s"]
                and isinstance(v.get("truth_spans"), int) and not isinstance(v["truth_spans"], bool) and v["truth_spans"] >= 0 for v in pp.values())):
            probs.append(f"{s}: predicate file identity or values"); continue
        mt = d.get("meta", {})
        if not (mt.get("duckdb") == "1.5.6" and mt.get("sample") == s and mt.get("manager") == manager and mt.get("threads") == 16 and mt.get("cold") == 2
                and mt.get("warm") == 3 and mt.get("timeout_s") == 120 and mt.get("mem_gb") == 52 and mt.get("preds_source") == P["source"] and mt.get("preds_source_md5") == P["source_md5"]
                and "mismatch" not in d and d.get("checks") == dict(iv_rows_minus_opens=0, wide_rows_minus_distinct_spans=0, bad_intervals=0)):
            probs.append(f"{s}: meta, mismatch or store checks"); continue
        by = {}
        for x in d.get("records", []):
            by.setdefault((x.get("pred"), x.get("plan")), []).append(x)
        if set(by) != {(q, pl) for q in P["predicates"] for pl in PLANS} or any(len(v) != 1 for v in by.values()):
            probs.append(f"{s}: record matrix"); continue
        bad = []
        for (q, pl), (x,) in by.items():
            truth, c, w = P["predicates"][q]["truth_spans"], x.get("cold_ms"), x.get("warm_ms")
            if not (isinstance(c, list) and isinstance(w, list) and all(num(t) and t > 0 for t in c + w)):
                bad.append(f"{q} {pl}"); continue
            if "over_limit" in x:                                 # a prefix of the schedule: cold runs first, then warm runs
                ok = (x["over_limit"] is True and (len(c) < 2 and not w or len(c) == 2 and len(w) < 3)
                      and (x.get("count") is None and x.get("equal") is False if not (c or w) else x.get("count") == truth and x.get("equal") is True))
            else:
                ok = len(c) == 2 and len(w) == 3 and x.get("equal") is True and x.get("count") == truth
            isint = lambda v: isinstance(v, int) and not isinstance(v, bool)
            if not (ok and isint(x.get("truth_spans")) and x["truth_spans"] == truth and (x.get("count") is None or isint(x["count"]))):
                bad.append(f"{q} {pl}")
        if bad:
            probs.append(f"{s}: records {bad[:5]}"); continue
        gb = [d.get("load", {}).get(t, {}).get("file_growth_bytes") for t in ("wide", "iv", "msg")]
        if not all(isinstance(g, int) and not isinstance(g, bool) and g > 0 for g in gb):
            probs.append(f"{s}: table growth bytes"); continue
        out = dict(over_limit=sorted(f"{q} {pl}" for (q, pl), (x,) in by.items() if x.get("over_limit")), bytes=dict(wide=gb[0], fcx=gb[1] + gb[2]))
        for pl in PLANS[1:]:
            for mode in ("warm_ms", "cold_ms"):
                xs = [statistics.median(by[(q, "W")][0][mode]) / statistics.median(by[(q, pl)][0][mode]) for q in P["predicates"]
                      if not by[(q, "W")][0].get("over_limit") and not by[(q, pl)][0].get("over_limit")]
                out[f"{pl}/{mode[:4]}"] = dict(n=len(xs), median=statistics.median(xs) if xs else None, min=min(xs, default=None), max=max(xs, default=None),
                                               wide_faster=sum(x < 1 for x in xs))
        out["w_over_limit_with_fcx_result"] = sum(1 for q in P["predicates"] if by[(q, "W")][0].get("over_limit") and not by[(q, "F.expand")][0].get("over_limit"))
        runs[s] = out
    if probs:
        return dict(status="blocked", problems=probs)
    if not runs:
        return dict(status="incomplete", problems=["no E-DUCK run present"])
    keys = [k for k in next(iter(runs.values())) if "/" in k]
    av = lambda k: [r[k] for r in runs.values() if r[k]["n"] > 0]       # runs where the plan and mode have at least one ratio
    return dict(status=complete_status if len(runs) == len(spec) else "partial", samples=sorted(runs), runs=runs,
                ranges={k: dict(runs_with_ratios=len(av(k)), median=rng([r["median"] for r in av(k)]), min=min(r["min"] for r in av(k)),
                                max=max(r["max"] for r in av(k)), wide_faster_max=max(r["wide_faster"] for r in av(k))) if av(k) else None for k in keys},
                bytes_ratio_wide_over_fcx=rng([r["bytes"]["wide"] / r["bytes"]["fcx"] for r in runs.values()]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RES / "online_duck_summary.json"))
    a = ap.parse_args()
    d2 = duck(DUCK2_RUNS)
    if d2.get("status") in ("ok", "partial"):                     # ranges per workload over its samples
        for wl, pre in (("wiki", "wiki_s"), ("chat", "cap_chat_s")):
            sub = {k: v for k, v in d2["runs"].items() if k.startswith(pre)}
            keys = [k for k in next(iter(d2["runs"].values())) if "/" in k]
            av = lambda k: [r[k] for r in sub.values() if r[k]["n"] > 0]
            d2.setdefault("ranges_by_workload", {})[wl] = {k: dict(runs=len(av(k)), median=rng([r["median"] for r in av(k)]), min=min(r["min"] for r in av(k)),
                                                                   max=max(r["max"] for r in av(k)), wide_faster_max=max(r["wide_faster"] for r in av(k)))
                                                            if av(k) else None for k in keys} if sub else None
            d2.setdefault("bytes_ratio_by_workload", {})[wl] = rng([r["bytes"]["wide"] / r["bytes"]["fcx"] for r in sub.values()]) if sub else None
    out = dict(online=online(), duck=duck(), duck2=d2)
    tmp = Path(a.out + ".tmp")
    json.dump(out, open(tmp, "w"), indent=1, allow_nan=False)
    tmp.replace(a.out)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("per_stream", "runs", "sweep")} for k, v in out.items()}, indent=1))


if __name__ == "__main__":
    main()
