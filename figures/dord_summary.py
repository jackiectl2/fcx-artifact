"""E-DORD summary: the index stage with the dictionary sorted by message ID (layout id, the paper's first runs) and by the first
call in which each message is open (layout first), run as pairs in one allocation (cluster/bench_pair.sbatch, the order of the
two layouts alternating across samples). Inputs: results/dord/<data>__<out>_<layout>.json, copied from the cluster unchanged.
Rules (any problem: status "blocked", no numbers):
 - identity of each run: meta.dict_order equals its layout; data, workload, policies, 16 threads, 2 cold and 3 warm runs,
   stages base, build, bytes, index; the six index kinds; ClickHouse 26.8.15.10; the paper's server settings;
 - pairing: both runs of a pair carry the same non-empty pair ID (array job and task) and pair order, ran on the same host
   (both host fields equal; the package copies drop host names, then the pair ID binds the pair),
   loaded the same store (embedded convert statistics, spans, traces), and took their predicates (--preds-from) from the
   planned legacy run of GROUPS (file name and md5 of results/<source>, and the predicates equal to that run's); identical
   ground-truth span counts per predicate and matched-message counts;
 - matrix: exactly the 12 calibration + 30 held-out predicates (wiki, chat; SWE agent samples: 12 substrings + 3 regular
   expressions for calibration, 18 substrings + 6 regular expressions + 6 needles held out, as in the paper's index runs), calib and variants over the six kinds, and for every kind and
   all 42 predicates exactly one complete, equal W and F+I.expand record with 2 cold and 3 warm runs; the first dictionary
   load at granularity 8192 with the columns of its layout (id: msg_id, text; first: also first_span);
 - dictionary checks (bench.dict_checks): every recorded check of both runs ok;
 - matrix: fcx/index_report.summarize, held-out predicates only, plan F+I.expand against W, per-predicate median latency,
   speedup = W / F; every index kind complete (30 predicates, W and F complete, equal and within the time limit);
 - per pair and index kind: both layouts' median speedups (warm and cold) and wide-faster counts; the paired effect of the
   layout on FCX, median over predicates of F_id / F_first (warm, cold), and the same ratio of W as a control for drift
   between the two runs; the parameters each side chose (dictionary granularity and index parameter, wide parameter), and
   whether they agree across the pair (independently tuned layouts: a change of chosen parameter is reported, not hidden);
 - storage: compressed bytes of the dictionary table at its first load (granularity 8192, before calibration; text and
   message-ID columns, plus first_span for layout first) and of the wide table, per layout;
 - ranges per group (min-max over its pairs) of every number the paper may quote.
usage: .venv/bin/python figures/dord_summary.py [--out results/dord_summary.json] [--groups wiki,chat,...]
"""
import argparse, json, statistics, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "results"
sys.path.insert(0, str(ROOT / "fcx"))
import index_report                                              # noqa: E402

INDEXES = ["none", "token", "ngrams3", "ngrams4", "sparsegrams", "bloom4"]
LAYOUTS = ("id", "first")
STAGES = ["base", "build", "bytes", "index"]
CLICKHOUSE, THREADS, COLD, WARM = "26.8.15.10", 16, 2, 3
SETTINGS = dict(max_threads=THREADS, use_query_condition_cache=0, use_query_cache=0, wait_end_of_query=1,
                merge_tree_min_rows_for_concurrent_read=0, merge_tree_min_bytes_for_concurrent_read=0)
GROUPS = {  # group -> [(data, out, workload, policy, legacy run whose predicates both layouts must use: results/<path>)]
    "wiki": [(f"wiki_s{s}", "index_dord", "wiki", "wiki", f"wiki/wiki_s{s}__index.json") for s in (1, 2, 3)],
    "chat_prefix": [(f"cap_chat_s{s}", "index_prefix_dord", "chat", "prefix", f"chat/cap_chat_s{s}__index_prefix.json") for s in (1, 2, 3)],
    "chat_managers": [(f"cap_chat_s{s}", f"index_{m}_dord", "chat", m, f"chat/cap_chat_s{s}__index_{m}.json") for s in (1, 2, 3)
                      for m in ("autogen_buf32", "autogen_ht", "lc_trim32k")],
    "wiki_large": [("wikix_s1", "index_dord", "wiki", "wiki", "wiki/wiki_s1__index.json")],
    "chat_large": [("cap_chatall_s1", "index_dord", "chat", "prefix", "chat/cap_chat_s1__index_prefix.json")],
    # the four seed-1 agent samples, append only; predicates from the paper's index runs (copied unchanged from the old project)
    "agents": [(f"cap_{c}_s1", "index_prefix_dord", "swe", "prefix", f"legacy_agent/cap_{c}_s1__index_prefix.json") for c in ("oh", "swea", "smith", "gym")],
}
EVAL = {f"eval:sub{L}:{k}" for L in (8, 16, 32) for k in range(10)}
CALIB = {f"calib:sub{L}:{k}" for L in (8, 16, 32) for k in range(4)}
SWE_EVAL = ({f"eval:sub{L}:{k}" for L in (8, 16, 32) for k in range(6)} | {f"eval:re:{k}" for k in range(6)}
            | {f"eval:t:{n}" for n in ("traceback", "nosuchfile", "assertionerror", "importnumpy", "deftest", "gitdiff")})
SWE_CALIB = {f"calib:sub{L}:{k}" for L in (8, 16, 32) for k in range(4)} | {f"calib:re:{k}" for k in range(3)}
PREDS = {"wiki": (EVAL, CALIB), "chat": (EVAL, CALIB), "swe": (SWE_EVAL, SWE_CALIB)}   # workload -> (held out, calibration)
COLUMNS = {"id": {"msg_id", "text"}, "first": {"msg_id", "first_span", "text"}}


def rng(xs):
    xs = list(xs)
    return dict(min=min(xs), max=max(xs)) if xs else None


def med_ratio(by_a, by_b, plan_a, plan_b, key):
    """Median over held-out predicates of median(a) / median(b)."""
    x = []
    for pred, pa in by_a.items():
        a, b = index_report.med(pa.get(plan_a), key), index_report.med(by_b.get(pred, {}).get(plan_b), key)
        if a is not None and b is not None:
            x.append(a / b)
    return statistics.median(x) if x else None


def held_out(r):
    by = {}
    for rec in r["records"]:
        if rec["pred"].startswith("eval"):
            by.setdefault(rec["index"], {}).setdefault(rec["pred"], {})[rec["plan"]] = rec
    return by


def matrix(r, workload):
    """Problems with the completeness of an index run over all predicates (calibration ones choose the parameters)."""
    probs = []
    ev, ca = PREDS[workload]
    if set(r.get("predicates", {})) != ev | ca:
        probs.append(f"predicates {len(r.get('predicates', {}))}, expected the {len(ev)} held-out + {len(ca)} calibration ones")
    for part in ("variants", "calib"):
        if list(r.get(part, {})) != INDEXES:
            probs.append(f"{part} kinds {list(r.get(part, {}))}")
    seen = {}
    for x in r.get("records", []):
        if x.get("plan") in ("W", index_report.FACT):
            seen.setdefault((x.get("index"), x.get("pred"), x["plan"]), []).append(x)
    for kind in INDEXES:
        for p in sorted(ev | ca):
            for plan in ("W", index_report.FACT):
                xs = seen.get((kind, p, plan), [])
                if len(xs) != 1 or xs[0].get("set_equal") is not True or "error" in xs[0] or len(xs[0].get("cold_ms", [])) != COLD or len(xs[0].get("warm_ms", [])) != WARM:
                    probs.append(f"{kind} {p} {plan}: not exactly one complete, equal, fully timed record")
    return probs[:10]


def md5(path):
    import hashlib
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def identity(run, layout, workload, policy, data=None):
    m, probs = run.get("meta", {}), []
    want = dict(dict_order=layout, workload=workload, policies=[policy], wide_policies=[policy], index_policies=[policy], indexes=INDEXES,
                threads=THREADS, cold=COLD, warm=WARM, stages=STAGES, clickhouse=CLICKHOUSE, settings=SETTINGS, data=data)
    probs += [f"meta {k}={m.get(k)!r}, expected {v!r}" for k, v in want.items() if m.get(k) != v]
    if not m.get("preds_from_md5"):
        probs.append("no --preds-from recorded")
    if not all(f"{st}_s" in m for st in STAGES):
        probs.append("not every stage recorded")
    return probs


def pair(data, out, workload, policy, source):
    runs, probs = {}, []
    for L in LAYOUTS:
        p = RES / "dord" / f"{data}__{out}_{L}.json"
        try:
            runs[L] = json.load(open(p))
        except (OSError, ValueError) as e:
            return None, [f"{p.relative_to(ROOT)}: {type(e).__name__}"], True
        probs += [f"{data} {L}: {x}" for x in identity(runs[L], L, workload, policy, data)]
    if probs:
        return None, probs, False
    ri = {L: runs[L]["index"][policy] for L in LAYOUTS}
    legacy = json.load(open(RES / source))
    hosts = [runs[L]["meta"].get("host") for L in LAYOUTS]            # the package drops host names: then the pair ID alone binds the pair
    if hosts[0] != hosts[1]:
        probs.append(f"{data}: host differs within the pair")
    for k, f in (("pair_id", lambda r: r["meta"].get("pair_id")), ("pair_order", lambda r: r["meta"].get("pair_order")),
                 ("preds_from_md5", lambda r: r["meta"]["preds_from_md5"]), ("convert", lambda r: r.get("convert")),
                 ("spans", lambda r: r.get("spans")), ("traces", lambda r: r.get("traces"))):
        if f(runs["id"]) != f(runs["first"]) or f(runs["id"]) in (None, ""):
            probs.append(f"{data}: {k} differs within the pair or is missing")
    if runs["id"]["meta"].get("pair_order") not in ("id,first", "first,id"):
        probs.append(f"{data}: pair_order {runs['id']['meta'].get('pair_order')!r}")
    if runs["id"]["meta"]["preds_from_md5"] != md5(RES / source) or runs["id"]["meta"].get("preds_from") != Path(source).name.split("__", 1)[1]:
        probs.append(f"{data}: predicates not taken from the planned source {source}")
    if ri["id"].get("predicates") != next(iter(legacy["index"].values())).get("predicates"):
        probs.append(f"{data}: predicates differ from the planned source {source}")
    for k in ("predicates", "truth_spans", "matched_messages"):
        if ri["id"].get(k) != ri["first"].get(k) or not ri["id"].get(k):
            probs.append(f"{data}: {k} differ within the pair or are missing")
    for L in LAYOUTS:
        probs += [f"{data} {L}: {x}" for x in matrix(ri[L], workload)]
        first_load = ri[L]["load"].get(f"mx_{policy}", {})
        if first_load.get("granularity") != 8192 or set(first_load.get("column_bytes", {})) != COLUMNS[L]:
            probs.append(f"{data} {L}: first dictionary load not at granularity 8192 with the columns {sorted(COLUMNS[L])}")
        dc = ri[L].get("dict_checks") or {}
        if set(dc) != {"initial", *INDEXES} or not all(v.get("ok") is True for v in dc.values()):
            probs.append(f"{data} {L}: dictionary checks missing or failed")
        if L == "first" and not all("wrong_first_span" in v for v in dc.values()):
            probs.append(f"{data} first: first_span not checked")
    rows = {}
    for L in LAYOUTS:
        tmp = RES / "dord" / f"{data}__{out}_{L}.json"
        _, rr = index_report.summarize(str(tmp))
        rows[L] = {x["index"]: x for x in rr}
        if set(rows[L]) != set(INDEXES) or not all(x["complete"] for x in rows[L].values()):
            probs.append(f"{data} {L}: index matrix incomplete")
    if probs:
        return None, probs, False
    by = {L: held_out(ri[L]) for L in LAYOUTS}
    res = dict(host=runs["id"]["meta"].get("host"), pair_id=runs["id"]["meta"]["pair_id"], pair_order=runs["id"]["meta"]["pair_order"], source=source, storage={}, kinds={})
    for L in LAYOUTS:
        ld = ri[L]["load"]
        res["storage"][L] = dict(dictionary_bytes=ld[f"mx_{policy}"]["compressed_bytes"], dictionary_granularity=ld[f"mx_{policy}"]["granularity"],
                                 dictionary_columns=ld[f"mx_{policy}"]["column_bytes"], wide_bytes=ld[f"wx_{policy}"]["compressed_bytes"])
    res["storage"]["dictionary_id_over_first"] = res["storage"]["id"]["dictionary_bytes"] / res["storage"]["first"]["dictionary_bytes"]
    for k in INDEXES:
        v = {L: ri[L]["variants"][k] for L in LAYOUTS}
        res["kinds"][k] = dict(
            speedup={L: dict(warm=rows[L][k]["warm_ms"]["median"], cold=rows[L][k]["cold_ms"]["median"], wide_faster_warm=rows[L][k]["warm_ms"]["wide_faster"],
                             wide_faster_cold=rows[L][k]["cold_ms"]["wide_faster"]) for L in LAYOUTS},
            fcx_id_over_first=dict(warm=med_ratio(by["id"][k], by["first"][k], index_report.FACT, index_report.FACT, "warm_ms"),
                                   cold=med_ratio(by["id"][k], by["first"][k], index_report.FACT, index_report.FACT, "cold_ms")),
            wide_id_over_first=dict(warm=med_ratio(by["id"][k], by["first"][k], "W", "W", "warm_ms"), cold=med_ratio(by["id"][k], by["first"][k], "W", "W", "cold_ms")),
            chosen={L: dict(dictionary_granularity=v[L]["factored"].get("granularity"), dictionary_param=v[L]["factored"].get("param"), wide_param=v[L]["wide"].get("param"))
                    for L in LAYOUTS})
        res["kinds"][k]["same_parameters"] = res["kinds"][k]["chosen"]["id"] == res["kinds"][k]["chosen"]["first"]
    return res, [], False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RES / "dord_summary.json")); ap.add_argument("--groups", default=",".join(GROUPS))
    a = ap.parse_args()
    out, blocked, missing = dict(protocol="refine-logs/E-DORD-PROTOCOL.md", groups={}), [], []
    for g in a.groups.split(","):
        pairs = {}
        for data, o, wl, pol, source in GROUPS[g]:
            res, probs, absent = pair(data, o, wl, pol, source)
            (missing if absent else blocked).extend(probs)
            if res:
                pairs[f"{data}/{pol}"] = res
        if len(pairs) == len(GROUPS[g]):
            P = list(pairs.values())
            out["groups"][g] = dict(pairs=pairs, ranges=dict(
                dictionary_id_over_first=rng(x["storage"]["dictionary_id_over_first"] for x in P),
                **{f"speedup_{L}_{t}_all_indexes": rng(x["kinds"][k]["speedup"][L][t] for x in P for k in INDEXES) for L in LAYOUTS for t in ("warm", "cold")},
                **{f"wide_faster_{L}_{t}_max": max(x["kinds"][k]["speedup"][L][f"wide_faster_{t}"] for x in P for k in INDEXES) for L in LAYOUTS for t in ("warm", "cold")},
                **{f"fcx_id_over_first_{t}": rng(x["kinds"][k]["fcx_id_over_first"][t] for x in P for k in INDEXES) for t in ("warm", "cold")},
                **{f"wide_id_over_first_{t}": rng(x["kinds"][k]["wide_id_over_first"][t] for x in P for k in INDEXES) for t in ("warm", "cold")},
                by_index={k: {f"{L}_{t}": rng(x["kinds"][k]["speedup"][L][t] for x in P) for L in LAYOUTS for t in ("warm", "cold")} for k in INDEXES},
                parameter_changes=sum(not x["kinds"][k]["same_parameters"] for x in P for k in INDEXES),
                pair_orders={o: sum(x["pair_order"] == o for x in P) for o in ("id,first", "first,id")}))
    out["status"] = "blocked" if blocked else ("incomplete" if missing else "ok")
    out["problems"], out["missing"] = blocked[:40], missing
    dst = Path(a.out); dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp"); json.dump(out, open(tmp, "w"), indent=1); tmp.replace(dst)
    print(out["status"], json.dumps({g: v["ranges"] for g, v in out["groups"].items()}, indent=1)[:4000])
    if blocked:
        raise SystemExit(f"blocked: {blocked[:5]}")


if __name__ == "__main__":
    main()
