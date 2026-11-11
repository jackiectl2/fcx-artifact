"""E-WIKI summary (refine-logs/E-WIKI-PROTOCOL.md, rule 7); frozen and reviewed before its output is used.

Inputs, for each sample s in {1, 2, 3} (copied from the cluster without change):
  results/wiki/wiki_s<s>/wiki_stats.json      converter counts, shape counts, sampled page IDs (fcx/wiki_convert.py)
  results/wiki/wiki_s<s>/convert_stats.json   ingestion statistics (fcx/ingest_run.py, delta parsing, 64 interleaved traces)
  results/ing_wiki_v1/wiki_s<s>__wiki.json    ingestion baselines and ablations (fcx/ingest_ablate.py, same code as E-ING)
  results/wiki/wiki_s<s>__checks.json         bench.py checks run
  results/wiki/wiki_s<s>__index.json          bench.py index run
Rules:
 - presence and completion: every input exists and is readable, and each bench run recorded every planned stage; otherwise
   status "incomplete" and no number;
 - identity and parameters, checked exactly (a wrong but complete file under an expected name is rejected): the converter
   sample (seed, 3,000 pages and traces, min 10 / max 256 revisions, 3,000 distinct page IDs); convert_stats (source sample,
   traces, spans, interleave 64, the one policy "wiki"); ingest_ablate (validate of figures/ing_ext_summary.py with identity
   wiki_s<s> / wiki); both bench runs (meta: data wiki_s<s>, workload wiki, policies wiki, 16 threads, 2 cold and 3 warm runs,
   the planned stage sequence, checkpoint interval 32, time limits 30 s, the paper's server settings and ClickHouse version;
   the index run also its seed s and the six index kinds; the embedded convert statistics equal convert_stats.json and the
   loaded spans and traces equal its counts); cross-artifact counts: ingested requests, array elements, request bytes and the
   ablation's requests equal the converter's snapshots, blocks and bytes, and both dictionaries hold the same distinct blocks;
 - exactness (wiki_summary.table2, the counting rule of the old project's figures/tab2_exact.py, kept read-only in
   refine-logs/ref/tab2_exact.py): no top-level mismatch; ctx_any results other than A.exists, aggregate results other than
   the reference, reconstruction results, logged requests rebuilt byte for byte and lifecycle results are counted and their
   differences summed; any ctx_any result over the time limit, any lifecycle result verified on 2,000 traces only, or a
   selfcheck (read by its four named cases) that missed a damaged input blocks; every exact ingest_ablate variant reproduces
   every request;
 - content queries: fcx/index_report.summarize (held-out predicates only, plan F+I.expand, per-predicate median latency,
   speedup = wide / FCX) per index run, after checking the whole matrix: the six index kinds in records, variants and calib;
   exactly the 30 held-out and 12 calibration substrings; for every index kind and every one of the 42 predicates exactly one
   W and one F+I.expand record, set_equal, no error, 2 cold and 3 warm repetitions; the bootstrap interval is dropped;
   selectivity is the minimum, median and maximum of matched distinct blocks and matching calls per sample and length;
 - shape: counts and shares with explicit denominators and the identity identical + append_only + other_change =
   snapshots - traces; pairwise page overlaps and the union of the three samples (they are not independent);
 - rows: transitions per call against array elements (normalized rows) per call;
 - repetition (added with the lifecycle section): request bytes over the bytes of the distinct blocks in the dictionary
   (convert_stats dictionary_bytes) and array elements over distinct blocks, per sample;
 - lifecycle (rule 5; added 2026-11-08 after the runs, before any paper use, because the paper reports the agent
   equivalents): per query L1-L4 the warm median on intervals (I) and on the normalized table (N), speedup N / I, rule of the
   old figures/fig4_encodings.py panel b; exactly one equal I and N record per query with 3 warm runs, else blocked;
 - ranges: every number the paper may quote across samples is given as a min-max range over the three samples here, so no
   number is formed by hand outside this script.
Any problem, including an unexpected schema error, gives status "blocked" with no numbers.
usage: .venv/bin/python figures/wiki_summary.py [--out results/wiki_summary.json]
"""
import argparse, json, math, os, statistics, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "results"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "fcx"))
import ing_ext_summary as ing                                    # noqa: E402
import index_report                                              # noqa: E402

SEEDS = (1, 2, 3)
INDEXES = ["none", "token", "ngrams3", "ngrams4", "sparsegrams", "bloom4"]
EVAL = {f"eval:sub{L}:{k}" for L in (8, 16, 32) for k in range(10)}
CALIB = {f"calib:sub{L}:{k}" for L in (8, 16, 32) for k in range(4)}
CHECK_STAGES = ["base", "build", "any", "point", "agg", "bytes", "selfcheck", "life"]
INDEX_STAGES = ["base", "build", "bytes", "index"]
POLICY, PAGES, MIN_REVS, MAX_REVS, THREADS, COLD, WARM = "wiki", 3000, 10, 256, 16, 2, 3
CLICKHOUSE, CKPT, TIMEOUT_S = "26.8.15.10", 32, 30
SETTINGS = dict(max_threads=THREADS, use_query_condition_cache=0, use_query_cache=0, wait_end_of_query=1,
                merge_tree_min_rows_for_concurrent_read=0, merge_tree_min_bytes_for_concurrent_read=0)   # the paper's server settings
SELFCHECK = ["aggregate relation, one span removed from the plan: rows that differ",
             "aggregate relation, one value off by one: rows that differ",
             "byte check, one stored message with one byte appended: requests not reproduced",
             "byte check, untouched dictionary: requests not reproduced"]


def paths(s):
    w = RES / "wiki"
    return dict(stats=w / f"wiki_s{s}" / "wiki_stats.json", convert=w / f"wiki_s{s}" / "convert_stats.json",
                ablate=RES / "ing_wiki_v1" / f"wiki_s{s}__wiki.json", checks=w / f"wiki_s{s}__checks.json", index=w / f"wiki_s{s}__index.json")


def mmm(xs):
    xs = list(xs)
    return dict(min=min(xs), median=statistics.median(xs), max=max(xs))


def rng(xs):
    xs = list(xs)
    return [min(xs), max(xs)]


def table2(r):
    """Table 2 counting rule (old figures/tab2_exact.py) on one checks run: (counts, differences, problems)."""
    t = dict(calls=r["spans"], any=0, agg=0, point=0, bytes=0, life=0, any_over_limit=0)
    diff, probs = 0, []
    if r.get("mismatch"):
        probs.append(f"mismatch {r['mismatch']}")
    for x in r["any"]:
        if "error" in x and x.get("set_equal") is not True:
            t["any_over_limit"] += x["plan"] != "A.exists"
            continue
        diff += x.get("set_equal") is not True
        t["any"] += x["plan"] != "A.exists"
    if t["any_over_limit"]:
        probs.append(f"{t['any_over_limit']} ctx_any results over the time limit (unverified)")
    for x in r["agg"]:
        if x["plan"] != "A (reference)":
            if "rows_differ" not in x:
                probs.append(f"aggregate record without rows_differ: {x.get('case')} {x.get('plan')}")
                continue
            t["agg"] += 1
            diff += x["rows_differ"]
    t["point"] = len(r["point"])
    diff += sum(not x["equal"] for x in r["point"])
    if any("equal_on_2000_traces" in x for x in r["life"]):
        probs.append("lifecycle verified on 2,000 traces only")
    t["life"] = len(r["life"])
    diff += sum(not x.get("equal", False) for x in r["life"])
    for b in r["bytes"].values():
        t["bytes"] += b["spans"]
        diff += b["requests_not_reproduced"] + sum(c["rows_differ"] for c in b["checks"].values())
    diff += sum(abs(v) for v in r["checks"].values())
    sc = r.get("selfcheck", {})
    if set(sc) != set(SELFCHECK) or not (sc[SELFCHECK[0]] > 0 and sc[SELFCHECK[1]] > 0 and sc[SELFCHECK[2]] > 0 and sc[SELFCHECK[3]] == 0):
        probs.append(f"selfcheck did not catch the damaged inputs: {sc}")
    return t, diff, probs


LIFE = ["L1", "L2", "L3", "L4"]
LIFE_QUERIES = {"L1 context volume by role (characters x calls)": "L1",                 # the four queries of fcx/bench.py
                "L2 evictions by role (runs that end before their trace does)": "L2",
                "L3 re-entries (messages present in more than one run of a trace)": "L3",
                "L4 residence by role (run length in calls: mean, max)": "L4"}
num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
nat = lambda v: isinstance(v, int) and not isinstance(v, bool)


def lifecycle(r):
    """Rule 5, lifecycle latencies of one checks run, by the rule of the paper's agent panel (old figures/fig4_encodings.py,
    panel b): per query and plan the median of the warm runs; speedup = normalized (N) / intervals (I). Returns (rows, problems);
    every query L1-L4 (exact bench.py name, policy wiki) needs exactly one I and one N record, equal, without error, with the
    planned 2 cold and 3 warm runs (finite positive numbers) and a nonnegative integer read_rows."""
    by, probs = {}, []
    for x in r["life"]:
        q, plan = LIFE_QUERIES.get(x.get("query")), x.get("plan")
        warm, cold, rows = x.get("warm_ms"), x.get("cold_ms"), x.get("read_rows")
        if (x.get("policy") != POLICY or q is None or plan not in ("I", "N") or (q, plan) in by or "error" in x or x.get("equal") is not True
                or not (isinstance(warm, list) and len(warm) == WARM and all(num(v) and v > 0 for v in warm))
                or not (isinstance(cold, list) and len(cold) == COLD and all(num(v) and v > 0 for v in cold))
                or not (nat(rows) and rows >= 0)):
            probs.append(f"lifecycle record unusable: {x.get('query')!r} {plan}")
            continue
        by[(q, plan)] = (statistics.median(warm), rows)
    if set(by) != {(q, p) for q in LIFE for p in ("I", "N")}:
        probs.append(f"lifecycle matrix incomplete: {sorted(by)}")
    if probs:
        return None, probs
    return {q: dict(intervals_ms=by[(q, "I")][0], normalized_ms=by[(q, "N")][0], speedup=by[(q, "N")][0] / by[(q, "I")][0],
                    intervals_read_rows=by[(q, "I")][1], normalized_read_rows=by[(q, "N")][1]) for q in LIFE}, []


def run_identity(r, s, stages, conv, index):
    """Problems with a bench run's recorded identity and parameters."""
    m, probs = r.get("meta", {}), []
    want = dict(data=f"wiki_s{s}", workload="wiki", policies=[POLICY], wide_policies=[POLICY], threads=THREADS, cold=COLD, warm=WARM, stages=stages,
                ckpt=CKPT, timeout_s=TIMEOUT_S, check_timeout_s=TIMEOUT_S, settings=SETTINGS, clickhouse=CLICKHOUSE)
    if index:
        want.update(index_policies=[POLICY], indexes=INDEXES, seed=s)
    probs += [f"meta {k}={m.get(k)!r}, expected {v!r}" for k, v in want.items() if m.get(k) != v]
    if r.get("convert") != conv:
        probs.append("embedded convert statistics differ from convert_stats.json")
    if r.get("spans") != conv.get("spans") or r.get("traces") != conv.get("traces"):   # the store the server actually loaded
        probs.append(f"loaded spans/traces {r.get('spans')}/{r.get('traces')} differ from convert_stats {conv.get('spans')}/{conv.get('traces')}")
    return probs


def index_matrix(r):
    """Problems with the completeness of an index run: kinds, predicates, records, repetitions."""
    probs = []
    if set(r.get("predicates", {})) != EVAL | CALIB:
        probs.append(f"predicates {len(r.get('predicates', {}))}, expected the 30 + 12 substrings")
    for part in ("variants", "calib"):
        if list(r.get(part, {})) != INDEXES:
            probs.append(f"{part} kinds {list(r.get(part, {}))}")
    seen = {}
    for x in r.get("records", []):
        if x.get("plan") in ("W", index_report.FACT):
            seen.setdefault((x.get("index"), x.get("pred"), x["plan"]), []).append(x)
    for kind in INDEXES:
        for p in sorted(EVAL | CALIB):
            for plan in ("W", index_report.FACT):
                xs = seen.get((kind, p, plan), [])
                if len(xs) != 1:
                    probs.append(f"{kind} {p} {plan}: {len(xs)} records")
                    continue
                x = xs[0]
                if x.get("set_equal") is not True or "error" in x or len(x.get("cold_ms", [])) != COLD or len(x.get("warm_ms", [])) != WARM:
                    probs.append(f"{kind} {p} {plan}: not complete, equal and fully timed")
    if len(seen) != len(INDEXES) * len(EVAL | CALIB) * 2:
        probs.append(f"{len(seen)} (index, predicate, plan) groups for W and {index_report.FACT}, expected {len(INDEXES) * 84}")
    return probs[:20]


def write(out, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1, allow_nan=False)
    os.replace(tmp, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RES / "wiki_summary.json"))
    dst = Path(ap.parse_args().out)
    base = dict(protocol="refine-logs/E-WIKI-PROTOCOL.md")
    try:
        summarize(dst, base)
    except SystemExit:
        raise
    except Exception as e:                                        # an unexpected schema error never leaves an older result in place
        write(dict(base, status="blocked", problems=[f"unexpected {type(e).__name__}: {e}"]), dst)
        raise SystemExit(f"blocked: unexpected {type(e).__name__}: {e}")


def summarize(dst, base):
    D, missing = {}, []
    for s in SEEDS:
        for k, p in paths(s).items():
            try:
                D[(s, k)] = json.load(open(p))
            except (OSError, ValueError) as e:
                missing.append(f"{p.relative_to(ROOT)}: {type(e).__name__}")
    unfinished = []
    for s in SEEDS:
        for k, stages in (("checks", CHECK_STAGES), ("index", INDEX_STAGES)):
            r = D.get((s, k))
            if r is not None and not (isinstance(r, dict) and all(f"{st}_s" in r.get("meta", {}) for st in stages)):
                unfinished.append(f"wiki_s{s} {k}: not every planned stage recorded")
    if missing or unfinished:
        write(dict(base, status="incomplete", missing=missing, unfinished=unfinished), dst)
        raise SystemExit(f"incomplete: {missing + unfinished}; see {dst}")

    blocked, out = [], dict(base, samples={})
    for s in SEEDS:
        st, cv, ab, ck, ix = (D[(s, k)] for k in ("stats", "convert", "ablate", "checks", "index"))
        name = f"wiki_s{s}"
        o = out["samples"][name] = {}
        # identities and parameters
        if not (st.get("seed") == s and st.get("pages") == PAGES and st.get("traces") == PAGES and st.get("min_revs") == MIN_REVS
                and st.get("max_revs") == MAX_REVS and len(st.get("page_ids", [])) == PAGES and len(set(st.get("page_ids", []))) == PAGES):
            blocked.append(f"{name}: converter sample identity or parameters")
        if not (Path(str(cv.get("src", ""))).name == name and cv.get("traces") == PAGES and cv.get("spans") == st.get("spans")
                and cv.get("interleave") == 64 and list(cv.get("policies", {})) == [POLICY]):
            blocked.append(f"{name}: convert_stats identity")
        pw = cv.get("policies", {}).get(POLICY, {})
        if not (pw.get("ingest", {}).get("requests") == st.get("spans") and pw.get("array_elems") == st.get("blocks")
                and pw.get("wide_chars") == st.get("logical_bytes") and ab.get("requests") == st.get("spans")
                and ab.get("distinct_messages") == cv.get("distinct_messages")):
            blocked.append(f"{name}: ingested store, ablation and converter output disagree on requests, blocks, bytes or distinct blocks")
        blocked += [f"{name} checks: {x}" for x in run_identity(ck, s, CHECK_STAGES, cv, index=False)]
        blocked += [f"{name} index: {x}" for x in run_identity(ix, s, INDEX_STAGES, cv, index=True)]
        # conversion and shape
        sh = st["shape"]
        trans = sh["snapshots"] - st["traces"]
        if sh["identical"] + sh["append_only"] + sh["other_change"] != trans or sh["snapshots"] != st["spans"]:
            blocked.append(f"{name}: shape counts inconsistent")
        db, dm = cv.get("dictionary_bytes"), cv.get("distinct_messages")
        if not (nat(db) and db > 0 and nat(dm) and dm > 0 and pw.get("new_message_bytes") == db and pw.get("new_messages") == dm
                and ab.get("dictionary_bytes") == db and ab.get("distinct_messages") == dm):
            blocked.append(f"{name}: dictionary size missing or not the same in convert_stats, its policy and the ablation")
            continue
        o["conversion"] = dict(pass1=st["pass1"], traces=st["traces"], snapshots=st["spans"], blocks=st["blocks"], logical_bytes=st["logical_bytes"],
                               distinct_blocks=cv["distinct_messages"], dictionary_bytes=cv["dictionary_bytes"],
                               logged_over_distinct_bytes=st["logical_bytes"] / cv["dictionary_bytes"],
                               blocks_over_distinct=st["blocks"] / cv["distinct_messages"],
                               capped_in_sample=st["capped_in_sample"],
                               shape=dict(counts=sh, transitions=trans,
                                          shares_of_snapshots={k: sh[k] / sh["snapshots"] for k in ("empty", "with_duplicate_block", "reentry")},
                                          shares_of_transitions={k: sh[k] / trans for k in ("identical", "append_only", "other_change")}))
        # exactness of the checks run
        t, diff, probs = table2(ck)
        if diff or probs:
            blocked.append(f"{name} checks: {diff} differences, {probs}")
        o["exactness_checks"] = dict(t, differences=diff)
        lc, lprobs = lifecycle(ck)
        if lprobs:
            blocked.append(f"{name} checks: {lprobs}")
        o["lifecycle"] = lc
        # ingestion baselines (E-ING definitions)
        err = ing.validate(ab, name, POLICY)
        if err:
            blocked.append(f"{name} ingest_ablate data errors {err}")
        else:
            bad = {k: ab["variants"][k]["wrong"] for k in ing.EXACT if ab["variants"][k]["wrong"]}
            if bad:
                blocked.append(f"{name} ingest_ablate exact variants wrong {bad}")
            o["ingestion"] = dict(requests=ab["requests"], exact_variants_zero_wrong=not bad, ratios=ing.ratios(ab),
                                  set_wrong_equals_dup=ab["variants"]["set_diff"]["wrong"] == ab["slice_requests"]["dup"],
                                  timing_kind="single run, fixed variant order, derivation only")
        # rows per call
        p = cv["policies"][POLICY]
        o["rows_per_call"] = dict(transitions=p["transitions_per_span"], normalized=p["mean_context_len"],
                                  normalized_over_transitions=p["mean_context_len"] / p["transitions_per_span"] if p["transitions_per_span"] else None)
        # content queries
        r = ix.get("index", {}).get(POLICY, {})
        mprobs = index_matrix(r)
        if ix.get("mismatch"):
            mprobs.append(f"mismatch {ix['mismatch']}")
        if mprobs:
            blocked.append(f"{name} index matrix: {mprobs}")
            continue
        _, rows = index_report.summarize(str(paths(s)["index"]))
        if sorted(x["index"] for x in rows) != sorted(INDEXES) or not all(x["complete"] for x in rows):
            blocked.append(f"{name}: index_report rows incomplete")
            continue
        o["content_queries"] = {x["index"]: dict({k: {kk: vv for kk, vv in x[k].items() if kk != "ci"} for k in ("warm_ms", "cold_ms")},
                                                 wide_ms=x["wide_ms"], fcx_ms=x["fact_ms"], normalized=x.get("F+N.inv"),
                                                 reconstruct_first=x.get("R.reconstruct"), build=x["variant"]) for x in rows}
        o["selectivity"] = {L: dict(matched_messages=mmm(r["matched_messages"][k] for k in EVAL if k.startswith(f"eval:sub{L}:")),
                                    matching_calls=mmm(r["truth_spans"][k] for k in EVAL if k.startswith(f"eval:sub{L}:"))) for L in (8, 16, 32)}
        o["content_plan_results_verified"] = sum(1 for x in r["records"] if x.get("set_equal") is True)
    if blocked:                                                   # no number is reported for a blocked summary
        write(dict(base, status="blocked", problems=blocked), dst)
        raise SystemExit(f"blocked: {blocked}")
    S = [out["samples"][f"wiki_s{s}"] for s in SEEDS]
    ids = {s: set(D[(s, "stats")]["page_ids"]) for s in SEEDS}
    out["sample_overlap"] = dict(pairwise={f"{a}-{b}": len(ids[a] & ids[b]) for a in SEEDS for b in SEEDS if a < b},
                                 union=len(set().union(*ids.values())), per_sample={str(s): len(v) for s, v in ids.items()})
    out["ranges"] = dict(
        requests=rng(x["ingestion"]["requests"] for x in S),
        exactness_totals={k: sum(x["exactness_checks"][k] for x in S) for k in ("any", "agg", "point", "bytes", "life")},
        shape_shares_of_snapshots={k: rng(x["conversion"]["shape"]["shares_of_snapshots"][k] for x in S) for k in ("empty", "with_duplicate_block", "reentry")},
        shape_shares_of_transitions={k: rng(x["conversion"]["shape"]["shares_of_transitions"][k] for x in S) for k in ("identical", "append_only", "other_change")},
        ingestion_ratios={k: rng(x["ingestion"]["ratios"][k] for x in S) for k in S[0]["ingestion"]["ratios"]},
        rows_normalized_over_transitions=rng(x["rows_per_call"]["normalized_over_transitions"] for x in S),
        logged_over_distinct_bytes=rng(x["conversion"]["logged_over_distinct_bytes"] for x in S),
        blocks_over_distinct=rng(x["conversion"]["blocks_over_distinct"] for x in S),
        lifecycle_speedup_by_query={q: rng(x["lifecycle"][q]["speedup"] for x in S) for q in LIFE},
        lifecycle_speedup_all=rng(x["lifecycle"][q]["speedup"] for x in S for q in LIFE),
        lifecycle_read_rows=dict(intervals=rng(x["lifecycle"][q]["intervals_read_rows"] for x in S for q in LIFE),
                                 normalized=rng(x["lifecycle"][q]["normalized_read_rows"] for x in S for q in LIFE)),
        content_warm_median_by_index={k: rng(x["content_queries"][k]["warm_ms"]["median"] for x in S) for k in INDEXES},
        content_cold_median_by_index={k: rng(x["content_queries"][k]["cold_ms"]["median"] for x in S) for k in INDEXES},
        content_warm_median_all_indexes=rng(x["content_queries"][k]["warm_ms"]["median"] for x in S for k in INDEXES),
        content_cold_median_all_indexes=rng(x["content_queries"][k]["cold_ms"]["median"] for x in S for k in INDEXES),
        content_wide_faster_warm_by_index={k: rng(x["content_queries"][k]["warm_ms"]["wide_faster"] for x in S) for k in INDEXES},
        content_wide_faster_cold_by_index={k: rng(x["content_queries"][k]["cold_ms"]["wide_faster"] for x in S) for k in INDEXES},
        content_warm_min_max_by_index={k: [min(x["content_queries"][k]["warm_ms"]["min"] for x in S), max(x["content_queries"][k]["warm_ms"]["max"] for x in S)]
                                       for k in INDEXES})
    out["status"] = "ok"
    write(out, dst)
    print(json.dumps({k: v for k, v in out.items() if k != "samples"}, indent=1))


if __name__ == "__main__":
    main()
