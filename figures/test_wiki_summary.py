"""Test of wiki_summary.py on self-contained synthetic inputs (no result file is read): a complete tree must pass, and each
damaged tree must give the status the protocol requires.
usage: python figures/test_wiki_summary.py      (project .venv or the project SIF)
"""
import json, sys, tempfile
from pathlib import Path

import wiki_summary as ws


def ablate(s):
    n, v = 1000, {}
    for name in ws.ing.EXACT + ws.ing.LOSSY:
        x = dict(requests=n, records=2000, payload_bytes=50000, wrong=0 if name in ws.ing.EXACT else 100, cpu_s=1.5, must_be_exact=name in ws.ing.EXACT)
        if name in ws.ing.LOG_VARIANTS:
            x.update(checkpoint_bytes=8000, fallback_requests=0)
        v[name] = x
    return dict(data=f"/x/data/wiki_s{s}", manager="wiki", interleave=64, seed=1, ckpt=32, requests=30000, distinct_messages=40000, dictionary_bytes=2 * 10 ** 6,
                slice_requests=dict(dup=100), variants={k: dict(x, requests=30000) for k, x in v.items()})


def convert(s):
    return dict(src=f"/x/data/wiki_s{s}", traces=3000, spans=30000, interleave=64, distinct_messages=40000, dictionary_bytes=2 * 10 ** 6,
                policies={"wiki": dict(transitions_per_span=2.0, mean_context_len=20.0, array_elems=90000, wide_chars=10 ** 7, ingest=dict(requests=30000),
                                       new_messages=40000, new_message_bytes=2 * 10 ** 6)})


def meta(s, stages, index):
    m = {f"{st}_s": 1.0 for st in stages}
    m.update(data=f"wiki_s{s}", workload="wiki", policies=["wiki"], wide_policies=["wiki"], threads=16, cold=2, warm=3, stages=list(stages),
             index_policies=["wiki"], indexes=list(ws.INDEXES), seed=s if index else 0, ckpt=32, timeout_s=30, check_timeout_s=30,
             clickhouse="26.8.15.10", settings=dict(ws.SETTINGS))
    return m


def index_run(s):
    recs, preds = [], sorted(ws.EVAL | ws.CALIB)
    for kind in ws.INDEXES:
        for p in preds:
            for plan, ms in (("W", 50.0), ("F+I.expand", 5.0), ("F+N.inv", 6.0)):
                recs.append(dict(policy="wiki", index=kind, pred=p, plan=plan, matched_messages=3, set_equal=True, cardinality=7,
                                 value="7", warm_ms=[ms] * 3, cold_ms=[2 * ms] * 2))
    var = {k: dict(wide=dict(build_s=1.0, index_bytes=10, param=0), factored=dict(build_s=0.1, index_bytes=1, granularity=64, param=0)) for k in ws.INDEXES}
    return dict(meta=meta(s, ws.INDEX_STAGES, True), convert=convert(s), spans=30000, traces=3000,
                index={"wiki": dict(records=recs, predicates={p: f"text {p}" for p in preds}, variants=var, calib={k: {} for k in ws.INDEXES},
                                    matched_messages={p: 3 for p in preds}, truth_spans={p: 7 for p in preds})})


def checks_run(s):
    return dict(meta=meta(s, ws.CHECK_STAGES, False), convert=convert(s), spans=30000, traces=3000,
                any=[dict(policy="wiki", plan=p, set_equal=True) for p in ("A.exists", "I.expand", "B.bitmap")],
                agg=[dict(case="c", plan="A (reference)"), dict(case="c", plan="I", rows_differ=0)],
                point=[dict(equal=True)] * 3,
                life=[dict(policy="wiki", query=q, plan=pl, equal=True, warm_ms=[ms] * 3, cold_ms=[2 * ms] * 2, read_rows=rows)
                      for q in ws.LIFE_QUERIES for pl, ms, rows in (("I", 4.0, 100), ("N", 12.0, 900))],
                bytes={"wiki": dict(spans=30000, requests_not_reproduced=0, checks={"x": dict(rows_differ=0)})},
                checks={"a_wiki_spans_differ_from_requests": 0}, selfcheck={k: v for k, v in zip(ws.SELFCHECK, (3, 1, 5, 0))})


def build(tmp):
    res = Path(tmp) / "results"
    for s in ws.SEEDS:
        (res / "wiki" / f"wiki_s{s}").mkdir(parents=True, exist_ok=True)
        (res / "ing_wiki_v1").mkdir(parents=True, exist_ok=True)
        st = dict(seed=s, pages=3000, min_revs=10, max_revs=256, pass1={}, traces=3000, spans=30000, blocks=90000, logical_bytes=10 ** 7,
                  capped_in_sample=5, page_ids=list(range(1000 * s, 1000 * s + 3000)),
                  shape=dict(snapshots=30000, empty=10, with_duplicate_block=20, identical=1000, append_only=16000, other_change=10000, reentry=300))
        json.dump(st, open(res / "wiki" / f"wiki_s{s}" / "wiki_stats.json", "w"))
        json.dump(convert(s), open(res / "wiki" / f"wiki_s{s}" / "convert_stats.json", "w"))
        json.dump(ablate(s), open(res / "ing_wiki_v1" / f"wiki_s{s}__wiki.json", "w"))
        json.dump(checks_run(s), open(res / "wiki" / f"wiki_s{s}__checks.json", "w"))
        json.dump(index_run(s), open(res / "wiki" / f"wiki_s{s}__index.json", "w"))
    ws.ROOT, ws.RES = Path(tmp), res
    return res


def run(tmp):
    sys.argv = ["wiki_summary.py", "--out", f"{tmp}/out.json"]
    try:
        ws.main(); code = 0
    except SystemExit as e:
        code = 1 if e.code else 0
    return code, json.load(open(f"{tmp}/out.json"))


def edit(p, fn):
    d = json.load(open(p)); fn(d); json.dump(d, open(p, "w"))


def expect(rel, damage, status="blocked"):
    with tempfile.TemporaryDirectory() as t:
        res = build(t)
        edit(res / rel, damage)
        code, out = run(t)
        assert code == 1 and out["status"] == status and "samples" not in out and "ranges" not in out, (rel, out)


def recs_of(d):
    return d["index"]["wiki"]["records"]


def main():
    with tempfile.TemporaryDirectory() as t:
        build(t)
        code, out = run(t)
        assert code == 0 and out["status"] == "ok", out.get("problems")
        o = out["samples"]["wiki_s1"]
        assert set(o["content_queries"]) == set(ws.INDEXES) and o["content_queries"]["none"]["warm_ms"]["median"] == 10.0
        assert "ci" not in o["content_queries"]["none"]["warm_ms"] and o["exactness_checks"]["any"] == 2 and o["exactness_checks"]["agg"] == 1
        assert out["sample_overlap"] == dict(pairwise={"1-2": 2000, "1-3": 1000, "2-3": 2000}, union=5000, per_sample={"1": 3000, "2": 3000, "3": 3000})
        assert o["rows_per_call"]["normalized_over_transitions"] == 10.0 and o["selectivity"]["8"]["matched_messages"]["median"] == 3
        assert out["ranges"]["content_warm_median_all_indexes"] == [10.0, 10.0] and out["ranges"]["exactness_totals"]["point"] == 9
        assert out["ranges"]["content_cold_median_all_indexes"] == [10.0, 10.0] and out["ranges"]["content_wide_faster_cold_by_index"]["none"] == [0, 0]
        assert o["exactness_checks"]["life"] == 8 and o["lifecycle"]["L3"]["speedup"] == 3.0 and out["ranges"]["lifecycle_speedup_all"] == [3.0, 3.0]
        assert out["ranges"]["lifecycle_read_rows"] == dict(intervals=[100, 100], normalized=[900, 900])
        assert out["ranges"]["logged_over_distinct_bytes"] == [5.0, 5.0] and out["ranges"]["blocks_over_distinct"] == [2.25, 2.25]
    with tempfile.TemporaryDirectory() as t:                     # a missing input
        res = build(t); (res / "wiki" / "wiki_s2__index.json").unlink()
        code, out = run(t)
        assert code == 1 and out["status"] == "incomplete" and len(out["missing"]) == 1 and "samples" not in out
    expect("wiki/wiki_s3__checks.json", lambda d: d["meta"].pop("life_s"), "incomplete")             # stopped before its last stage
    checks = "wiki/wiki_s1__checks.json"
    for damage in (lambda d: d.update(mismatch={"x": 1}),                                               # stopped on a mismatch
                   lambda d: d["point"].append(dict(equal=False)),                                      # a reconstruction differs
                   lambda d: d["life"].append(dict(equal_on_2000_traces=True)),                         # lifecycle on a subset only
                   lambda d: d["life"].pop(),                                                           # a lifecycle plan missing
                   lambda d: d["life"].append(dict(d["life"][0])),                                      # a duplicate lifecycle record
                   lambda d: d["life"][1].update(warm_ms=[1.0]),                                        # too few warm runs
                   lambda d: d["life"][2].update(error="over limit"),                                   # a lifecycle error
                   lambda d: d["life"][0].update(policy="other"),                                       # another policy
                   lambda d: d["life"][0].update(query="L1 something else"),                            # another query
                   lambda d: d["life"][3].update(cold_ms=[1.0]),                                        # too few cold runs
                   lambda d: d["life"][4].update(warm_ms=[True, True, True]),                           # boolean latencies
                   lambda d: d["life"][5].update(warm_ms=[1.0, float("inf"), 1.0]),                     # non-finite latency
                   lambda d: d["life"][6].update(read_rows="100"),                                      # malformed read_rows
                   lambda d: d["selfcheck"].update({ws.SELFCHECK[0]: 0}),                               # selfcheck missed a damage
                   lambda d: d["selfcheck"].pop(ws.SELFCHECK[3]),                                       # selfcheck case missing
                   lambda d: d["any"].append(dict(policy="wiki", plan="I.expand", error="over limit")), # an unverified ctx_any result
                   lambda d: d["meta"].update(data="wiki_s2"),                                          # another sample's run
                   lambda d: d["meta"].update(workload="swe"),                                          # the SWE predicates
                   lambda d: d["meta"].update(warm=1),                                                  # other repetitions
                   lambda d: d["convert"].update(spans=1),                                              # another store
                   lambda d: d.update(spans=29999),                                                     # a truncated loaded store
                   lambda d: d.update(traces=2999),
                   lambda d: d["meta"]["settings"].update(merge_tree_min_rows_for_concurrent_read=163840),   # default read thresholds
                   lambda d: d["meta"].update(check_timeout_s=1800),                                    # another check limit
                   lambda d: d["meta"].update(ckpt=64),                                                 # another checkpoint interval
                   lambda d: d["meta"].update(clickhouse="25.1.1.1")):                                  # another server version
        expect(checks, damage)
    index = "wiki/wiki_s2__index.json"
    for damage in (lambda d: d["index"]["wiki"]["predicates"].pop("eval:sub8:9"),                      # a held-out predicate missing
                   lambda d: d["index"]["wiki"].update(records=[r for r in recs_of(d) if r["index"] != "bloom4"]),   # a kind missing
                   lambda d: d["index"]["wiki"]["variants"].pop("bloom4"),                              # a variant missing
                   lambda d: d["index"]["wiki"]["calib"].pop("token"),                                  # a calibration kind missing
                   lambda d: next(r for r in recs_of(d) if r["pred"].startswith("calib") and r["plan"] == "W").update(set_equal=False),
                   lambda d: next(r for r in recs_of(d) if r["pred"].startswith("calib") and r["plan"] == "F+I.expand").update(error="over limit"),
                   lambda d: d["index"]["wiki"].update(records=[r for r in recs_of(d) if not (r["pred"] == "calib:sub16:2" and r["plan"] == "W" and r["index"] == "none")]),
                   lambda d: recs_of(d).append(dict(recs_of(d)[0])),                                    # a duplicate record
                   lambda d: next(r for r in recs_of(d) if r["pred"].startswith("eval")).update(warm_ms=[1.0]),   # too few repetitions
                   lambda d: d["meta"].update(seed=1),                                                  # the wrong predicate seed
                   lambda d: d["meta"].update(indexes=["none"])):                                       # not the six kinds
        expect(index, damage)
    expect("ing_wiki_v1/wiki_s3__wiki.json", lambda d: d["variants"]["nokey"].update(wrong=1))          # an exact variant wrong
    expect("ing_wiki_v1/wiki_s3__wiki.json", lambda d: d.update(data="/x/data/wiki_s1"))                # another sample's ablation
    expect("wiki/wiki_s1/wiki_stats.json", lambda d: d["shape"].update(identical=2))                     # inconsistent shape counts
    expect("wiki/wiki_s1/wiki_stats.json", lambda d: d.update(seed=2))                                   # another sample's stats
    expect("wiki/wiki_s1/wiki_stats.json", lambda d: d.update(page_ids=d["page_ids"][:-1] + [d["page_ids"][0]]))   # repeated page
    expect("wiki/wiki_s2/convert_stats.json", lambda d: d.update(interleave=1))                          # other ingestion settings
    expect("wiki/wiki_s2/convert_stats.json", lambda d: d.pop("dictionary_bytes"))                       # no dictionary size
    expect("wiki/wiki_s2/convert_stats.json", lambda d: d.update(dictionary_bytes=3 * 10 ** 6))           # positive but wrong size
    expect("ing_wiki_v1/wiki_s3__wiki.json", lambda d: d.update(dictionary_bytes=1))                     # ablation disagrees on size
    expect("ing_wiki_v1/wiki_s1__wiki.json", lambda d: d.update(distinct_messages=1))                    # another dictionary
    expect("ing_wiki_v1/wiki_s1__wiki.json", lambda d: [d.update(requests=29000)] + [x.update(requests=29000) for x in d["variants"].values()])
    expect("wiki/wiki_s3/convert_stats.json", lambda d: d["policies"]["wiki"].update(array_elems=1))     # another ingested store
    expect("wiki/wiki_s3__index.json", lambda d: d["index"]["wiki"].update(records="oops"))              # unexpected schema error
    with tempfile.TemporaryDirectory() as t:                     # an unexpected error replaces an earlier successful output
        res = build(t)
        code, out = run(t)
        assert code == 0 and out["status"] == "ok"
        edit(res / "wiki" / "wiki_s1__index.json", lambda d: d.update(index=None))
        code, out = run(t)
        assert code == 1 and out["status"] == "blocked" and "samples" not in out
    print("TEST_WIKI_SUMMARY_OK")


if __name__ == "__main__":
    main()
