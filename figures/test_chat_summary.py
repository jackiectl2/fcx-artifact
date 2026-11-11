"""Test of chat_summary.py on self-contained synthetic inputs (no result file is read): a complete tree must pass, and each
damaged tree must give the status the protocol requires.
usage: python figures/test_chat_summary.py      (project .venv or the project SIF)
"""
import hashlib, json, sys, tempfile
from pathlib import Path

import chat_summary as cs

ws, M = cs.ws, cs.MANAGERS
CALLS = 45000


def convert(s):
    return dict(src=f"/x/data/cap_chat_s{s}", traces=3000, spans=CALLS, interleave=64, budget=64, ckpt=32, distinct_messages=60000, dictionary_bytes=3 * 10 ** 7,
                policies={m: dict(transitions_per_span=1.0 + i, mean_context_len=10.0 + i, array_elems=(10 + i) * CALLS, opens=CALLS, closes=i * CALLS, wide_chars=3 * 10 ** 8,
                                  ingest=dict(requests=CALLS, delta=1, interleave=64)) for i, m in enumerate(M)})


def ablate(s, m):
    v = {}
    for name in ws.ing.EXACT + ws.ing.LOSSY:
        x = dict(requests=CALLS, records=2000, payload_bytes=50000, wrong=0 if name in ws.ing.EXACT else 100, cpu_s=1.5, must_be_exact=name in ws.ing.EXACT,
                 logical_bytes=3 * 10 ** 8)
        if name in ws.ing.LOG_VARIANTS:
            x.update(checkpoint_bytes=8000, fallback_requests=0)
        v[name] = x
    return dict(data=f"/x/data/cap_chat_s{s}", manager=m, interleave=64, seed=1, ckpt=32, requests=CALLS, distinct_messages=60000,
                dictionary_bytes=3 * 10 ** 7 // (1 + M.index(m)), slice_requests=dict(dup=100), variants=v)


def meta(s, stages, policies, index):
    m = {f"{st}_s": 1.0 for st in stages}
    m.update(data=f"cap_chat_s{s}", workload="chat", policies=list(policies), wide_policies=list(policies), threads=16, cold=2, warm=3, stages=list(stages),
             index_policies=list(policies), indexes=list(ws.INDEXES), seed=s if index else 0, ckpt=32, timeout_s=30, check_timeout_s=30,
             clickhouse="26.8.15.10", settings=dict(ws.SETTINGS))
    return m


def index_run(s, m):
    recs, preds = [], sorted(ws.EVAL | ws.CALIB)
    for kind in ws.INDEXES:
        for p in preds:
            for plan, ms in (("W", 40.0), ("F+I.expand", 5.0), ("F+N.inv", 6.0)):
                recs.append(dict(policy=m, index=kind, pred=p, plan=plan, matched_messages=3, set_equal=True, cardinality=7,
                                 value="7", warm_ms=[ms] * 3, cold_ms=[2 * ms] * 2))
    var = {k: dict(wide=dict(build_s=1.0, index_bytes=10, param=0), factored=dict(build_s=0.1, index_bytes=1, granularity=64, param=0)) for k in ws.INDEXES}
    return dict(meta=meta(s, ws.INDEX_STAGES, [m], True), convert=convert(s), spans=CALLS, traces=3000,
                index={m: dict(records=recs, predicates={p: f"text {p}" for p in preds}, variants=var, calib={k: {} for k in ws.INDEXES},
                               matched_messages={p: 3 for p in preds}, truth_spans={p: 7 for p in preds})})


def checks_run(s):
    return dict(meta=meta(s, ws.CHECK_STAGES, M, False), convert=convert(s), spans=CALLS, traces=3000,
                matched={n: (30000 if n == "r10-s1" else 50) for n in cs.SETS} | {"c0.1-s1": 9},
                any=[dict(policy=m, matched=n, n_matched=(30000 if n == "r10-s1" else 50), filter=f, plan=p, set_equal=True) for m in M for n in cs.SETS
                     for f in cs.FILTERS for p in sorted(cs.CORE_ANY | (set() if n == "r10-s1" else {"A.bloom"}) | (cs.FILTER_ANY if f != "none" else set()))],
                agg=[dict(policy=m, case=c, plan=p, **({} if p == "A (reference)" else dict(rows_differ=0))) for m in M for c, ps in cs.AGG_CASES.items() for p in sorted(ps)],
                point=[dict(policy=m, n=n, seed=sd, plan=p, equal=True) for m in M for n, sd in sorted(cs.POINTS) for p in "ADIN"],
                life=[dict(policy=m, query=q, plan=pl, equal=True, warm_ms=[ms] * 3, cold_ms=[2 * ms] * 2, read_rows=rows)
                      for m in M for q in ws.LIFE_QUERIES for pl, ms, rows in (("I", 4.0, 100), ("N", 8.0, 900))],
                bytes={m: dict(spans=CALLS, requests_not_reproduced=0, checks={k: dict(rows_differ=0) for k in cs.BYTES_CHECKS}) for m in M},
                checks={c.format(m): 0 for m in M for c in cs.STORE_CHECKS}, selfcheck={k: v for k, v in zip(ws.SELFCHECK, (3, 1, 5, 0))})


def build(tmp):
    res = Path(tmp) / "results"
    (res / "chat").mkdir(parents=True); (res / "ing_chat_v1").mkdir(parents=True)
    root = Path(tmp)
    (root / "cluster").mkdir(); (root / "fcx").mkdir()
    (root / "cluster" / "chat_capture.sbatch").write_text("$RUN ingest_run.py --data x --managers m --work w --delta 1 --interleave 64 --seed 1\n")
    files = ["fcx/capture.py", "fcx/ingest.py", "fcx/ingest_run.py", "fcx/convert.py"]
    for k, f in enumerate(files):
        (root / f).write_text(f"code {k}\n")
    json.dump(dict(cs.MANIFEST, purpose="test", sbatch_sha256=hashlib.sha256((root / "cluster" / "chat_capture.sbatch").read_bytes()).hexdigest(),
                   code_md5_printed_by_job=hashlib.md5(b"".join((root / f).read_bytes() for f in files)).hexdigest()),
              open(res / "chat" / "RUN-MANIFEST.json", "w"))
    json.dump(dict(conversations=837989, min_turns=10, seed=1, excluded={"few_turns": 813491, "duplicate_hash": 0}, eligible=24498, eligible_turns={"10": 24000, "11": 498}, src=cs.SOURCES,
                   samples={f"s{s}": dict(conversations=3000, calls=CALLS, messages=2 * CALLS, content_chars=10 ** 8) for s in cs.SEEDS}),
              open(res / "chat" / "chat_prep_stats.json", "w"))
    for s in cs.SEEDS:
        (res / "chat" / f"cap_chat_s{s}").mkdir()
        json.dump(dict(src=f"/x/data/chat_pools/chat_pool_s{s}.parquet", format="swegym", traces=3000, spans=CALLS, corpus_trajectories=3000, sample=0, seed=s,
                       managers={m: dict(requests=CALLS) for m in M}), open(res / "chat" / f"cap_chat_s{s}" / "capture_stats.json", "w"))
        json.dump(convert(s), open(res / "chat" / f"cap_chat_s{s}" / "convert_stats.json", "w"))
        json.dump(checks_run(s), open(res / "chat" / f"cap_chat_s{s}__checks.json", "w"))
        for m in M:
            json.dump(ablate(s, m), open(res / "ing_chat_v1" / f"cap_chat_s{s}__{m}.json", "w"))
            json.dump(index_run(s, m), open(res / "chat" / f"cap_chat_s{s}__index_{m}.json", "w"))
    cs.ROOT, cs.RES = Path(tmp), res
    return res


def run(tmp):
    sys.argv = ["chat_summary.py", "--out", f"{tmp}/out.json"]
    try:
        cs.main(); code = 0
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


def main():
    with tempfile.TemporaryDirectory() as t:
        build(t)
        code, out = run(t)
        assert code == 0 and out["status"] == "ok", out.get("problems")
        r = out["ranges"]
        assert r["content_warm_median_all"] == [8.0, 8.0] and r["content_cold_median_all"] == [8.0, 8.0] and r["content_wide_faster_warm_max"] == 0
        per_set = lambda bloom: (len(cs.CORE_ANY) - 1 + bloom) + 2 * (len(cs.CORE_ANY) - 1 + bloom + len(cs.FILTER_ANY))
        n_any = 3 * 4 * ((len(cs.SETS) - 1) * per_set(1) + per_set(0))
        assert r["exactness_totals"] == dict(calls=3 * CALLS, any=n_any, agg=3 * 4 * 27, point=3 * 4 * 44, bytes=12 * CALLS, life=96), r["exactness_totals"]
        assert r["by_manager"]["prefix"]["logged_over_distinct_bytes"] == [10.0, 10.0] and r["by_manager"]["autogen_buf32"]["logged_over_distinct_bytes"] == [20.0, 20.0]
        assert r["by_manager"]["prefix"]["array_elems_over_distinct"] == [7.5, 7.5] and r["logged_over_distinct_bytes_all"] == [10.0, 40.0]
        assert r["by_manager"]["prefix"]["lifecycle_ms_by_query"]["L2"] == dict(intervals=[4.0, 4.0], normalized=[8.0, 8.0])
        assert r["by_manager"]["lc_trim32k"]["content_min_max_by_index"]["bloom4"]["warm_ms"] == [8.0, 8.0]
        assert r["by_manager"]["autogen_ht"]["rows_transitions_per_call"] == [3.0, 3.0] and r["lifecycle_speedup_all"] == [2.0, 2.0]
        assert r["by_manager"]["prefix"]["content_warm_median_by_index"]["none"] == [8.0, 8.0]
        assert "ci" not in out["samples"]["cap_chat_s1"]["managers"]["prefix"]["content_queries"]["none"]["warm_ms"]
    with tempfile.TemporaryDirectory() as t:                     # a missing input
        res = build(t); (res / "chat" / "cap_chat_s2__index_autogen_ht.json").unlink()
        code, out = run(t)
        assert code == 1 and out["status"] == "incomplete" and len(out["missing"]) == 1 and "samples" not in out
    expect("chat/cap_chat_s3__checks.json", lambda d: d["meta"].pop("life_s"), "incomplete")
    expect("chat/cap_chat_s1__index_prefix.json", lambda d: d["meta"].pop("index_s"), "incomplete")
    checks, idx = "chat/cap_chat_s1__checks.json", "chat/cap_chat_s2__index_lc_trim32k.json"
    for rel, damage in (
            (checks, lambda d: d.update(mismatch={"x": 1})),
            (checks, lambda d: d["point"].append(dict(equal=False))),
            (checks, lambda d: d["life"].pop()),
            (checks, lambda d: d["life"][0].update(policy="other")),
            (checks, lambda d: d["life"][1].update(warm_ms=[1.0])),
            (checks, lambda d: d["meta"].update(workload="wiki")),
            (checks, lambda d: d["meta"].update(policies=M[:3])),
            (checks, lambda d: d["any"].append(dict(policy="prefix", plan="I.expand", error="over limit"))),
            (checks, lambda d: d["selfcheck"].update({ws.SELFCHECK[0]: 0})),
            (checks, lambda d: d["any"].pop(0)),                                                   # a ctx_any record removed
            (checks, lambda d: d["any"].append(dict(d["any"][0]))),                                # a duplicated ctx_any record
            (checks, lambda d: d["any"].append(dict(d["any"][0], filter="other"))),               # an extra filter
            (checks, lambda d: d["any"].append(dict(d["any"][0], plan="X.bogus"))),               # an unknown plan
            (checks, lambda d: d.update(any=[x for x in d["any"] if not (x["plan"] == "I.join" and x["matched"] == "h:code" and x["filter"] == "repo")])),
            (checks, lambda d: d["any"].append(dict(d["any"][0], plan="I.join", filter="none"))),  # a filter-only plan without a filter
            (checks, lambda d: d.update(any=[x for x in d["any"] if not (x["plan"] == "A.bloom" and x["matched"] == "r1-s2")])),   # bloom missing
            (checks, lambda d: d["any"].append(dict([x for x in d["any"] if x["matched"] == "r10-s1"][0], plan="A.bloom"))),     # bloom not applicable
            (checks, lambda d: d["matched"].update({"h:story": 51})),                              # n_matched differs from the matched set
            (checks, lambda d: (d["matched"].update({"h:story": -1}), [x.update(n_matched=-1) for x in d["any"] if x["matched"] == "h:story"])),  # negative size
            ("chat/RUN-MANIFEST.json", lambda d: d.update(capture_job="1")),
            ("chat/RUN-MANIFEST.json", lambda d: d.update(ingest_arguments="--delta 1 --interleave 64 --seed 2")),
            (idx, lambda d: d["index"]["lc_trim32k"]["records"][0].update(policy="prefix")),       # an index record of another policy
            ("chat/chat_prep_stats.json", lambda d: d["eligible_turns"].update({"11": 497})),       # histogram does not sum to eligible
            ("chat/chat_prep_stats.json", lambda d: d["excluded"].update(few_turns=813490)),       # exclusions do not add up
            ("chat/chat_prep_stats.json", lambda d: d["samples"]["s1"].update(messages=1)),         # messages != 2 x calls
            ("ing_chat_v1/cap_chat_s2__lc_trim32k.json", lambda d: d["variants"]["fcx_b64"].pop("logical_bytes")),
            ("chat/cap_chat_s2/convert_stats.json", lambda d: d["policies"]["autogen_ht"].update(opens=1)),
            ("chat/RUN-MANIFEST.json", lambda d: d.update(sbatch_sha256="0" * 64)),
            (checks, lambda d: d["any"].append(dict(d["any"][0], policy="other"))),               # a record of another policy
            (checks, lambda d: d["agg"].pop()),                                                    # an aggregate record removed
            (checks, lambda d: d["agg"].append(dict(d["agg"][1]))),                                # a duplicated aggregate record
            (checks, lambda d: d["point"].pop()),                                                  # a reconstruction record removed
            (checks, lambda d: d["life"].append(dict(d["life"][0], policy="other"))),             # lifecycle of another policy
            (checks, lambda d: d["bytes"]["prefix"]["checks"].pop("ctx_sum[n_err]/all")),         # a byte check missing
            (checks, lambda d: d["checks"].pop("i_prefix_rows_minus_opens")),                      # a store check missing
            (idx, lambda d: d["index"]["lc_trim32k"]["records"][1].update(warm_ms=[5.0, 0.0, 5.0])),
            (idx, lambda d: d["index"]["lc_trim32k"]["matched_messages"].pop("eval:sub8:0")),
            ("ing_chat_v1/cap_chat_s1__prefix.json", lambda d: d["variants"]["hash_array"].update(logical_bytes=1)),
            ("chat/chat_prep_stats.json", lambda d: d.update(eligible=24000)),
            ("chat/chat_prep_stats.json", lambda d: d["src"].pop()),
            ("chat/cap_chat_s1/capture_stats.json", lambda d: d.update(seed=2)),
            ("chat/cap_chat_s3/convert_stats.json", lambda d: d["policies"]["autogen_ht"]["ingest"].update(delta=0)),
            (idx, lambda d: d.update(mismatch={"x": 1})),
            (idx, lambda d: d["meta"].update(seed=1)),
            (idx, lambda d: d["meta"].update(policies=["prefix"])),
            (idx, lambda d: d["index"]["lc_trim32k"].update(records=[x for x in d["index"]["lc_trim32k"]["records"]
                                                                     if not (x["plan"] == "W" and x["index"] == "none" and x["pred"] == "eval:sub8:0")])),
            (idx, lambda d: d["index"]["lc_trim32k"]["records"][0].update(set_equal=False)),
            (idx, lambda d: d["convert"].update(spans=1)),
            ("ing_chat_v1/cap_chat_s1__autogen_buf32.json", lambda d: d["variants"]["fcx_b64"].update(wrong=1)),
            ("ing_chat_v1/cap_chat_s1__autogen_buf32.json", lambda d: d.update(manager="prefix")),
            ("ing_chat_v1/cap_chat_s3__prefix.json", lambda d: d.update(requests=CALLS - 1)),
            ("chat/cap_chat_s2/capture_stats.json", lambda d: d.update(format="openhands")),
            ("chat/cap_chat_s2/capture_stats.json", lambda d: d["managers"].pop("lc_trim32k")),
            ("chat/cap_chat_s1/convert_stats.json", lambda d: d.update(interleave=1)),
            ("chat/chat_prep_stats.json", lambda d: d.update(min_turns=8)),
            ("chat/chat_prep_stats.json", lambda d: d["samples"]["s2"].update(conversations=2999))):
        expect(rel, damage)
    print("TEST_CHAT_SUMMARY_OK")


if __name__ == "__main__":
    main()
