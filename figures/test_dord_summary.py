"""Synthetic test of figures/dord_summary.py: a correct pair gives status ok with the expected speedups and paired ratios; each
damage (wrong layout recorded, different hosts, different predicates, a failed dictionary check, a missing first_span check,
an incomplete matrix) gives status blocked. usage: .venv/bin/python figures/test_dord_summary.py"""
import copy, json, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dord_summary as D                                          # noqa: E402

EVAL = [f"eval:sub{L}:{k}" for L in (8, 16, 32) for k in range(10)]
ALL = EVAL + [f"calib:sub{L}:{k}" for L in (8, 16, 32) for k in range(4)]
PREDS = {p: "x%d" % i for i, p in enumerate(ALL)}


def run(layout, f_scale, data="wiki_s1", md5="", order="id,first"):
    recs = []
    for k in D.INDEXES:
        for i, p in enumerate(ALL):
            w = 100.0 + i
            recs.append(dict(index=k, pred=p, plan="W", set_equal=True, cold_ms=[2 * w, 2 * w], warm_ms=[w, w, w]))
            recs.append(dict(index=k, pred=p, plan="F+I.expand", set_equal=True, cold_ms=[2 * w / (2 * f_scale)] * 2, warm_ms=[w / (4 * f_scale)] * 3))
    checks = {x: dict(ok=True, **({"wrong_first_span": 0} if layout == "first" else {})) for x in ["initial", *D.INDEXES]}
    var = {k: dict(wide=dict(param=0), factored=dict(granularity=512, param=0)) for k in D.INDEXES}
    load = {"mx_wiki": dict(compressed_bytes=100 if layout == "first" else 200, granularity=8192, column_bytes={c: 1 for c in D.COLUMNS[layout]}),
            "wx_wiki": dict(compressed_bytes=150)}
    meta = dict(dict_order=layout, workload="wiki", policies=["wiki"], wide_policies=["wiki"], index_policies=["wiki"], indexes=D.INDEXES, threads=16, cold=2,
                warm=3, stages=D.STAGES, clickhouse=D.CLICKHOUSE, settings=D.SETTINGS, preds_from="index.json", preds_from_md5=md5, host="gl1", data=data,
                pair_id="77_0", pair_order=order, **{f"{s}_s": 1.0 for s in D.STAGES})
    return dict(meta=meta, convert={"spans": 9}, spans=9, traces=3,
                index={"wiki": dict(records=recs, dict_checks=checks, variants=var, calib={k: {} for k in D.INDEXES}, load=load, predicates=dict(PREDS),
                                    truth_spans={p: 3 for p in ALL}, matched_messages={p: 1 for p in ALL})})


def status(mutate=None):
    with tempfile.TemporaryDirectory() as d:
        D.RES = Path(d); (D.RES / "dord").mkdir(); (D.RES / "wiki").mkdir()
        for s in (1, 2, 3):                                       # the legacy runs that supply the predicates
            json.dump({"index": {"wiki": {"predicates": dict(PREDS), "seed": s}}}, open(D.RES / "wiki" / f"wiki_s{s}__index.json", "w"))
        for s in (1, 2, 3):
            md5 = D.md5(D.RES / "wiki" / f"wiki_s{s}__index.json")
            runs = {"id": run("id", 1.0, f"wiki_s{s}", md5), "first": run("first", 2.0, f"wiki_s{s}", md5)}
            if mutate:
                mutate(runs)
            for L, r in runs.items():
                json.dump(r, open(D.RES / "dord" / f"wiki_s{s}__index_dord_{L}.json", "w"))
        sys.argv = ["x", "--out", str(D.RES / "out.json"), "--groups", "wiki"]
        try:
            D.main()
        except SystemExit:
            pass
        return json.load(open(D.RES / "out.json"))


ok = status()
assert ok["status"] == "ok", ok["problems"]
g = ok["groups"]["wiki"]["ranges"]
assert g["speedup_id_warm_all_indexes"] == dict(min=4.0, max=4.0) and g["speedup_first_warm_all_indexes"] == dict(min=8.0, max=8.0), g
assert g["fcx_id_over_first_cold"] == dict(min=2.0, max=2.0) and g["wide_id_over_first_warm"] == dict(min=1.0, max=1.0), g
assert g["dictionary_id_over_first"] == dict(min=2.0, max=2.0) and g["parameter_changes"] == 0, g
damages = {
    "wrong layout": lambda r: r["first"]["meta"].update(dict_order="id"),
    "hosts differ": lambda r: r["first"]["meta"].update(host="gl2"),
    "predicates differ": lambda r: r["first"]["index"]["wiki"]["predicates"].update({EVAL[0]: "y"}),
    "check failed": lambda r: r["first"]["index"]["wiki"]["dict_checks"]["ngrams3"].update(ok=False),
    "first_span unchecked": lambda r: r["first"]["index"]["wiki"]["dict_checks"]["initial"].pop("wrong_first_span"),
    "matrix incomplete": lambda r: r["id"]["index"]["wiki"]["records"].pop(),
    "calibration record missing": lambda r: r["first"]["index"]["wiki"]["records"].remove(
        next(x for x in r["first"]["index"]["wiki"]["records"] if x["pred"].startswith("calib"))),
    "both use a wrong source": lambda r: [x["meta"].update(preds_from_md5="0" * 32) for x in r.values()],
    "pair IDs differ": lambda r: r["first"]["meta"].update(pair_id="78_0"),
    "initial granularity": lambda r: r["first"]["index"]["wiki"]["load"]["mx_wiki"].update(granularity=512),
    "wrong data": lambda r: [x["meta"].update(data="wiki_s9") for x in r.values()],
    "different store": lambda r: r["first"].update(spans=10),
    "one host missing": lambda r: r["first"]["meta"].pop("host"),
}
pkg = status(lambda r: [x["meta"].pop("host") for x in r.values()])            # package copies: no host field in either run
assert pkg["status"] == "ok", pkg["problems"]
for name, f in damages.items():
    st = status(f)["status"]
    assert st == "blocked", (name, st)
print("TEST_DORD_SUMMARY_OK: correct pair ok;", len(damages), "damages blocked")
