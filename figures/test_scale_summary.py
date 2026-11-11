"""Test of scale_summary.py on self-contained synthetic inputs (no result file is read): a complete tree must pass, and each
damaged tree must give status "blocked" or "incomplete".
usage: python figures/test_scale_summary.py      (project .venv or the project SIF)
"""
import copy, hashlib, json, sys, tempfile
from pathlib import Path

import scale_summary as sc
import test_chat_summary as tc
import test_wiki_summary as tw

ws, cs = sc.ws, sc.cs


def exact_checks(policies, needles, calls, base):
    """A checks run whose record matrices are exact for each policy (as bench.py writes them), on top of base (meta, convert)."""
    sets = sorted([f"r{q}-s{k}" for q in ("0.1", "1", "10") for k in (1, 2, 3)] + needles)
    matched = {n: (30000 if n == "r10-s1" else 50) for n in sets}
    anyr, agg, pts, life, byt, chk = [], [], [], [], {}, {}
    cases = {**{f"ctx_count/{x}": ["A (reference)", "D.replay", "I.expand", "N.inv"] for x in ("r1-s1", needles[0])},
             **{f"ctx_sum[{v}]/{x}": ["A (reference)", "I.expand", "N.inv"] for v in ("n_chars", "n_bytes", "n_err") for x in ("r1-s1", needles[0])},
             **{f"ctx_sum[{v}]/all": ["A (reference)", "D.sweep", "I.expand", "N.fwd"] for v in ("n_chars", "n_bytes", "n_err")}}
    for m in policies:
        for n in sets:
            for f in cs.FILTERS:
                plans = cs.CORE_ANY | ({"A.bloom"} if matched[n] <= cs.BLOOM_MAX else set()) | (cs.FILTER_ANY if f != "none" else set())
                anyr += [dict(policy=m, matched=n, n_matched=matched[n], filter=f, plan=q, set_equal=True) for q in sorted(plans)]
        agg += [dict(policy=m, case=c, plan=q, **({} if q == "A (reference)" else dict(rows_differ=0))) for c, qs in cases.items() for q in qs]
        pts += [dict(policy=m, n=n, seed=sd, plan=q, equal=True) for n, sd in sorted(cs.POINTS) for q in "ADIN"]
        life += [dict(policy=m, query=q, plan=pl, equal=True, warm_ms=[ms] * 3, cold_ms=[2 * ms] * 2, read_rows=rows)
                 for q in ws.LIFE_QUERIES for pl, ms, rows in (("I", 4.0, 100), ("N", 8.0, 900))]
        byt[m] = dict(spans=calls, requests_not_reproduced=0, checks={k: dict(rows_differ=0) for k in
                      {f"ctx_any/{x}" for x in needles} | {"ctx_sum[n_bytes]/all", "ctx_sum[n_err]/all", f"ctx_count/{needles[0]} (1% of spans)"}})
        chk.update({c.format(m): 0 for c in cs.STORE_CHECKS})
    return dict(base, matched=matched, any=anyr, agg=agg, point=pts, life=life, bytes=byt, checks=chk,
                selfcheck={k: v for k, v in zip(ws.SELFCHECK, (3, 1, 5, 0))})


def load_fix(r, p, big):
    r["index"][p]["load"] = {f"wx_{p}": dict(bytes_on_disk=10 ** 9 if big else 10 ** 8), f"mx_{p}/none": dict(bytes_on_disk=2 * 10 ** 7 if big else 3 * 10 ** 6),
                             "i": dict(bytes_on_disk=10 ** 7 if big else 10 ** 6)}
    return r


def frozen(conv):
    conv.update(budget=64, ckpt=32)
    for v in conv["policies"].values():
        v["ingest"].update(delta=1, interleave=64, requests=conv["spans"])
    return conv


def wiki_pair(res):
    small_conv = frozen(tw.convert(1))
    small_ix = load_fix(tw.index_run(1), "wiki", False)
    small_ix["convert"] = small_conv
    (res / "wiki" / "wiki_s1").mkdir(parents=True, exist_ok=True)
    json.dump(small_conv, open(res / "wiki" / "wiki_s1" / "convert_stats.json", "w"))
    json.dump(small_ix, open(res / "wiki" / "wiki_s1__index.json", "w"))
    json.dump(dict(page_ids=list(range(3000))), open(res / "wiki" / "wiki_s1" / "wiki_stats.json", "w"))
    sck = tw.checks_run(1)
    json.dump(exact_checks(["wiki"], sc.NEEDLES["wiki"], 30000, dict(meta=sck["meta"], convert=small_conv, spans=30000, traces=3000)),
              open(res / "wiki" / "wiki_s1__checks.json", "w"))
    md5 = hashlib.md5(open(res / "wiki" / "wiki_s1__index.json", "rb").read()).hexdigest()
    conv = dict(tw.convert(1), src="/x/data/wikix_s1", traces=30000, spans=300000)
    conv["policies"]["wiki"].update(wide_chars=10 ** 8, array_elems=900000, ingest=dict(requests=300000))
    frozen(conv)
    (res / "scale" / "wikix_s1").mkdir(parents=True, exist_ok=True)
    json.dump(conv, open(res / "scale" / "wikix_s1" / "convert_stats.json", "w"))
    json.dump(dict(seed=1, pages=30000, traces=30000, spans=300000, blocks=900000, logical_bytes=10 ** 8, min_revs=10, max_revs=256, page_ids=list(range(30000))),
              open(res / "scale" / "wikix_s1" / "wiki_stats.json", "w"))
    m0 = tw.checks_run(1)["meta"]; m0.update(data="wikix_s1", index_policies=["wiki"])
    ck = exact_checks(["wiki"], sc.NEEDLES["wiki"], 300000, dict(meta=m0, convert=conv, spans=300000, traces=30000))
    ix = load_fix(copy.deepcopy(small_ix), "wiki", True); ix.update(convert=conv, spans=300000, traces=30000)
    ix["meta"].update(data="wikix_s1", seed=1, preds_from="index.json", preds_from_md5=md5)
    json.dump(ck, open(res / "scale" / "wikix_s1__checks.json", "w"))
    json.dump(ix, open(res / "scale" / "wikix_s1__index.json", "w"))


def chat_pair(res):
    small_conv = tc.convert(1)
    small_ix = load_fix(tc.index_run(1, "prefix"), "prefix", False)
    (res / "chat" / "cap_chat_s1").mkdir(parents=True, exist_ok=True); (res / "ing_chat_v1").mkdir(parents=True, exist_ok=True)
    json.dump(small_conv, open(res / "chat" / "cap_chat_s1" / "convert_stats.json", "w"))
    json.dump(small_ix, open(res / "chat" / "cap_chat_s1__index_prefix.json", "w"))
    json.dump(tc.ablate(1, "prefix"), open(res / "ing_chat_v1" / "cap_chat_s1__prefix.json", "w"))
    sck = tc.checks_run(1)
    json.dump(exact_checks(cs.MANAGERS, sc.NEEDLES["chat"], tc.CALLS, dict(meta=sck["meta"], convert=small_conv, spans=tc.CALLS, traces=3000)),
              open(res / "chat" / "cap_chat_s1__checks.json", "w"))
    md5 = hashlib.md5(open(res / "chat" / "cap_chat_s1__index_prefix.json", "rb").read()).hexdigest()
    calls = 360000
    conv = dict(src="/x/data/cap_chatall_s1", traces=cs.ELIGIBLE, spans=calls, interleave=64, budget=64, ckpt=32, distinct_messages=500000,
                dictionary_bytes=4 * 10 ** 8, policies={"prefix": dict(wide_chars=6 * 10 ** 9, ingest=dict(requests=calls), new_messages=500000,
                                                                       new_message_bytes=4 * 10 ** 8)})
    frozen(conv)
    (res / "scale" / "cap_chatall_s1").mkdir(parents=True, exist_ok=True)
    json.dump(dict(src="/x/data/chat_pools_all/chat_pool_s1.parquet", format="swegym", sample=0, seed=1, corpus_trajectories=cs.ELIGIBLE,
                   traces=cs.ELIGIBLE, spans=calls, managers={"prefix": dict(requests=calls, logical_bytes=6 * 10 ** 9)}),
              open(res / "scale" / "cap_chatall_s1" / "capture_stats.json", "w"))
    (res / "scale" / "cap_chatall_s1").mkdir(parents=True, exist_ok=True)
    json.dump(conv, open(res / "scale" / "cap_chatall_s1" / "convert_stats.json", "w"))
    json.dump(dict(min_turns=10, seed=1, conversations=cs.CORPUS, eligible=cs.ELIGIBLE, src=cs.SOURCES, excluded={"few_turns": cs.CORPUS - cs.ELIGIBLE},
                   eligible_turns={"10": cs.ELIGIBLE}, samples={"s1": dict(conversations=cs.ELIGIBLE, calls=calls, messages=2 * calls, content_chars=10 ** 9)}),
              open(res / "scale" / "chat_pools_all_prep_stats.json", "w"))
    m0 = tc.checks_run(1)["meta"]
    m0.update(data="cap_chatall_s1", policies=["prefix"], wide_policies=["prefix"], index_policies=["prefix"])
    ck = exact_checks(["prefix"], sc.NEEDLES["chat"], calls, dict(meta=m0, convert=conv, spans=calls, traces=cs.ELIGIBLE))
    ix = load_fix(copy.deepcopy(small_ix), "prefix", True); ix.update(convert=conv, spans=calls, traces=cs.ELIGIBLE)
    ix["meta"].update(data="cap_chatall_s1", seed=1, preds_from="index_prefix.json", preds_from_md5=md5)
    json.dump(ck, open(res / "scale" / "cap_chatall_s1__checks.json", "w"))
    json.dump(ix, open(res / "scale" / "cap_chatall_s1__index.json", "w"))


def build(tmp):
    res = Path(tmp) / "results"
    wiki_pair(res); chat_pair(res)
    sc.ROOT, sc.RES = Path(tmp), res
    sc.PAIRS["wiki"].update(small_index=res / "wiki" / "wiki_s1__index.json", small_convert=res / "wiki" / "wiki_s1" / "convert_stats.json",
                            small_checks=res / "wiki" / "wiki_s1__checks.json")
    sc.PAIRS["chat"].update(small_index=res / "chat" / "cap_chat_s1__index_prefix.json", small_convert=res / "chat" / "cap_chat_s1" / "convert_stats.json",
                            small_checks=res / "chat" / "cap_chat_s1__checks.json")
    return res


def run(tmp):
    sys.argv = ["scale_summary.py", "--out", f"{tmp}/out.json"]
    try:
        sc.main(); code = 0
    except SystemExit as e:
        code = 1 if e.code else 0
    return code, json.load(open(f"{tmp}/out.json"))


def expect(rel, damage, status="blocked"):
    with tempfile.TemporaryDirectory() as t:
        res = build(t)
        d = json.load(open(res / rel)); damage(d); json.dump(d, open(res / rel, "w"))
        code, out = run(t)
        assert code == 1 and out["status"] == status and "pairs" not in out, (rel, out.get("status"), out.get("problems"))


def main():
    with tempfile.TemporaryDirectory() as t:
        build(t)
        code, out = run(t)
        assert code == 0 and out["status"] == "ok", out.get("problems")
        r = out["pairs"]["wiki"]["ranges"]
        assert r["scale_calls"] == 10.0 and r["disk_ratio_large"] == 10 ** 9 / (3 * 10 ** 7) and r["warm_median_large"] == [10.0, 10.0]
        rc = out["pairs"]["chat"]["ranges"]
        assert rc["repetition_large"] == 15.0 and rc["repetition_small"] == 10.0, rc      # small: the prefix ablation's dictionary
    with tempfile.TemporaryDirectory() as t:
        res = build(t); (res / "scale" / "wikix_s1__index.json").unlink()
        code, out = run(t)
        assert code == 1 and out["status"] == "incomplete"
    expect("scale/wikix_s1__checks.json", lambda d: d["meta"].pop("life_s"), "incomplete")
    for rel, damage in (
            ("scale/wikix_s1__index.json", lambda d: d["meta"].update(preds_from_md5="0" * 32)),            # other predicate file
            ("scale/wikix_s1__index.json", lambda d: d["index"]["wiki"]["predicates"].update({"eval:sub8:0": "other"})),
            ("scale/wikix_s1__index.json", lambda d: d["meta"].update(seed=2)),
            ("scale/wikix_s1__index.json", lambda d: d.update(mismatch={"x": 1})),
            ("scale/wikix_s1__index.json", lambda d: d["index"]["wiki"]["records"].pop(0)),
            ("scale/wikix_s1__checks.json", lambda d: d["point"].append(dict(equal=False))),
            ("scale/wikix_s1__checks.json", lambda d: d["meta"].update(workload="chat")),
            ("scale/wikix_s1/wiki_stats.json", lambda d: d.update(page_ids=list(range(5, 30005)))),          # not nested
            ("scale/wikix_s1/wiki_stats.json", lambda d: d.update(pages=3000)),
            ("scale/wikix_s1/convert_stats.json", lambda d: d.update(interleave=1)),
            ("scale/cap_chatall_s1__index.json", lambda d: d["meta"].update(preds_from="index.json")),
            ("scale/cap_chatall_s1__checks.json", lambda d: d["meta"].update(policies=["prefix", "autogen_ht"])),
            ("scale/cap_chatall_s1__checks.json", lambda d: d["life"].pop()),
            ("scale/chat_pools_all_prep_stats.json", lambda d: d.update(eligible=24000)),
            ("scale/cap_chatall_s1/convert_stats.json", lambda d: d.update(traces=3000)),
            ("chat/cap_chat_s1__index_prefix.json", lambda d: d.update(mismatch={"x": 1})),
            ("scale/wikix_s1__checks.json", lambda d: d["any"].pop(0)),                                      # deletions per matrix section
            ("scale/wikix_s1__checks.json", lambda d: d["agg"].pop()),
            ("scale/wikix_s1__checks.json", lambda d: d["point"].pop()),
            ("scale/wikix_s1__checks.json", lambda d: d["bytes"]["wiki"]["checks"].pop("ctx_sum[n_err]/all")),
            ("scale/wikix_s1__checks.json", lambda d: d["checks"].pop("i_wiki_rows_minus_opens")),
            ("scale/cap_chatall_s1__checks.json", lambda d: d["any"].append(dict(d["any"][0], policy="autogen_ht"))),
            ("wiki/wiki_s1__checks.json", lambda d: d["life"].pop()),                                          # small lifecycle record missing
            ("chat/cap_chat_s1__checks.json", lambda d: d["meta"].update(threads=8)),                            # small run identity
            ("scale/cap_chatall_s1/capture_stats.json", lambda d: d.update(format="openhands")),
            ("scale/cap_chatall_s1/capture_stats.json", lambda d: d["managers"].update(autogen_ht=dict(requests=1))),
            ("scale/cap_chatall_s1/capture_stats.json", lambda d: d.update(seed=2)),
            ("scale/chat_pools_all_prep_stats.json", lambda d: d["src"].pop()),
            ("scale/chat_pools_all_prep_stats.json", lambda d: d["samples"]["s1"].update(messages=1)),
            ("ing_chat_v1/cap_chat_s1__prefix.json", lambda d: d.update(manager="autogen_ht")),                   # wrong-manager ablation
            ("ing_chat_v1/cap_chat_s1__prefix.json", lambda d: d.update(dictionary_bytes=-1)),
            ("scale/wikix_s1/convert_stats.json", lambda d: d["policies"]["wiki"].update(array_elems=1)),
            ("wiki/wiki_s1__checks.json", lambda d: d["point"][0].update(equal=False)),                          # an unequal small check
            ("chat/cap_chat_s1__checks.json", lambda d: d["agg"][1].update(rows_differ=1)),
            ("scale/chat_pools_all_prep_stats.json", lambda d: d.update(eligible_turns={"10": 24499, "11": -1})),   # negative histogram value
            ("scale/cap_chatall_s1/capture_stats.json", lambda d: d.update(src="/x/data/chat_pools_all_old/chat_pool_s1.parquet"))):
        expect(rel, damage)
    with tempfile.TemporaryDirectory() as t:                     # ckpt changed coherently in convert_stats and both bench runs
        res = build(t)
        for rel in ("scale/wikix_s1/convert_stats.json", "scale/wikix_s1__checks.json", "scale/wikix_s1__index.json"):
            d = json.load(open(res / rel))
            (d if rel.endswith("convert_stats.json") else d["convert"]).update(ckpt=64)
            json.dump(d, open(res / rel, "w"))
        code, out = run(t)
        assert code == 1 and out["status"] == "blocked", out.get("status")
    print("TEST_SCALE_SUMMARY_OK")


if __name__ == "__main__":
    main()
