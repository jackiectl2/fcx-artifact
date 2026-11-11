"""E-SCALE summary (refine-logs/E-SCALE-PROTOCOL.md, rule 5); reviewed before its output is used.

Pairs (large sample against the 3,000-item sample it contains, same predicates):
  wiki: results/scale/wikix_s1__{checks,index}.json, results/scale/wikix_s1/{convert_stats,wiki_stats}.json
        against results/wiki/wiki_s1__index.json (and results/wiki/wiki_s1/convert_stats.json)
  chat: results/scale/cap_chatall_s1__{checks,index}.json, results/scale/cap_chatall_s1/{convert_stats,capture_stats}.json,
        results/scale/chat_pools_all_prep_stats.json against results/chat/cap_chat_s1__index_prefix.json
        (and results/chat/cap_chat_s1/convert_stats.json)
Rules:
 - presence and completion: every input exists, each bench run recorded every planned stage; else "incomplete", no number;
 - identity: large runs have data wikix_s1 / cap_chatall_s1, workload wiki / chat, policies [wiki] / [prefix] (also wide and
   index policies), 16 threads, 2 cold / 3 warm, the planned stages, checkpoint 32, time limits 30 s, the paper's settings and
   ClickHouse version; the index runs seed 1, the six kinds, preds_from = basename of the small run (index.json /
   index_prefix.json) with an md5 equal to that of the local copy of the small run; their 42 predicates equal the small run's;
   embedded convert statistics equal convert_stats; loaded spans/traces equal its counts; wikix_s1 has 30,000 pages (seed 1,
   10-256 revisions) and contains every page of wiki_s1 (wiki_stats page IDs); cap_chatall_s1 has the 24,498 drawn conversations;
 - small runs (review round 2): the 3,000-item checks run of each pair (results/wiki/wiki_s1__checks.json,
   results/chat/cap_chat_s1__checks.json) and its index run pass the reviewed identity helpers (wiki_summary.run_identity,
   chat_summary.run_identity); the prefix ablation results/ing_chat_v1/cap_chat_s1__prefix.json passes ing_ext_summary.validate
   with requests, request bytes and a positive dictionary; cap_chatall_s1's capture_stats (source, swegym, sample 0, seed 1, the
   one manager prefix, 24,498 trajectories, calls) and all-conversation drawer statistics (the chat_summary drawer checks) are
   validated; the large convert_stats are cross-checked against wiki_stats (spans, blocks, bytes) or capture_stats (spans, bytes);
 - exactness: wiki_summary.table2 on each checks run (no difference, no problem) and an exact single-policy checks matrix
   (chat_summary.checks_matrix generalized to the workload's needles: ctx_any sets x filters x generated plans, aggregate cases,
   reconstruction targets, byte checks, store checks, 8 lifecycle records; the large runs hold no other policy); lifecycle
   (wiki_summary rule) for the large and the small checks run, per query;
 - content queries: wiki_summary.index_matrix and chat_summary.index_values on both runs of a pair, then
   fcx/index_report.summarize; per kind warm and cold median, min, max, wide-faster counts for large and small;
 - sizes: calls, request bytes (convert_stats wide_chars), distinct messages and dictionary bytes (for cap_chat_s1, whose store
   shares one dictionary among four managers, those of the prefix stream alone from results/ing_chat_v1); bytes on disk of the wide
   table (load wx_<p>) and of FCX's dictionary plus intervals without an index (load mx_<p>/none + i) in each index run.
Any problem gives status "blocked" with no numbers.
usage: .venv/bin/python figures/scale_summary.py [--out results/scale_summary.json]
"""
import argparse, hashlib, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "results"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "fcx"))
import chat_summary as cs                                        # noqa: E402
import ing_ext_summary as ing                                    # noqa: E402
import index_report                                              # noqa: E402
import wiki_summary as ws                                        # noqa: E402

NEEDLES = {"wiki": ["w:citeweb", "w:category", "w:infobox", "w:ref", "w:file", "w:reflist"], "chat": cs.NEEDLES}   # bench.py order


def checks_matrix(r, policy, needles, calls, only_policy):
    """chat_summary.checks_matrix for one policy and the workload's needles (review round 2): exact ctx_any, aggregate,
    reconstruction, byte, store and lifecycle matrices of that policy; with only_policy, no record of another policy."""
    probs, pol = [], (lambda x: x.get("policy"))
    sets = sorted([f"r{q}-s{k}" for q in ("0.1", "1", "10") for k in (1, 2, 3)] + needles)
    agg_cases = {**{f"ctx_count/{x}": {"A (reference)", "D.replay", "I.expand", "N.inv"} for x in ("r1-s1", needles[0])},
                 **{f"ctx_sum[{v}]/{x}": {"A (reference)", "I.expand", "N.inv"} for v in ("n_chars", "n_bytes", "n_err") for x in ("r1-s1", needles[0])},
                 **{f"ctx_sum[{v}]/all": {"A (reference)", "D.sweep", "I.expand", "N.fwd"} for v in ("n_chars", "n_bytes", "n_err")}}
    byte_checks = {f"ctx_any/{x}" for x in needles} | {"ctx_sum[n_bytes]/all", "ctx_sum[n_err]/all", f"ctx_count/{needles[0]} (1% of spans)"}
    if only_policy:
        for part in ("any", "agg", "point", "life"):
            if any(pol(x) != policy for x in r.get(part, [])):
                probs.append(f"{part}: a record of another policy")
        if set(r.get("bytes", {})) != {policy} or set(r.get("checks", {})) != {c.format(policy) for c in cs.STORE_CHECKS}:
            probs.append("bytes or store checks of another policy")
    g, nm = {}, {}
    for x in r.get("any", []):
        if pol(x) == policy:
            g.setdefault((x.get("matched"), x.get("filter")), []).append(x.get("plan"))
            nm.setdefault(x.get("matched"), set()).add(x.get("n_matched"))
    ok = set(g) == {(n, f) for n in sets for f in cs.FILTERS}
    for (n, f), v in g.items():
        k = r.get("matched", {}).get(n)
        valid_k = ws.nat(k) and k >= 0
        want = cs.CORE_ANY | ({"A.bloom"} if valid_k and k <= cs.BLOOM_MAX else set()) | (cs.FILTER_ANY if f != "none" else set())
        ok &= valid_k and nm.get(n) == {k} and sorted(v) == sorted(want)
    if not ok:
        probs.append("ctx_any matrix not exact")
    ga = {}
    for x in r.get("agg", []):
        if pol(x) == policy:
            ga.setdefault(x.get("case"), []).append(x.get("plan"))
    if set(ga) != set(agg_cases) or any(sorted(v) != sorted(agg_cases[k]) for k, v in ga.items()):
        probs.append("aggregate matrix not exact")
    pts = sorted((x.get("n"), x.get("seed"), x.get("plan")) for x in r.get("point", []) if pol(x) == policy)
    if pts != sorted((n, sd, q) for n, sd in cs.POINTS for q in "ADIN"):
        probs.append("reconstruction matrix not exact")
    if sum(pol(x) == policy for x in r.get("life", [])) != 8:
        probs.append("lifecycle records not 8")
    b = r.get("bytes", {}).get(policy, {})
    if set(b.get("checks", {})) != byte_checks or b.get("spans") != calls:
        probs.append("byte checks incomplete")
    if not {c.format(policy) for c in cs.STORE_CHECKS} <= set(r.get("checks", {})):
        probs.append("store checks incomplete")
    return probs


def convert_ok(conv, name, policies):
    """Frozen ingestion parameters of a convert_stats file (review round 3)."""
    pp = conv.get("policies", {})
    ok = (Path(str(conv.get("src", ""))).name == name and conv.get("budget") == 64 and conv.get("ckpt") == 32 and conv.get("interleave") == 64
          and list(pp) == policies and all(v.get("ingest", {}).get("delta") == 1 and v.get("ingest", {}).get("interleave") == 64
                                           and v.get("ingest", {}).get("requests") == conv.get("spans") for v in pp.values()))
    if ok and len(policies) == 1:                                # one-policy store: its new messages are the dictionary
        v = pp[policies[0]]
        ok = v.get("new_messages") == conv.get("distinct_messages") and v.get("new_message_bytes") == conv.get("dictionary_bytes")
    return ok


PAIRS = {
    "wiki": dict(data="wikix_s1", workload="wiki", policy="wiki", small_index=RES / "wiki" / "wiki_s1__index.json", small_name="index.json",
                 small_checks=RES / "wiki" / "wiki_s1__checks.json",
                 small_convert=RES / "wiki" / "wiki_s1" / "convert_stats.json", small_data="wiki_s1"),
    "chat": dict(data="cap_chatall_s1", workload="chat", policy="prefix", small_index=RES / "chat" / "cap_chat_s1__index_prefix.json",
                 small_checks=RES / "chat" / "cap_chat_s1__checks.json",
                 small_name="index_prefix.json", small_convert=RES / "chat" / "cap_chat_s1" / "convert_stats.json", small_data="cap_chat_s1"),
}


def identity(r, P, stages, conv, index, small_md5):
    m, probs = r.get("meta", {}), []
    pol = [P["policy"]]
    want = dict(data=P["data"], workload=P["workload"], policies=pol, wide_policies=pol, index_policies=pol, threads=ws.THREADS, cold=ws.COLD,
                warm=ws.WARM, stages=stages, ckpt=ws.CKPT, timeout_s=ws.TIMEOUT_S, check_timeout_s=ws.TIMEOUT_S, settings=ws.SETTINGS,
                clickhouse=ws.CLICKHOUSE)
    if index:
        want.update(indexes=ws.INDEXES, seed=1, preds_from=P["small_name"], preds_from_md5=small_md5)
    probs += [f"meta {k}={m.get(k)!r}, expected {v!r}" for k, v in want.items() if m.get(k) != v]
    if r.get("convert") != conv:
        probs.append("embedded convert statistics differ from convert_stats.json")
    if r.get("spans") != conv.get("spans") or r.get("traces") != conv.get("traces"):
        probs.append("loaded spans/traces differ from convert_stats")
    return probs


def sizes(r, p, conv, dictionary=None):
    """dictionary: (distinct messages, bytes) of this policy alone when the store's dictionary is shared by several policies."""
    L = r["load"]
    dm, db = dictionary or (conv["distinct_messages"], conv["dictionary_bytes"])
    return dict(calls=conv["spans"], request_bytes=conv["policies"][p]["wide_chars"], distinct_messages=dm,
                dictionary_bytes=db, wide_bytes_on_disk=L[f"wx_{p}"]["bytes_on_disk"],
                fcx_bytes_on_disk=L[f"mx_{p}/none"]["bytes_on_disk"] + L["i"]["bytes_on_disk"])


def queries(path, r, policy):
    probs = ws.index_matrix(r) + cs.index_values(r, policy)
    if probs:
        return None, probs
    _, rows = index_report.summarize(str(path))
    if sorted(x["index"] for x in rows) != sorted(ws.INDEXES) or not all(x["complete"] for x in rows):
        return None, ["index_report rows incomplete"]
    return {x["index"]: {k: {kk: vv for kk, vv in x[k].items() if kk != "ci"} for k in ("warm_ms", "cold_ms")} for x in rows}, []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RES / "scale_summary.json"))
    dst = Path(ap.parse_args().out)
    base = dict(protocol="refine-logs/E-SCALE-PROTOCOL.md")
    try:
        summarize(dst, base)
    except SystemExit:
        raise
    except Exception as e:
        ws.write(dict(base, status="blocked", problems=[f"unexpected {type(e).__name__}: {e}"]), dst)
        raise SystemExit(f"blocked: unexpected {type(e).__name__}: {e}")


def summarize(dst, base):
    S = RES / "scale"
    files = {}
    for key, P in PAIRS.items():
        d = P["data"]
        files[key] = dict(checks=S / f"{d}__checks.json", index=S / f"{d}__index.json", convert=S / d / "convert_stats.json",
                          small_index=P["small_index"], small_convert=P["small_convert"], small_checks=P["small_checks"])
        files[key]["stats"] = S / d / "wiki_stats.json" if key == "wiki" else S / "chat_pools_all_prep_stats.json"
        if key == "wiki":
            files[key]["small_stats"] = RES / "wiki" / "wiki_s1" / "wiki_stats.json"
        else:
            files[key]["capture"] = S / d / "capture_stats.json"
            files[key]["small_ablate"] = RES / "ing_chat_v1" / "cap_chat_s1__prefix.json"
    D, missing, unfinished = {}, [], []
    for key, fs in files.items():
        for k, p in fs.items():
            try:
                D[(key, k)] = json.load(open(p))
            except (OSError, ValueError) as e:
                missing.append(f"{p.relative_to(ROOT)}: {type(e).__name__}")
        for k, stages in (("checks", ws.CHECK_STAGES), ("index", ws.INDEX_STAGES)):
            r = D.get((key, k))
            if r is not None and not all(f"{st}_s" in r.get("meta", {}) for st in stages):
                unfinished.append(f"{key} {k}: not every planned stage recorded")
    if missing or unfinished:
        ws.write(dict(base, status="incomplete", missing=missing, unfinished=unfinished), dst)
        raise SystemExit(f"incomplete: {missing + unfinished}; see {dst}")

    blocked, out = [], dict(base, pairs={})
    for key, P in PAIRS.items():
        p, conv, ck, ix = P["policy"], D[(key, "convert")], D[(key, "checks")], D[(key, "index")]
        small, small_conv, small_ck = D[(key, "small_index")], D[(key, "small_convert")], D[(key, "small_checks")]
        small_md5 = hashlib.md5(open(P["small_index"], "rb").read()).hexdigest()
        small_dict = None                                         # cap_chat_s1's store shares one dictionary among four managers
        if not convert_ok(conv, P["data"], [p]):
            blocked.append(f"{key}: large convert_stats parameters")
        if not convert_ok(small_conv, P["small_data"], [p] if key == "wiki" else cs.MANAGERS):
            blocked.append(f"{key}: small convert_stats parameters")
        if key == "wiki":                                         # the small runs' identities (reviewed helpers)
            blocked += [f"wiki small checks: {x}" for x in ws.run_identity(small_ck, 1, ws.CHECK_STAGES, small_conv, False)]
            blocked += [f"wiki small index: {x}" for x in ws.run_identity(small, 1, ws.INDEX_STAGES, small_conv, True)]
        else:
            blocked += [f"chat small checks: {x}" for x in cs.run_identity(small_ck, 1, ws.CHECK_STAGES, small_conv, cs.MANAGERS, False)]
            blocked += [f"chat small index: {x}" for x in cs.run_identity(small, 1, ws.INDEX_STAGES, small_conv, ["prefix"], True)]
            ab = D[(key, "small_ablate")]                         # the prefix stream interned alone (E-CHAT ablation)
            pw = small_conv.get("policies", {}).get("prefix", {})
            if (ing.validate(ab, "cap_chat_s1", "prefix") or ab.get("requests") != small_conv.get("spans")
                    or not all(isinstance(v, dict) and v.get("logical_bytes") == pw.get("wide_chars") for v in ab.get("variants", {}).values())
                    or not (ws.nat(ab.get("distinct_messages")) and ab["distinct_messages"] > 0 and ws.nat(ab.get("dictionary_bytes")) and ab["dictionary_bytes"] > 0)):
                blocked.append("chat: prefix ablation identity, requests, bytes or dictionary")
                continue
            small_dict = (ab["distinct_messages"], ab["dictionary_bytes"])
        blocked += [f"{key} checks: {x}" for x in identity(ck, P, ws.CHECK_STAGES, conv, False, small_md5)]
        blocked += [f"{key} index: {x}" for x in identity(ix, P, ws.INDEX_STAGES, conv, True, small_md5)]
        st = D[(key, "stats")]
        if key == "wiki":
            small_ids = set(D[(key, "small_stats")].get("page_ids", []))
            ids = st.get("page_ids", [])
            pw = conv.get("policies", {}).get("wiki", {})
            if not (st.get("seed") == 1 and st.get("pages") == 30000 and st.get("traces") == 30000 and st.get("min_revs") == ws.MIN_REVS
                    and st.get("max_revs") == ws.MAX_REVS and len(set(ids)) == 30000 and len(small_ids) == 3000 and small_ids <= set(ids)
                    and conv.get("traces") == 30000 and conv.get("spans") == st.get("spans") and Path(str(conv.get("src", ""))).name == "wikix_s1"
                    and pw.get("array_elems") == st.get("blocks") and pw.get("wide_chars") == st.get("logical_bytes")
                    and pw.get("ingest", {}).get("requests") == st.get("spans")):
                blocked.append("wiki: wikix_s1 identity, size, nesting or ingested counts")
        else:
            cap, ss = D[(key, "capture")], st.get("samples", {}).get("s1", {})
            ex, et = st.get("excluded", {}), st.get("eligible_turns", {})
            calls = ss.get("calls")
            if not (st.get("min_turns") == 10 and st.get("seed") == 1 and st.get("conversations") == cs.CORPUS and st.get("eligible") == cs.ELIGIBLE
                    and st.get("src") == cs.SOURCES and list(st.get("samples", {})) == ["s1"] and ss.get("conversations") == cs.ELIGIBLE
                    and isinstance(ex, dict) and all(ws.nat(v) and v >= 0 for v in ex.values()) and st["conversations"] == st["eligible"] + sum(ex.values())
                    and isinstance(et, dict) and sum(et.values()) == cs.ELIGIBLE and all(k.isdigit() and int(k) >= 10 and ws.nat(v) and v > 0 for k, v in et.items())
                    and ws.nat(calls) and calls >= 10 * cs.ELIGIBLE and ss.get("messages") == 2 * calls):
                blocked.append("chat: all-conversation drawer identity or arithmetic")
            if not (Path(str(cap.get("src", ""))).name == "chat_pool_s1.parquet" and Path(str(cap.get("src", ""))).parent.name == "chat_pools_all" and cap.get("format") == "swegym"
                    and cap.get("sample") == 0 and cap.get("seed") == 1 and list(cap.get("managers", {})) == ["prefix"]
                    and cap.get("corpus_trajectories") == cs.ELIGIBLE and cap.get("traces") == cs.ELIGIBLE and cap.get("spans") == calls
                    and cap["managers"]["prefix"].get("requests") == calls):
                blocked.append("chat: capture identity or counts")
            pw = conv.get("policies", {}).get("prefix", {})
            if not (conv.get("traces") == cs.ELIGIBLE and conv.get("spans") == calls and Path(str(conv.get("src", ""))).name == "cap_chatall_s1"
                    and pw.get("wide_chars") == cap.get("managers", {}).get("prefix", {}).get("logical_bytes") and pw.get("ingest", {}).get("requests") == calls):
                blocked.append("chat: cap_chatall_s1 ingested counts")
        if list(conv.get("policies", {})) != [p] or conv.get("interleave") != 64:
            blocked.append(f"{key}: convert_stats policies or interleave")
        t, diff, probs = ws.table2(ck)
        if diff or probs:
            blocked.append(f"{key} checks: {diff} differences, {probs}")
        blocked += [f"{key} checks matrix: {x}" for x in checks_matrix(ck, p, NEEDLES[key], conv.get("spans"), True)]
        _, sdiff, sprobs = ws.table2(small_ck)
        if sdiff or sprobs:
            blocked.append(f"{key} small checks: {sdiff} differences, {sprobs}")
        blocked += [f"{key} small checks matrix: {x}" for x in checks_matrix(small_ck, p, NEEDLES[key], small_conv.get("spans"), key == "wiki")]
        lc, lp = cs.lifecycle(ck, p)
        lcs, lps = cs.lifecycle(small_ck, p)
        if lp or lps:
            blocked.append(f"{key} lifecycle: {lp + lps}")
        rl, rs = ix.get("index", {}).get(p, {}), small.get("index", {}).get(p, {})
        if ix.get("mismatch") or small.get("mismatch"):
            blocked.append(f"{key}: index run stopped on a mismatch")
            continue
        if rl.get("predicates") != rs.get("predicates"):
            blocked.append(f"{key}: the large run's predicates differ from the small run's")
        ql, pl = queries(files[key]["index"], rl, p)
        qs, ps = queries(P["small_index"], rs, p)
        if pl or ps:
            blocked.append(f"{key} index matrices: {pl + ps}")
            continue
        out["pairs"][key] = dict(exactness_checks=dict(t, differences=diff), lifecycle=dict(large=lc, small=lcs),
                                 sizes=dict(large=sizes({"load": rl["load"]}, p, conv), small=sizes({"load": rs["load"]}, p, small_conv, small_dict)),
                                 content_queries=dict(large=ql, small=qs))
    if blocked:
        ws.write(dict(base, status="blocked", problems=blocked), dst)
        raise SystemExit(f"blocked: {blocked}")
    rng = ws.rng
    for key, o in out["pairs"].items():
        L, Sm = o["content_queries"]["large"], o["content_queries"]["small"]
        o["ranges"] = dict(
            warm_median_large=rng(L[k]["warm_ms"]["median"] for k in ws.INDEXES), warm_median_small=rng(Sm[k]["warm_ms"]["median"] for k in ws.INDEXES),
            cold_median_large=rng(L[k]["cold_ms"]["median"] for k in ws.INDEXES), cold_median_small=rng(Sm[k]["cold_ms"]["median"] for k in ws.INDEXES),
            wide_faster_warm_large=rng(L[k]["warm_ms"]["wide_faster"] for k in ws.INDEXES), wide_faster_cold_large=rng(L[k]["cold_ms"]["wide_faster"] for k in ws.INDEXES),
            wide_faster_warm_small=rng(Sm[k]["warm_ms"]["wide_faster"] for k in ws.INDEXES), wide_faster_cold_small=rng(Sm[k]["cold_ms"]["wide_faster"] for k in ws.INDEXES),
            lifecycle_speedup_large=rng(o["lifecycle"]["large"][q]["speedup"] for q in ws.LIFE),
            lifecycle_speedup_small=rng(o["lifecycle"]["small"][q]["speedup"] for q in ws.LIFE),
            lifecycle_speedup_by_query={q: dict(large=o["lifecycle"]["large"][q]["speedup"], small=o["lifecycle"]["small"][q]["speedup"]) for q in ws.LIFE},
            scale_calls=o["sizes"]["large"]["calls"] / o["sizes"]["small"]["calls"],
            scale_request_bytes=o["sizes"]["large"]["request_bytes"] / o["sizes"]["small"]["request_bytes"],
            repetition_large=o["sizes"]["large"]["request_bytes"] / o["sizes"]["large"]["dictionary_bytes"],
            repetition_small=o["sizes"]["small"]["request_bytes"] / o["sizes"]["small"]["dictionary_bytes"],
            disk_ratio_large=o["sizes"]["large"]["wide_bytes_on_disk"] / o["sizes"]["large"]["fcx_bytes_on_disk"],
            disk_ratio_small=o["sizes"]["small"]["wide_bytes_on_disk"] / o["sizes"]["small"]["fcx_bytes_on_disk"])
    out["status"] = "ok"
    ws.write(out, dst)
    print(json.dumps({k: v["ranges"] for k, v in out["pairs"].items()}, indent=1))


if __name__ == "__main__":
    main()
