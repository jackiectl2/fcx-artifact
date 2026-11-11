"""E-CHAT summary (refine-logs/E-CHAT-PROTOCOL.md, rule 6); reviewed before its output is used.

Inputs (copied from the cluster without change), for each sample s in {1, 2, 3} and manager m in MANAGERS:
  results/chat/chat_prep_stats.json                    sample drawer counts (fcx/chat_prep.py; one file for the three samples)
  results/chat/cap_chat_s<s>/capture_stats.json        capture counts (fcx/capture.py)
  results/chat/cap_chat_s<s>/convert_stats.json        ingestion statistics (fcx/ingest_run.py, delta parsing, 64 interleaved traces)
  results/ing_chat_v1/cap_chat_s<s>__<m>.json          ingestion baselines and ablations (fcx/ingest_ablate.py, same code as E-ING)
  results/chat/cap_chat_s<s>__checks.json              bench.py checks run (all four managers)
  results/chat/cap_chat_s<s>__index_<m>.json           bench.py index run (one manager)
Rules (the E-WIKI rules of figures/wiki_summary.py, applied per (sample, manager); its helpers are reused unchanged where they
are generic: table2, index_matrix, write, mmm, rng):
 - presence and completion: every input exists, and each bench run recorded every planned stage; else "incomplete", no number;
 - identity and parameters, checked exactly: drawer (min turns 10, seed 1, 3,000 conversations per sample); capture (source
   chat_pool_s<s>.parquet, format swegym, 3,000 traces, the four managers, calls = the drawer's calls); convert_stats (source
   cap_chat_s<s>, traces, spans, interleave 64, the four managers); ingest_ablate (validate of figures/ing_ext_summary.py with
   identity cap_chat_s<s> / m; requests = spans; distinct messages and dictionary bytes = convert_stats); bench runs (data
   cap_chat_s<s>, workload chat, policies, 16 threads, 2 cold / 3 warm, planned stages, checkpoint 32, time limits 30 s, the
   paper's server settings and ClickHouse version; index runs also their seed s and the six index kinds; embedded convert
   statistics equal convert_stats.json; loaded spans and traces equal its counts);
 - exactness: table2 on each checks run (all four managers), no difference, no problem; every exact ingest_ablate variant
   reproduces every request;
 - content queries: fcx/index_report.summarize per index run after index_matrix (30 held-out + 12 calibration substrings, six
   kinds, one W and one F+I.expand record per predicate, equal, fully timed); the bootstrap interval is dropped;
 - checks matrices (review round 3): per manager, ctx_any covers exactly the 15 evaluation sets (9 selectivity sets, the 6
   chat needles) under the filters none, trace and repo, each with exactly the plans fcx/query.py ctx_any generates (9 always,
   A.bloom iff the set has at most 20,000 messages, five more iff a filter applies), n_matched equal to the run's matched set; aggregates
   exactly the 11 cases with their plan sets; reconstruction exactly 11 (n, seed) targets x plans A, D, I, N; bytes exactly the
   9 named checks over every call; the 4 store checks; lifecycle exactly 8 records; no record of another policy;
 - rows, repetition, lifecycle per (sample, manager): transitions per call vs array elements per call; repetition from the
   manager's own ablation dictionary (ingest_ablate interns that manager's stream alone): logged request bytes over its
   dictionary bytes, array elements over its distinct messages; lifecycle L1-L4 warm medians on intervals and the normalized
   table, speedup N / I, exactly one equal I and N record per query and manager;
 - the arrival seed of ingest_run.py is not recorded in convert_stats; results/chat/RUN-MANIFEST.json records the capture job,
   the sha256 of cluster/chat_capture.sbatch and the md5 of the code the job printed; both must equal the files in this
   repository, and the sbatch's ingest_run.py line must pass --delta 1 --interleave 64 --seed 1;
 - drawer arithmetic: conversations = eligible + all exclusions, the turn histogram sums to eligible with keys >= 10, and per
   sample messages = 2 x calls; rows per call are recomputed from opens + closes, array elements and spans;
 - ranges: every number the paper may quote is a min-max range over the three samples, per manager and over all managers.
Any problem, including an unexpected schema error, gives status "blocked" with no numbers.
usage: .venv/bin/python figures/chat_summary.py [--out results/chat_summary.json]
"""
import argparse, hashlib, json, re, statistics, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "results"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "fcx"))
import ing_ext_summary as ing                                    # noqa: E402
import index_report                                              # noqa: E402
import wiki_summary as ws                                        # noqa: E402

SEEDS = (1, 2, 3)
MANAGERS = ["prefix", "autogen_buf32", "autogen_ht", "lc_trim32k"]
INDEXES, EVAL, CALIB = ws.INDEXES, ws.EVAL, ws.CALIB
CHECK_STAGES, INDEX_STAGES = ws.CHECK_STAGES, ws.INDEX_STAGES
PER_SAMPLE, MIN_TURNS, THREADS, COLD, WARM = 3000, 10, ws.THREADS, ws.COLD, ws.WARM
CORPUS, ELIGIBLE = 837989, 24498                                 # WildChat-1M at commit 7d6490e4 and its turn >= 10 pool (pilot, protocol)
SOURCES = [f"train-{k:05d}-of-00014.parquet" for k in range(14)]
NEEDLES = ["h:code", "h:python", "h:write", "h:please", "h:continue", "h:story"]          # bench.py CHAT_NEEDLES, in order
SETS = sorted([f"r{r}-s{k}" for r in ("0.1", "1", "10") for k in (1, 2, 3)] + NEEDLES)
CORE_ANY = {"A.exists", "A.textidx", "A.unnest", "B.or", "D.replay", "I.expand", "I.sweep", "N.fwd", "N.inv"}
FILTER_ANY = {"B.and-or", "D.pruned", "I.join", "I.pruned", "I.pruned-sweep"}     # fcx/query.py ctx_any: only with a trace-level filter
BLOOM_MAX = 20000                                                 # ctx_any: A.bloom only when the matched set has at most this many messages
MANIFEST = dict(capture_job="65700588 (array 0-2, one element per sample)", sbatch="cluster/chat_capture.sbatch",
                code_md5_files=["fcx/capture.py", "fcx/ingest.py", "fcx/ingest_run.py", "fcx/convert.py"],
                ingest_arguments="--delta 1 --interleave 64 --seed 1")
FILTERS = {"none", "trace", "repo"}
AGG_CASES = {**{f"ctx_count/{x}": {"A (reference)", "D.replay", "I.expand", "N.inv"} for x in ("r1-s1", NEEDLES[0])},
             **{f"ctx_sum[{v}]/{x}": {"A (reference)", "I.expand", "N.inv"} for v in ("n_chars", "n_bytes", "n_err") for x in ("r1-s1", NEEDLES[0])},
             **{f"ctx_sum[{v}]/all": {"A (reference)", "D.sweep", "I.expand", "N.fwd"} for v in ("n_chars", "n_bytes", "n_err")}}
POINTS = {(1000, 7)} | {(1, k) for k in range(100, 110)}
BYTES_CHECKS = {f"ctx_any/{x}" for x in NEEDLES} | {"ctx_sum[n_bytes]/all", "ctx_sum[n_err]/all", f"ctx_count/{NEEDLES[0]} (1% of spans)"}
STORE_CHECKS = ["a_{}_spans_differ_from_requests", "arrays_{}_rows_not_one_per_span", "i_{}_rows_minus_opens", "n_{}_rows_minus_array_elems"]
pos = lambda v: num(v) and v > 0
num, nat = ws.num, ws.nat


def paths(s):
    c = RES / "chat"
    p = dict(prep=c / "chat_prep_stats.json", capture=c / f"cap_chat_s{s}" / "capture_stats.json", convert=c / f"cap_chat_s{s}" / "convert_stats.json",
             checks=c / f"cap_chat_s{s}__checks.json")
    for m in MANAGERS:
        p[f"ablate/{m}"] = RES / "ing_chat_v1" / f"cap_chat_s{s}__{m}.json"
        p[f"index/{m}"] = c / f"cap_chat_s{s}__index_{m}.json"
    return p


def lifecycle(r, policy):
    """wiki_summary.lifecycle for one policy of a multi-policy checks run."""
    by, probs = {}, []
    for x in r["life"]:
        if x.get("policy") != policy:
            continue
        q, plan = ws.LIFE_QUERIES.get(x.get("query")), x.get("plan")
        warm, cold, rows = x.get("warm_ms"), x.get("cold_ms"), x.get("read_rows")
        if (q is None or plan not in ("I", "N") or (q, plan) in by or "error" in x or x.get("equal") is not True
                or not (isinstance(warm, list) and len(warm) == WARM and all(num(v) and v > 0 for v in warm))
                or not (isinstance(cold, list) and len(cold) == COLD and all(num(v) and v > 0 for v in cold))
                or not (nat(rows) and rows >= 0)):
            probs.append(f"lifecycle record unusable: {policy} {x.get('query')!r} {plan}")
            continue
        by[(q, plan)] = (statistics.median(warm), rows)
    if set(by) != {(q, p) for q in ws.LIFE for p in ("I", "N")}:
        probs.append(f"lifecycle matrix incomplete for {policy}: {sorted(by)}")
    if probs:
        return None, probs
    return {q: dict(intervals_ms=by[(q, "I")][0], normalized_ms=by[(q, "N")][0], speedup=by[(q, "N")][0] / by[(q, "I")][0],
                    intervals_read_rows=by[(q, "I")][1], normalized_read_rows=by[(q, "N")][1]) for q in ws.LIFE}, []


def checks_matrix(r, calls):
    """Problems with the completeness of a checks run's record matrices (rule: checks matrices)."""
    probs = []
    pol = lambda x: x.get("policy")
    for part in ("any", "agg", "point", "life"):
        if any(pol(x) not in MANAGERS for x in r.get(part, [])):
            probs.append(f"{part}: a record of another policy")
    for m in MANAGERS:
        g, nm = {}, {}
        for x in r.get("any", []):
            if pol(x) == m:
                g.setdefault((x.get("matched"), x.get("filter")), []).append(x.get("plan"))
                nm.setdefault(x.get("matched"), set()).add(x.get("n_matched"))
        ok = set(g) == {(n, f) for n in SETS for f in FILTERS}
        for (n, f), v in g.items():                               # exactly the plans ctx_any generates for this set and filter
            k = r.get("matched", {}).get(n)
            valid_k = nat(k) and k >= 0
            want = CORE_ANY | ({"A.bloom"} if valid_k and k <= BLOOM_MAX else set()) | (FILTER_ANY if f != "none" else set())
            ok &= valid_k and nm.get(n) == {k} and sorted(v) == sorted(want)
        if not ok:
            probs.append(f"{m}: ctx_any matrix not exact")
        ga = {}
        for x in r.get("agg", []):
            if pol(x) == m:
                ga.setdefault(x.get("case"), []).append(x.get("plan"))
        if set(ga) != set(AGG_CASES) or any(sorted(v) != sorted(AGG_CASES[k]) for k, v in ga.items()):
            probs.append(f"{m}: aggregate matrix incomplete or duplicated")
        pts = sorted((x.get("n"), x.get("seed"), x.get("plan")) for x in r.get("point", []) if pol(x) == m)
        if pts != sorted((n, sd, p) for n, sd in POINTS for p in "ADIN"):
            probs.append(f"{m}: reconstruction matrix incomplete or duplicated")
        if sum(pol(x) == m for x in r.get("life", [])) != 8:
            probs.append(f"{m}: lifecycle records {sum(pol(x) == m for x in r.get('life', []))}, expected 8")
        b = r.get("bytes", {}).get(m, {})
        if set(b.get("checks", {})) != BYTES_CHECKS or b.get("spans") != calls:
            probs.append(f"{m}: byte checks incomplete")
    if set(r.get("bytes", {})) != set(MANAGERS):
        probs.append(f"bytes policies {sorted(r.get('bytes', {}))}")
    if set(r.get("checks", {})) != {c.format(m) for m in MANAGERS for c in STORE_CHECKS}:
        probs.append(f"store checks {sorted(r.get('checks', {}))}")
    return probs


def index_values(r, m):
    """Timed values finite and positive; selectivity maps cover the held-out predicates with nonnegative integers."""
    probs = []
    for x in r.get("records", []):
        if x.get("policy") != m:
            probs.append(f"index record of policy {x.get('policy')!r}")
        if x.get("plan") in ("W", index_report.FACT) and not all(pos(v) for v in x.get("warm_ms", []) + x.get("cold_ms", [])):
            probs.append(f"{x.get('index')} {x.get('pred')} {x.get('plan')}: timing not finite and positive")
    for k in ("matched_messages", "truth_spans"):
        mp = r.get(k, {})
        if not (EVAL <= set(mp) and all(nat(mp[p]) and mp[p] >= 0 for p in EVAL)):
            probs.append(f"{k} does not cover the held-out predicates with nonnegative integers")
    return probs[:10]


def run_identity(r, s, stages, conv, policies, index):
    m, probs = r.get("meta", {}), []
    want = dict(data=f"cap_chat_s{s}", workload="chat", policies=policies, wide_policies=policies, threads=THREADS, cold=COLD, warm=WARM, stages=stages,
                ckpt=ws.CKPT, timeout_s=ws.TIMEOUT_S, check_timeout_s=ws.TIMEOUT_S, settings=ws.SETTINGS, clickhouse=ws.CLICKHOUSE)
    if index:
        want.update(index_policies=policies, indexes=INDEXES, seed=s)
    probs += [f"meta {k}={m.get(k)!r}, expected {v!r}" for k, v in want.items() if m.get(k) != v]
    if r.get("convert") != conv:
        probs.append("embedded convert statistics differ from convert_stats.json")
    if r.get("spans") != conv.get("spans") or r.get("traces") != conv.get("traces"):
        probs.append(f"loaded spans/traces {r.get('spans')}/{r.get('traces')} differ from convert_stats {conv.get('spans')}/{conv.get('traces')}")
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RES / "chat_summary.json"))
    dst = Path(ap.parse_args().out)
    base = dict(protocol="refine-logs/E-CHAT-PROTOCOL.md")
    try:
        summarize(dst, base)
    except SystemExit:
        raise
    except Exception as e:                                        # an unexpected schema error never leaves an older result in place
        ws.write(dict(base, status="blocked", problems=[f"unexpected {type(e).__name__}: {e}"]), dst)
        raise SystemExit(f"blocked: unexpected {type(e).__name__}: {e}")


def summarize(dst, base):
    D, missing, unfinished = {}, [], []
    for s in SEEDS:
        for k, p in paths(s).items():
            try:
                D[(s, k)] = json.load(open(p))
            except (OSError, ValueError) as e:
                missing.append(f"{p.relative_to(ROOT)}: {type(e).__name__}")
        for k, stages in [("checks", CHECK_STAGES)] + [(f"index/{m}", INDEX_STAGES) for m in MANAGERS]:
            r = D.get((s, k))
            if r is not None and not (isinstance(r, dict) and all(f"{st}_s" in r.get("meta", {}) for st in stages)):
                unfinished.append(f"cap_chat_s{s} {k}: not every planned stage recorded")
    if missing or unfinished:
        ws.write(dict(base, status="incomplete", missing=missing, unfinished=unfinished), dst)
        raise SystemExit(f"incomplete: {missing + unfinished}; see {dst}")

    blocked, out = [], dict(base, samples={})
    prep = D[(1, "prep")]
    if not (prep.get("min_turns") == MIN_TURNS and prep.get("seed") == 1 and list(prep.get("samples", {})) == ["s1", "s2", "s3"]
            and all(v.get("conversations") == PER_SAMPLE for v in prep["samples"].values()) and prep.get("conversations") == CORPUS
            and prep.get("eligible") == ELIGIBLE and prep.get("src") == SOURCES):
        blocked.append("drawer identity or parameters")
    ex, et = prep.get("excluded", {}), prep.get("eligible_turns", {})
    if not (isinstance(ex, dict) and all(nat(v) and v >= 0 for v in ex.values()) and prep.get("conversations") == prep.get("eligible", -1) + sum(ex.values())
            and isinstance(et, dict) and all(k.isdigit() and int(k) >= MIN_TURNS and nat(v) and v > 0 for k, v in et.items())
            and sum(et.values()) == prep.get("eligible")
            and all(nat(v.get("calls")) and v["calls"] >= MIN_TURNS * PER_SAMPLE and v.get("messages") == 2 * v["calls"] and nat(v.get("content_chars"))
                    and v["content_chars"] > 0 for v in prep.get("samples", {}).values())):
        blocked.append("drawer counts inconsistent")
    man = json.load(open(RES / "chat" / "RUN-MANIFEST.json"))
    sb = (ROOT / MANIFEST["sbatch"]).read_bytes()
    code = hashlib.md5(b"".join((ROOT / f).read_bytes() for f in MANIFEST["code_md5_files"])).hexdigest()
    if not (all(man.get(k) == v for k, v in MANIFEST.items()) and set(man) == set(MANIFEST) | {"purpose", "sbatch_sha256", "code_md5_printed_by_job"}
            and hashlib.sha256(sb).hexdigest() == man.get("sbatch_sha256") and code == man.get("code_md5_printed_by_job")
            and re.search(rb"ingest_run\.py [^\n]*" + re.escape(MANIFEST["ingest_arguments"].encode()) + rb"\n", sb)):
        blocked.append("run manifest does not match the repository's capture script and code")
    out["drawer"] = dict(conversations=prep.get("conversations"), excluded=prep.get("excluded"), eligible=prep.get("eligible"),
                         eligible_turns=prep.get("eligible_turns"), samples=prep.get("samples"))
    for s in SEEDS:
        name = f"cap_chat_s{s}"
        cap, cv, ck = D[(s, "capture")], D[(s, "convert")], D[(s, "checks")]
        o = out["samples"][name] = dict(managers={})
        ps = prep.get("samples", {}).get(f"s{s}", {})
        if not (Path(str(cap.get("src", ""))).name == f"chat_pool_s{s}.parquet" and cap.get("format") == "swegym" and cap.get("traces") == PER_SAMPLE
                and cap.get("corpus_trajectories") == PER_SAMPLE and cap.get("sample") == 0 and cap.get("seed") == s
                and cap.get("spans") == ps.get("calls") and list(cap.get("managers", {})) == MANAGERS
                and all(v.get("requests") == ps.get("calls") for v in cap.get("managers", {}).values())):
            blocked.append(f"{name}: capture identity, counts or managers")
        if not (Path(str(cv.get("src", ""))).name == name and cv.get("traces") == PER_SAMPLE and cv.get("spans") == ps.get("calls")
                and cv.get("interleave") == 64 and cv.get("budget") == 64 and cv.get("ckpt") == 32 and list(cv.get("policies", {})) == MANAGERS
                and all(p.get("ingest", {}).get("delta") == 1 and p.get("ingest", {}).get("interleave") == 64 for p in cv["policies"].values())):
            blocked.append(f"{name}: convert_stats identity")
        db, dm = cv.get("dictionary_bytes"), cv.get("distinct_messages")
        if not (nat(db) and db > 0 and nat(dm) and dm > 0):
            blocked.append(f"{name}: dictionary size missing")
            continue
        blocked += [f"{name} checks: {x}" for x in run_identity(ck, s, CHECK_STAGES, cv, MANAGERS, index=False)]
        blocked += [f"{name} checks: {x}" for x in checks_matrix(ck, ps.get("calls"))]
        t, diff, probs = ws.table2(ck)
        if diff or probs:
            blocked.append(f"{name} checks: {diff} differences, {probs}")
        o["exactness_checks"] = dict(t, differences=diff)
        for m in MANAGERS:
            om = o["managers"][m] = {}
            ab, ix, pw = D[(s, f"ablate/{m}")], D[(s, f"index/{m}")], cv["policies"][m]
            adb, adm = ab.get("dictionary_bytes"), ab.get("distinct_messages")
            if not (pw.get("ingest", {}).get("requests") == ps.get("calls") and ab.get("requests") == ps.get("calls") and nat(adb) and adb > 0
                    and nat(adm) and adm > 0 and all(isinstance(v, dict) and v.get("requests") == ps.get("calls") and v.get("logical_bytes") == pw.get("wide_chars")
                                                     for v in ab.get("variants", {}).values())
                    and all(nat(pw.get(k)) and pw[k] >= 0 for k in ("opens", "closes", "array_elems", "wide_chars"))):
                blocked.append(f"{name}/{m}: ingested store and ablation disagree on requests or request bytes")
                continue
            om["repetition"] = dict(logged_over_distinct_bytes=pw["wide_chars"] / adb, array_elems_over_distinct=pw["array_elems"] / adm,
                                    distinct_messages=adm, dictionary_bytes=adb, logical_bytes=pw["wide_chars"])
            blocked += [f"{name} index {m}: {x}" for x in run_identity(ix, s, INDEX_STAGES, cv, [m], index=True)]
            err = ing.validate(ab, name, m)
            if err:
                blocked.append(f"{name}/{m} ingest_ablate data errors {err}")
            else:
                bad = {k: ab["variants"][k]["wrong"] for k in ing.EXACT if ab["variants"][k]["wrong"]}
                if bad:
                    blocked.append(f"{name}/{m} ingest_ablate exact variants wrong {bad}")
                om["ingestion"] = dict(requests=ab["requests"], exact_variants_zero_wrong=not bad, ratios=ing.ratios(ab),
                                       set_wrong_equals_dup=ab["variants"]["set_diff"]["wrong"] == ab["slice_requests"]["dup"],
                                       timing_kind="single run, fixed variant order, derivation only")
            tr, nr = (pw["opens"] + pw["closes"]) / ps["calls"], pw["array_elems"] / ps["calls"]
            if abs(tr - pw.get("transitions_per_span", -9)) > 0.0006 or abs(nr - pw.get("mean_context_len", -9)) > 0.006:
                blocked.append(f"{name}/{m}: rows per call disagree with opens, closes and array elements")
            om["rows_per_call"] = dict(transitions=tr, normalized=nr, normalized_over_transitions=nr / tr if tr else None)
            lc, lprobs = lifecycle(ck, m)
            if lprobs:
                blocked.append(f"{name} checks: {lprobs}")
            om["lifecycle"] = lc
            r = ix.get("index", {}).get(m, {})
            mprobs = ws.index_matrix(r) + index_values(r, m)
            if ix.get("mismatch"):
                mprobs.append(f"mismatch {ix['mismatch']}")
            if mprobs:
                blocked.append(f"{name}/{m} index matrix: {mprobs}")
                continue
            _, rows = index_report.summarize(str(paths(s)[f"index/{m}"]))
            if sorted(x["index"] for x in rows) != sorted(INDEXES) or not all(x["complete"] for x in rows):
                blocked.append(f"{name}/{m}: index_report rows incomplete")
                continue
            om["content_queries"] = {x["index"]: dict({k: {kk: vv for kk, vv in x[k].items() if kk != "ci"} for k in ("warm_ms", "cold_ms")},
                                                      wide_ms=x["wide_ms"], fcx_ms=x["fact_ms"], build=x["variant"]) for x in rows}
            om["selectivity"] = {L: dict(matched_messages=ws.mmm(r["matched_messages"][k] for k in EVAL if k.startswith(f"eval:sub{L}:")),
                                         matching_calls=ws.mmm(r["truth_spans"][k] for k in EVAL if k.startswith(f"eval:sub{L}:"))) for L in (8, 16, 32)}
    if blocked:
        ws.write(dict(base, status="blocked", problems=blocked), dst)
        raise SystemExit(f"blocked: {blocked}")
    S = [out["samples"][f"cap_chat_s{s}"] for s in SEEDS]
    M = lambda m: [x["managers"][m] for x in S]
    A = [y for m in MANAGERS for y in M(m)]
    cq = lambda xs, k, mode: ws.rng(y["content_queries"][k][mode]["median"] for y in xs)
    out["ranges"] = dict(
        calls=ws.rng(x["managers"]["prefix"]["ingestion"]["requests"] for x in S),
        exactness_totals={k: sum(x["exactness_checks"][k] for x in S) for k in ("calls", "any", "agg", "point", "bytes", "life")},
        logged_over_distinct_bytes_all=ws.rng(y["repetition"]["logged_over_distinct_bytes"] for y in A),
        by_manager={m: dict(
            logged_over_distinct_bytes=ws.rng(y["repetition"]["logged_over_distinct_bytes"] for y in M(m)),
            array_elems_over_distinct=ws.rng(y["repetition"]["array_elems_over_distinct"] for y in M(m)),
            lifecycle_ms_by_query={q: dict(intervals=ws.rng(y["lifecycle"][q]["intervals_ms"] for y in M(m)),
                                           normalized=ws.rng(y["lifecycle"][q]["normalized_ms"] for y in M(m))) for q in ws.LIFE},
            content_min_max_by_index={k: {mode: [min(y["content_queries"][k][mode]["min"] for y in M(m)), max(y["content_queries"][k][mode]["max"] for y in M(m))]
                                          for mode in ("warm_ms", "cold_ms")} for k in INDEXES},
            ingestion_ratios={k: ws.rng(y["ingestion"]["ratios"][k] for y in M(m)) for k in M(m)[0]["ingestion"]["ratios"]},
            rows_transitions_per_call=ws.rng(y["rows_per_call"]["transitions"] for y in M(m)),
            rows_normalized_per_call=ws.rng(y["rows_per_call"]["normalized"] for y in M(m)),
            rows_normalized_over_transitions=ws.rng(y["rows_per_call"]["normalized_over_transitions"] for y in M(m)),
            lifecycle_speedup_by_query={q: ws.rng(y["lifecycle"][q]["speedup"] for y in M(m)) for q in ws.LIFE},
            content_warm_median_by_index={k: cq(M(m), k, "warm_ms") for k in INDEXES},
            content_cold_median_by_index={k: cq(M(m), k, "cold_ms") for k in INDEXES},
            content_warm_median_all_indexes=ws.rng(y["content_queries"][k]["warm_ms"]["median"] for y in M(m) for k in INDEXES),
            content_cold_median_all_indexes=ws.rng(y["content_queries"][k]["cold_ms"]["median"] for y in M(m) for k in INDEXES),
            content_wide_faster_warm_by_index={k: ws.rng(y["content_queries"][k]["warm_ms"]["wide_faster"] for y in M(m)) for k in INDEXES},
            content_wide_faster_cold_by_index={k: ws.rng(y["content_queries"][k]["cold_ms"]["wide_faster"] for y in M(m)) for k in INDEXES})
            for m in MANAGERS},
        content_warm_median_all=ws.rng(y["content_queries"][k]["warm_ms"]["median"] for y in A for k in INDEXES),
        content_cold_median_all=ws.rng(y["content_queries"][k]["cold_ms"]["median"] for y in A for k in INDEXES),
        content_warm_median_managed=ws.rng(y["content_queries"][k]["warm_ms"]["median"] for m in MANAGERS[1:] for y in M(m) for k in INDEXES),
        content_wide_faster_warm_max=max(y["content_queries"][k]["warm_ms"]["wide_faster"] for y in A for k in INDEXES),
        content_wide_faster_cold_max=max(y["content_queries"][k]["cold_ms"]["wide_faster"] for y in A for k in INDEXES),
        lifecycle_speedup_all=ws.rng(y["lifecycle"][q]["speedup"] for y in A for q in ws.LIFE),
        lifecycle_speedup_by_query_all={q: ws.rng(y["lifecycle"][q]["speedup"] for y in A) for q in ws.LIFE},
        lifecycle_ms_by_query_all={q: dict(intervals=ws.rng(y["lifecycle"][q]["intervals_ms"] for y in A),
                                           normalized=ws.rng(y["lifecycle"][q]["normalized_ms"] for y in A)) for q in ws.LIFE},
        content_min_max_by_index_all={k: {mode: [min(y["content_queries"][k][mode]["min"] for y in A), max(y["content_queries"][k][mode]["max"] for y in A)]
                                          for mode in ("warm_ms", "cold_ms")} for k in INDEXES},
        content_warm_median_by_index_all={k: cq(A, k, "warm_ms") for k in INDEXES},
        content_cold_median_by_index_all={k: cq(A, k, "cold_ms") for k in INDEXES},
        ingestion_ratios_all={k: ws.rng(y["ingestion"]["ratios"][k] for y in A) for k in A[0]["ingestion"]["ratios"]})
    out["status"] = "ok"
    ws.write(out, dst)
    print(json.dumps({k: v for k, v in out.items() if k != "samples"}, indent=1))


if __name__ == "__main__":
    main()
