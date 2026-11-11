"""E-ING extension aggregator, applying checks 1-5 of refine-logs/E-ING-EXTENSION-PROTOCOL.md (frozen and reviewed before use).

Inputs (85 result files): results/ing_s1_v1, results/ing_s1_rep2, results/ing_s1_rep3 (the 17 seed-1 streams of
cluster/tasks/s1.txt, repetitions 1-3) and results/ing_s23_v1 (the 34 seed-2/3 streams of cluster/tasks/s23.txt), plus the
attempt ledger results/ing_ext_attempts.json (figures/ing_ext_ledger.py, from Slurm accounting).
Checks:
 1. manifest: 17 + 34 unique, disjoint streams; every manager in exactly one group; all 85 files present and readable and no
    other file (a partial result) in the result directories; otherwise the output has status "incomplete", every claim is
    blocked, and no number is reported;
 2. determinism: every non-timing field (all keys except cpu_s, wall_s, total_wall_s, peak_rss_mb, at any depth of dicts and
    lists) of repetitions 2 and 3 equals repetition 1 for every seed-1 stream; a difference blocks the claims that use the
    differing variant (a difference outside "variants" blocks every claim);
 3. timing: CPU ratios are three fixed-order descriptive measurements per seed-1 stream (all three, median, min, max) and one
    single run per seed-2/3 stream; group ranges are derived from those per-stream values;
 4. attempts: each file has one ledger entry per attempt, at most two; only the last may be completed, and an earlier attempt
    must be an infrastructure or incomplete failure; otherwise the output is "incomplete";
 5. no dropping: every stream and repetition enters; a correctness failure (an exact variant with a wrong request, in any
    repetition) blocks the claims that depend on it instead of removing the stream; a data error in any file (identity, fixed
    parameters, schema, types, denominators) blocks every claim and no number is computed.
Deterministic claims use the 51 streams (seed 1 from repetition 1). Ratios use the definitions of figures/fig7_ingest_baselines.py:
hash arrays' payload against FCX's transition plus checkpoint bytes; dense positions with their checkpoint bytes.
usage: .venv/bin/python figures/ing_ext_summary.py [--out results/ing_ext_summary.json]
"""
import argparse, json, math, os, re, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
SEED1_RUNS = ["ing_s1_v1", "ing_s1_rep2", "ing_s1_rep3"]
S23_RUN = "ing_s23_v1"
EXACT = ["fcx_b0", "fcx_b2", "fcx_b8", "fcx_b64", "fcx_b256", "fcx_full", "nokey", "hash_array", "pos_diff"]
LOSSY = ["set_diff", "mset_lifo"]
TIMING = {"cpu_s", "wall_s", "total_wall_s", "peak_rss_mb"}
GROUPS = {"append": {"prefix"}, "trim": {"lc_trim32k"}, "autogen": {"autogen_buf32", "autogen_ht"}, "swea": {"swea_last5", "swea_window", "swea_cache2"}}
MANAGED = GROUPS["trim"] | GROUPS["autogen"] | GROUPS["swea"]
FIXED = dict(interleave=64, seed=1, ckpt=32)                     # seed: the arrival-order seed of ingest_ablate.py, the same for all
SAMPLE = re.compile(r"^cap_[a-z]+_s([0-9]+)$")
CLAIMS = {  # claim -> variants it reads
    "C1": EXACT, "C2": LOSSY, "C3": ["hash_array", "fcx_b64"], "C4": ["nokey", "fcx_b64"],
    "C5": ["fcx_full", "hash_array", "fcx_b64"], "C6": ["fcx_b0", "fcx_b2", "fcx_b8", "fcx_b64", "fcx_b256"]}
SLICES_FROM = "fcx_b64"                                          # the request slices (dup, ...) come from the replayed fcx_b64 log
COUNTS = {"requests", "records", "payload_bytes", "wrong"}       # nonnegative integers in every variant
LOG_COUNTS = {"checkpoint_bytes", "fallback_requests"}           # and in the log-based variants only
LOG_VARIANTS = {"fcx_b0", "fcx_b2", "fcx_b8", "fcx_b64", "fcx_b256", "fcx_full", "nokey"}
LEDGER_FIELDS = dict(run=str, data=str, manager=str, attempt=int, state=str, exit_code=str)


def count(x):
    return isinstance(x, int) and not isinstance(x, bool) and x >= 0


def measure(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x >= 0


def manifest(task_file, errors):
    rows = []
    try:
        lines = open(ROOT / "cluster" / "tasks" / task_file).read().splitlines()
    except OSError as e:
        errors.append(f"{task_file}: {e}")
        return rows
    for line in lines:
        if not line.strip():
            continue
        f = line.split()
        m = SAMPLE.match(f[0]) if len(f) == 2 else None
        if not m:
            errors.append(f"{task_file}: bad line {line!r}")
            continue
        rows.append((f[0], f[1], int(m.group(1))))
    if len(set(rows)) != len(rows):
        errors.append(f"{task_file}: duplicate streams")
    return rows


def strip_timing(x):
    if isinstance(x, dict):
        return {k: strip_timing(v) for k, v in x.items() if k not in TIMING}
    if isinstance(x, list):
        return [strip_timing(v) for v in x]
    return x


def diff_paths(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                yield f"{path}.{k}"
            else:
                yield from diff_paths(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            yield path + "[len]"
        for i, (x, y) in enumerate(zip(a, b)):
            yield from diff_paths(x, y, f"{path}[{i}]")
    elif a != b or type(a) is not type(b):
        yield path


def validate(d, data, manager):
    """Errors in one result file: identity, fixed parameters, variant set, field types, ranges, positive denominators."""
    if not isinstance(d, dict):
        return ["not a JSON object"]
    err = []
    if Path(str(d.get("data", ""))).name != data or d.get("manager") != manager:
        err.append(f"identity {d.get('data')!r} {d.get('manager')!r}")
    err += [f"{k}={d.get(k)!r}" for k, v in FIXED.items() if d.get(k) != v or type(d.get(k)) is not int]
    n = d.get("requests")
    if not (count(n) and n > 0):
        return err + [f"requests={n!r}"]
    sl = d.get("slice_requests")
    if not (isinstance(sl, dict) and count(sl.get("dup")) and sl["dup"] <= n):
        err.append(f"slice_requests={sl!r}")
    v = d.get("variants")
    if not isinstance(v, dict) or set(v) != set(EXACT + LOSSY):
        return err + [f"variants {sorted(v) if isinstance(v, dict) else v!r}"]
    for name, x in v.items():
        if not isinstance(x, dict):
            err.append(f"{name}: not an object")
            continue
        log = name in LOG_VARIANTS
        bad = [k for k in COUNTS | (LOG_COUNTS if log else set()) if not count(x.get(k))] + ([] if measure(x.get("cpu_s")) else ["cpu_s"])
        extra = [k for k in LOG_COUNTS if not log and k in x]
        if bad or extra:
            err.append(f"{name}: invalid {sorted(bad)}, unexpected {sorted(extra)}")
        elif x["requests"] != n or x["wrong"] > n or x.get("fallback_requests", 0) > n:
            err.append(f"{name}: requests/wrong/fallback out of range")
        elif x.get("must_be_exact") is not (name in EXACT):
            err.append(f"{name}: must_be_exact={x.get('must_be_exact')!r}")
    if not err:
        f = v["fcx_b64"]
        if not (f["records"] > 0 and f["payload_bytes"] + f["checkpoint_bytes"] > 0 and f["cpu_s"] > 0):
            err.append("fcx_b64: zero denominator")
    return err


def ratios(d):
    v, n = d["variants"], d["requests"]
    f = v["fcx_b64"]
    fb = f["payload_bytes"] + f["checkpoint_bytes"]
    return dict(
        dup=d["slice_requests"]["dup"] / n, set_wrong=v["set_diff"]["wrong"] / n, mset_wrong=v["mset_lifo"]["wrong"] / n,
        hash_bytes=v["hash_array"]["payload_bytes"] / fb, nokey_bytes=(v["nokey"]["payload_bytes"] + v["nokey"]["checkpoint_bytes"]) / fb,
        nokey_rec=v["nokey"]["records"] / f["records"],
        b0_rec=v["fcx_b0"]["records"] / f["records"], b2_rec=v["fcx_b2"]["records"] / f["records"], b8_rec=v["fcx_b8"]["records"] / f["records"],
        b256_dev=abs(v["fcx_b256"]["records"] / f["records"] - 1),
        b0_fb=v["fcx_b0"]["fallback_requests"] / n, b64_fb=f["fallback_requests"] / n,
        full_cpu=v["fcx_full"]["cpu_s"] / f["cpu_s"], hash_cpu=v["hash_array"]["cpu_s"] / f["cpu_s"])


def rng(vals):
    vals = list(vals)
    if not vals:
        raise ValueError("empty group")                          # every group is non-empty by the manifest check
    return [round(min(vals), 4), round(max(vals), 4)]


def check_ledger(required):
    try:
        entries = json.load(open(RES / "ing_ext_attempts.json")).get("entries")
    except (OSError, ValueError, AttributeError) as e:
        return [f"attempt ledger unreadable: {e}"]
    if not isinstance(entries, list):
        return ["attempt ledger: entries is not a list"]
    by, probs = {}, []
    for i, x in enumerate(entries):
        if not (isinstance(x, dict) and all(isinstance(x.get(k), t) and not isinstance(x.get(k), bool) for k, t in LEDGER_FIELDS.items())
                and "failure_class" in x and x["failure_class"] in (None, "infrastructure", "incomplete", "failed")):
            probs.append(f"malformed ledger entry {i}")
            continue
        by.setdefault((x["run"], x["data"], x["manager"]), []).append(x)
    probs += [f"ledger entry for a file outside the manifest: {k}" for k in by if k not in required]
    for k in sorted(required):
        atts = sorted(by.get(k, []), key=lambda x: x["attempt"])
        if not atts:
            probs.append(f"no ledger entry: {k}")
            continue
        if [x["attempt"] for x in atts] != list(range(1, len(atts) + 1)) or len(atts) > 2:
            probs.append(f"attempt numbering or more than one rerun: {k}")
        last = atts[-1]
        if not (last["state"] == "COMPLETED" and last["exit_code"] == "0:0" and last["failure_class"] is None):
            probs.append(f"last attempt not completed: {k} {last['state']} {last['exit_code']}")
        for x in atts[:-1]:
            if x["failure_class"] not in ("infrastructure", "incomplete"):
                probs.append(f"rerun after a {x['failure_class'] or 'completed'} attempt: {k}")
    return probs


def write(out, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1, allow_nan=False)
    os.replace(tmp, dst)


def blocked_all(base, status, reason, **extra):
    return dict(base, status=status, **extra, claims={c: dict(status="blocked", reasons=[reason]) for c in CLAIMS}, c5_per_stream=[])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RES / "ing_ext_summary.json"))
    dst = Path(ap.parse_args().out)
    base = dict(protocol="refine-logs/E-ING-EXTENSION-PROTOCOL.md", timing_kind="fixed-order descriptive measurements; derivation only")

    # check 1: manifest, presence, partial artifacts; check 4: attempts
    merr = []
    s1, s23 = manifest("s1.txt", merr), manifest("s23.txt", merr)
    union = set().union(*GROUPS.values())
    if not merr:
        if len(s1) != 17 or len(s23) != 34 or set(s1) & set(s23):
            merr.append(f"manifest sizes {len(s1)}/{len(s23)} or overlap")
        if {s for *_, s in s1} != {1} or {s for *_, s in s23} != {2, 3}:
            merr.append("manifest seeds")
        if sum(len(g) for g in GROUPS.values()) != len(union) or {m for _, m, _ in s1 + s23} != union:
            merr.append("manager groups do not partition the manifest managers")
    if merr:
        write(blocked_all(base, "incomplete", "manifest errors", manifest_errors=merr), dst)
        raise SystemExit(f"incomplete: manifest errors {merr}; see {dst}")
    required = {(r, d, m) for r in SEED1_RUNS for d, m, _ in s1} | {(S23_RUN, d, m) for d, m, _ in s23}
    loaded, missing = {}, []
    for k in sorted(required):
        try:
            loaded[k] = json.load(open(RES / k[0] / f"{k[1]}__{k[2]}.json"))
        except (OSError, ValueError) as e:
            missing.append(f"{k[0]}/{k[1]}__{k[2]}.json: {type(e).__name__}")
    stray = sorted(f"{r}/{p.name}" for r in SEED1_RUNS + [S23_RUN] if (RES / r).is_dir() for p in (RES / r).iterdir() if p.suffix != ".json")
    ledger = check_ledger(required)
    if missing or stray or ledger:
        write(blocked_all(base, "incomplete", "incomplete inputs", missing=missing, partial_files=stray, ledger_problems=ledger), dst)
        raise SystemExit(f"incomplete: {len(missing)} missing, {len(stray)} partial files, {len(ledger)} ledger problems; see {dst}")

    # check 5: data errors block everything; nothing else is computed from damaged files
    data_errors = {f"{k[0]}/{k[1]}__{k[2]}": e for k in sorted(loaded) if (e := validate(loaded[k], k[1], k[2]))}
    if data_errors:
        write(blocked_all(base, "blocked", "data errors", data_errors=data_errors), dst)
        raise SystemExit(f"blocked: data errors in {len(data_errors)} files; see {dst}")

    blocked = {c: [] for c in CLAIMS}
    stream_reasons = {}                                          # (data, manager) -> reasons its C5 timing is blocked

    def block(variants, why, stream):
        for c, used in CLAIMS.items():
            if variants is None or set(used) & set(variants):
                blocked[c].append(why)
        if variants is None or set(CLAIMS["C5"]) & set(variants):  # a stream's timing is blocked exactly when C5 is
            stream_reasons.setdefault(stream, []).append(why)

    # check 5: correctness of every exact variant in every repetition
    failures = []
    for k in sorted(loaded):
        bad = [x for x in EXACT if loaded[k]["variants"][x]["wrong"]]
        if bad:
            why = f"correctness failure of {bad} in {k[0]}/{k[1]}__{k[2]}"
            failures.append(why)
            block(bad, why, k[1:])
            if SLICES_FROM in bad:                               # C2's duplicate slices come from this variant's replay
                blocked["C2"].append(why)
    # check 2: determinism of seed-1 repetitions
    nondet = []
    for data, m, _ in s1:
        ref = strip_timing(loaded[(SEED1_RUNS[0], data, m)])
        for r in SEED1_RUNS[1:]:
            paths = list(diff_paths(ref, strip_timing(loaded[(r, data, m)])))
            if paths:
                nondet.append(dict(run=r, data=data, manager=m, paths=paths[:20], n_paths=len(paths)))
                local = all(p.startswith(".variants.") and p.count(".") >= 2 for p in paths)
                block({p.split(".")[2] for p in paths} if local else None, f"non-timing difference in {r}/{data}__{m}", (data, m))

    out = dict(base, status="blocked" if any(blocked.values()) else "ok", streams=51, files=85, correctness_failures=failures,
               nondeterministic=nondet,
               requests_51=sum(loaded[(SEED1_RUNS[0], d, m)]["requests"] for d, m, _ in s1) + sum(loaded[(S23_RUN, d, m)]["requests"] for d, m, _ in s23))
    S = [dict(data=d, manager=m, seed=1, reps=[loaded[(r, d, m)] for r in SEED1_RUNS]) for d, m, _ in s1] + \
        [dict(data=d, manager=m, seed=s, reps=[loaded[(S23_RUN, d, m)]]) for d, m, s in s23]
    for s in S:
        s["rr"] = [ratios(x) for x in s["reps"]]
        s["r"] = s["rr"][0]
    G = lambda *names: (lambda s: s["manager"] in set().union(*(GROUPS[n] for n in names)))
    ALL, MAN = (lambda s: True), (lambda s: s["manager"] in MANAGED)
    R = lambda pred, key: rng(s["r"][key] for s in S if pred(s))
    raw = lambda s, v, f: s["reps"][0]["variants"][v][f]
    dup = lambda s: s["reps"][0]["slice_requests"]["dup"]
    claims = {}

    def emit(c, fn):
        claims[c] = dict(status="blocked", reasons=sorted(set(blocked[c]))) if blocked[c] else dict(status="ok", **fn())

    emit("C1", lambda: dict(zero_wrong_all_exact_variants_all_files=True, files_checked=85, exact_variants=EXACT))
    emit("C2", lambda: dict(
        set_wrong=R(ALL, "set_wrong"), mset_wrong=R(ALL, "mset_wrong"), set_wrong_managed=R(MAN, "set_wrong"),
        append_set_wrong_equals_dup_requests=all(raw(s, "set_diff", "wrong") == dup(s) for s in S if G("append")(s)),
        append_streams=sum(1 for s in S if G("append")(s)),
        no_dup_streams=[f"{s['data']}/{s['manager']}" for s in S if dup(s) == 0],
        no_dup_set_wrong=rng(s["r"]["set_wrong"] for s in S if dup(s) == 0) if any(dup(s) == 0 for s in S) else None))
    emit("C3", lambda: dict(hash_bytes_append_or_trim=R(G("append", "trim"), "hash_bytes"), hash_bytes_swea=R(G("swea"), "hash_bytes"),
                            hash_bytes_autogen=R(G("autogen"), "hash_bytes")))
    emit("C4", lambda: dict(nokey_bytes_autogen_or_trim=R(G("autogen", "trim"), "nokey_bytes"), nokey_bytes_append_or_swea=R(G("append", "swea"), "nokey_bytes"),
                            nokey_records_all=R(ALL, "nokey_rec")))
    emit("C6", lambda: dict(
        b8_records_trim=R(G("trim"), "b8_rec"), b8_records_all=R(ALL, "b8_rec"), b2_records_trim=R(G("trim"), "b2_rec"), b2_records_all=R(ALL, "b2_rec"),
        b256_abs_deviation_all=R(ALL, "b256_dev"), b0_fallback_managed=R(MAN, "b0_fb"), b0_records_managed=R(MAN, "b0_rec"),
        append_b0_fallback_requests_zero=all(raw(s, "fcx_b0", "fallback_requests") == 0 for s in S if G("append")(s)),
        b64_fallback_all=R(ALL, "b64_fb")))

    # check 3: timing, per stream, then group ranges derived from the per-stream values
    per_stream = []
    for s in S:
        e = dict(data=s["data"], manager=s["manager"], seed=s["seed"])
        why = stream_reasons.get((s["data"], s["manager"]))
        if why:
            e.update(status="blocked", reasons=why)
        else:
            e["status"] = "ok"
            for key in ("full_cpu", "hash_cpu"):
                vals = [round(x[key], 4) for x in s["rr"]]
                e[key] = dict(kind="three fixed-order descriptive measurements", values=vals, median=round(statistics.median(vals), 4),
                              min=min(vals), max=max(vals)) if s["seed"] == 1 else dict(kind="single run", value=vals[0])
        per_stream.append(e)

    def timing(pred, key):
        rows = [e for e, s in zip(per_stream, S) if pred(s)]
        one = [e[key] for e in rows if e["seed"] == 1]
        return dict(seed1_streams=len(one), seed1_median_range=rng(x["median"] for x in one), seed1_min=min(x["min"] for x in one),
                    seed1_max=max(x["max"] for x in one), seed23_streams=len(rows) - len(one),
                    seed23_single_range=rng(e[key]["value"] for e in rows if e["seed"] != 1))
    emit("C5", lambda: dict(full_append=timing(G("append"), "full_cpu"), full_managed=timing(MAN, "full_cpu"),
                            hash_append=timing(G("append"), "hash_cpu"), hash_trim=timing(G("trim"), "hash_cpu"),
                            hash_autogen=timing(G("autogen"), "hash_cpu"), hash_swea=timing(G("swea"), "hash_cpu")))
    out["claims"], out["c5_per_stream"] = claims, per_stream
    write(out, dst)
    print(json.dumps({k: v for k, v in out.items() if k != "c5_per_stream"}, indent=1))
    if out["status"] != "ok":
        raise SystemExit(f"status {out['status']}; see {dst}")


if __name__ == "__main__":
    main()
