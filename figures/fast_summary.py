"""E-FAST summary: derivation timings of the final (v2) ingestion code (refine-logs/E-FAST-PROTOCOL.md); frozen before use.

Inputs (copied from the cluster unchanged), all written by fcx/ingest_ablate.py with ingest_impl "v2" and the extra variant
hash_array_delta:
  results/ing_s1_v2, ing_s1_v2r2, ing_s1_v2r3   the 17 seed-1 streams of cluster/tasks/s1.txt, three fixed-order runs
  results/ing_s23_v2                            the 34 seed-2/3 streams of cluster/tasks/s23.txt
  results/ing_chat_v2, results/ing_wiki_v2      the 12 WildChat and 3 Wikipedia streams (cluster/tasks/chat.txt, wiki.txt)
  results/online_v2                             E-ONLINE with v2 (figures/online_duck_summary.online, tag online_v2, impl v2)
and, for the cross-implementation check, the v1 results of the same streams (ing_s1_v1, ing_s1_rep2, ing_s1_rep3, ing_s23_v1,
ing_chat_v1, ing_wiki_v1).
Rules (any problem: status "blocked", no number; a missing file: status "incomplete"):
 1. each file: figures/ing_ext_summary.validate on the file without hash_array_delta (identity, fixed parameters, variant set,
    types, ranges, denominators); ingest_impl == "v2"; hash_array_delta present, exact (must_be_exact, wrong = 0), with the
    requests, records (one per call), array elements and payload of hash_array (the same arrays);
 2. cross-implementation agreement: every non-timing field (all keys except cpu_s, wall_s, total_wall_s, peak_rss_mb) of the
    file without hash_array_delta and ingest_impl equals the v1 result of the same stream and run, i.e. every reported
    non-timing aggregate matches v1, and every exact v2 variant rebuilds every request byte for byte (per-request identity
    of the outputs is shown by fcx/check_ingest_v2.py on seven full streams and fcx/test_ingest_v2.py, not here);
 3. ratios per stream, against FCX (fcx_b64, delta parsing): full_cpu = fcx_full / fcx_b64 (FCX without delta parsing),
    hash_cpu = hash_array / fcx_b64 (full parsing), hash_delta_cpu = hash_array_delta / fcx_b64 (same delta-parsing front end
    and dictionary; no occurrence or small-edit variant derivation); seed 1 as the median, min and max of the three runs, seeds 2/3 and WildChat/Wikipedia as
    single runs; group ranges as in figures/ing_ext_summary.py (append, trim, autogen, swea, managed) and per WildChat manager;
 4. online: figures/online_duck_summary.online(tag="online_v2", impl="v2"), unchanged rules.
 5. ingest_cmp (Fig. 5a and the ingestion records of Table IV; separate status, required by figures/fig5_ingest.py and
    figures/tab3_cost.py when FCX_INGEST_V2=1): results/fast_cmp_v2/cap_<c>_s1__ingest_cmp.json for the samples of
    cluster/tasks/fast_cmp.txt, each with ingest_impl v2; zero differing tables between full and delta parsing (logical
    tables, pyarrow Table.equals) and files_compared = 2 + 3 x managers; both parses from the same source sample, with
    interleave 64, budget 64, checkpoint interval 32, and the same traces, spans and distinct messages; the managers in the
    exact order of fast_cmp.txt (the dictionary is shared across managers, so later managers' timings depend on the order;
    prefix first), which as a set equal those of cluster/tasks/s1.txt; per manager the requests of the E-ING v2 run, delta 1
    against absent, finite positive wall time and throughput, 0 <= parsed bytes <= request bytes, and parsed bytes = request
    bytes under full parsing; ranges of full and delta MB/s and of the parsed share.
usage: .venv/bin/python figures/fast_summary.py [--out results/fast_summary.json]
"""
import argparse, copy, json, statistics, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ing_ext_summary as ing                                    # noqa: E402
import online_duck_summary as ods                                # noqa: E402

RES = ing.RES
V2 = {"s1": ["ing_s1_v2", "ing_s1_v2r2", "ing_s1_v2r3"], "s23": ["ing_s23_v2"], "chat": ["ing_chat_v2"], "wiki": ["ing_wiki_v2"]}
V1 = {"s1": ["ing_s1_v1", "ing_s1_rep2", "ing_s1_rep3"], "s23": ["ing_s23_v1"], "chat": ["ing_chat_v1"], "wiki": ["ing_wiki_v1"]}
TASKS = {"s1": "s1.txt", "s23": "s23.txt", "chat": "chat.txt", "wiki": "wiki.txt"}
KEYS = ("full_cpu", "hash_cpu", "hash_delta_cpu")


def streams(part):
    return [tuple(l.split()) for l in (ing.ROOT / "cluster" / "tasks" / TASKS[part]).read_text().splitlines() if l.strip()]


def check(d, data, manager):
    """Problems of one v2 file (rule 1) and the file without the E-FAST additions."""
    if not isinstance(d, dict) or not isinstance(d.get("variants"), dict):
        return ["not a result object"], None
    probs = [] if d.get("ingest_impl") == "v2" else [f"ingest_impl {d.get('ingest_impl')!r}"]
    base = copy.deepcopy(d); base.pop("ingest_impl", None)
    hd = base["variants"].pop("hash_array_delta", None)
    probs += ing.validate(base, data, manager)
    h, f = base["variants"].get("hash_array", {}), base["variants"].get("fcx_b64", {})
    if not isinstance(hd, dict):
        probs.append("hash_array_delta missing")
    else:
        sl = hd.get("by_slice")
        ok = (hd.get("must_be_exact") is True and hd.get("wrong") == 0 and all(ing.measure(hd.get(k)) and hd[k] > 0 for k in ("cpu_s", "wall_s"))
              and all(ing.count(hd.get(k)) for k in ("requests", "records", "array_elems", "payload_bytes", "parsed_bytes", "logical_bytes"))
              and hd.get("requests") == d.get("requests") and hd.get("records") == d.get("requests") and hd.get("records_per_request") == 1
              and all(hd.get(k) == h.get(k) for k in ("array_elems", "payload_bytes", "records", "logical_bytes"))
              and hd.get("parsed_bytes") == f.get("parsed_bytes") and hd.get("logical_bytes") == f.get("logical_bytes")
              and isinstance(sl, dict) and isinstance(f.get("by_slice"), dict) and set(sl) == set(f["by_slice"])
              and all(sl[k].get("requests") == f["by_slice"][k].get("requests") and sl[k].get("wrong") == 0 for k in sl))
        if not ok:
            probs.append("hash_array_delta not exact, not the arrays of hash_array, or not the delta parse of fcx_b64")
    return probs, base


def ratios(d):
    v = d["variants"]; f = v["fcx_b64"]["cpu_s"]
    return dict(full_cpu=v["fcx_full"]["cpu_s"] / f, hash_cpu=v["hash_array"]["cpu_s"] / f, hash_delta_cpu=v["hash_array_delta"]["cpu_s"] / f)


def ingest_cmp():
    import math
    probs, rows, files = [], [], []
    order = {l.split()[0]: l.split()[1].split(":") for l in (ing.ROOT / "cluster" / "tasks" / "fast_cmp.txt").read_text().splitlines() if l.strip()}
    sets = {}
    for data, manager in streams("s1"):
        sets.setdefault(data, set()).add(manager)
    if set(order) != set(sets) or any(set(order[d]) != sets[d] for d in sets):
        return dict(status="blocked", problems=["cluster/tasks/fast_cmp.txt does not cover the managers of s1.txt"])
    pos = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0
    for data, managers in order.items():
        p = RES / "fast_cmp_v2" / f"{data}__ingest_cmp.json"
        try:
            d = json.load(open(p))
        except (OSError, ValueError) as e:
            return dict(status="incomplete", problems=[f"{p.name}: {type(e).__name__}"])
        f, dl = d.get("delta0", {}), d.get("delta1", {})
        if not (d.get("ingest_impl") == "v2" and d.get("files_that_differ") == [] and d.get("files_compared") == 2 + 3 * len(managers)
                and all(Path(str(x.get("src", ""))).name == data and x.get("interleave") == 64 and x.get("budget") == 64 and x.get("ckpt") == 32 for x in (f, dl))
                and all(f.get(k) == dl.get(k) for k in ("traces", "spans", "distinct_messages"))
                and list(f.get("policies", {})) == managers and list(dl.get("policies", {})) == managers):
            probs.append(f"{p.name}: implementation, table equality, source, parameters or manager order")
            continue
        files.append(d["files_compared"])
        for m in managers:
            a, b = f["policies"][m], dl["policies"][m]
            n = json.load(open(RES / "ing_s1_v2" / f"{data}__{m}.json"))["requests"]
            ok = (a["ingest"]["requests"] == b["ingest"]["requests"] == n and int(b["ingest"].get("delta", 0)) == 1 and not a["ingest"].get("delta")
                  and all(pos(x["ingest"].get(k)) for x in (a, b) for k in ("wall_s", "logical_mb_per_s"))
                  and 0 <= b["parsed_bytes"] <= b["wide_chars"] and a["parsed_bytes"] == a["wide_chars"] > 0 and a["wide_chars"] == b["wide_chars"])
            if not ok:
                probs.append(f"{p.name} {m}: requests, parse mode, times or parsed bytes")
                continue
            rows.append(dict(data=data, manager=m, full_mb_per_s=a["ingest"]["logical_mb_per_s"], delta_mb_per_s=b["ingest"]["logical_mb_per_s"],
                             parsed_share=b["parsed_bytes"] / b["wide_chars"], full_wall_s=a["ingest"]["wall_s"], delta_wall_s=b["ingest"]["wall_s"]))
    if probs:
        return dict(status="blocked", problems=probs)
    return dict(status="ok", files_compared=ing.rng(files), files_that_differ=0, streams=len(rows),
                full_mb_per_s=ing.rng(r["full_mb_per_s"] for r in rows), delta_mb_per_s=ing.rng(r["delta_mb_per_s"] for r in rows),
                parsed_share=ing.rng(r["parsed_share"] for r in rows), per_stream=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RES / "fast_summary.json"))
    dst = Path(ap.parse_args().out)
    out, probs, missing, per = dict(protocol="refine-logs/E-FAST-PROTOCOL.md"), [], [], {}
    for part in V2:
        for data, manager in streams(part):
            rr = []
            for r2, r1 in zip(V2[part], V1[part]):
                p2, p1 = RES / r2 / f"{data}__{manager}.json", RES / r1 / f"{data}__{manager}.json"
                try:
                    d2 = json.load(open(p2))
                except (OSError, ValueError) as e:
                    missing.append(f"{r2}/{p2.name}: {type(e).__name__}")
                    continue
                pr, base = check(d2, data, manager)
                if not pr:
                    d1 = json.load(open(p1))
                    diffs = list(ing.diff_paths(ing.strip_timing(base), ing.strip_timing(d1)))
                    if diffs:
                        pr.append(f"differs from v1 at {diffs[:5]}")
                if pr:
                    probs.append(f"{r2}/{p2.name}: {pr[:3]}")
                    continue
                rr.append(ratios(d2))
            if len(rr) == len(V2[part]):
                per[(part, data, manager)] = rr
    if probs or missing:
        out.update(status="blocked" if probs else "incomplete", problems=probs[:40], missing=missing)
    else:
        rows = []
        for (part, data, manager), rr in per.items():
            e = dict(part=part, data=data, manager=manager)
            for k in KEYS:
                vals = [round(x[k], 4) for x in rr]
                e[k] = dict(values=vals, median=round(statistics.median(vals), 4), min=min(vals), max=max(vals)) if len(vals) > 1 else dict(value=vals[0])
            rows.append(e)
        G = {g: (lambda s, ms=ms: s["manager"] in ms) for g, ms in ing.GROUPS.items()}
        G["managed"] = lambda s: s["manager"] in ing.MANAGED

        def group(part, pred, key):
            xs = [e for e in rows if e["part"] == part and pred(e)]
            if part == "s1":
                return dict(streams=len(xs), median_range=ing.rng(e[key]["median"] for e in xs), min=min(e[key]["min"] for e in xs), max=max(e[key]["max"] for e in xs))
            return dict(streams=len(xs), single_range=ing.rng(e[key]["value"] for e in xs))
        swe = {part: {g: {k: group(part, pred, k) for k in KEYS} for g, pred in G.items()} for part in ("s1", "s23")}
        chat = {m: {k: group("chat", lambda e, m=m: e["manager"] == m, k) for k in KEYS} for m in ("prefix", "autogen_buf32", "autogen_ht", "lc_trim32k")}
        chat["all"] = {k: group("chat", lambda e: True, k) for k in KEYS}
        wiki = {k: group("wiki", lambda e: True, k) for k in KEYS}
        onl = ods.online(tag="online_v2", impl="v2")
        out["ingest_cmp"] = ingest_cmp()
        status = "ok" if onl.get("status") == "ok" else onl.get("status", "blocked")
        out.update(status=status, swe=swe, chat=chat, wiki=wiki, online=onl, per_stream=rows,
                   files=sum(len(v) for v in per.values()), cross_implementation="every reported non-timing aggregate equals v1 on every file; every exact variant rebuilds every request")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp"); json.dump(out, open(tmp, "w"), indent=1); tmp.replace(dst)
    print(out["status"], json.dumps({k: out.get(k) for k in ("swe", "chat", "wiki")}, indent=1)[:5000])
    if out["status"] != "ok":
        raise SystemExit(f"status {out['status']}: {(out.get('problems') or out.get('missing') or [out.get('online', {}).get('problems')])[:3]}")


if __name__ == "__main__":
    main()
