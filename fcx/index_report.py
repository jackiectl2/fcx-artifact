"""Fixed analysis of the index comparison (stage `index` of bench.py). One rule for every run:

  - held-out predicates only (names starting with "eval");
  - the factored side is the plan F+I.expand (the default plan without a trace filter), end to end: predicate on the message
    dictionary plus the mapping; no per-predicate choice between plans;
  - a predicate's latency is the median of its repetitions; the speedup is wide / factored;
  - per index kind: median, minimum and maximum speedup over the predicates, the number of predicates on which the wide table
    is faster, and a 95% bootstrap interval of the median (10,000 resamples of the predicates, fixed seed);
  - the same for the cold runs, with the same plan;
  - a configuration counts only if W and F+I.expand are complete, equal and within the time limit on all 30 held-out
    predicates; otherwise the script names it and exits with an error;
  - other plans are reported against the same wide latency: F+N.inv (normalized membership) and R.reconstruct;
  - "ready": wall and CPU seconds from raw requests to a store that answers queries, per side, at the run's thread count.
    Wide: insert of the requests, plus the index build. Factored: the ingester (one Python process), the loads of the
    dictionary, log and checkpoints, the builds of interval rows and arrays, plus the index build on the dictionary.

usage: index_report.py [--per-predicate] results.json [more.json ...]   -> Markdown on stdout
"""
import json, random, statistics, sys

FACT = "F+I.expand"


def med(rec, key):
    return statistics.median(rec[key]) if rec and "error" not in rec and rec.get(key) else None


def ratios(by_pred, num, den, key):
    out = {}
    for pred, plans in by_pred.items():
        a, b = med(plans.get(num), key), med(plans.get(den), key)
        if a is not None and b is not None:
            out[pred] = a / b
    return out


def boot(xs, n=10000, seed=7):
    rng, xs = random.Random(seed), list(xs)
    ms = sorted(statistics.median(rng.choices(xs, k=len(xs))) for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n)]


def summarize(path):
    res = json.load(open(path))
    out = []
    for policy, r in res["index"].items():
        by = {}
        for rec in r["records"]:
            if rec["pred"].startswith("eval"):
                by.setdefault(rec["index"], {}).setdefault(rec["pred"], {})[rec["plan"]] = rec
        for kind, by_pred in by.items():
            row = dict(run=path.split("/")[-1], policy=policy, index=kind, predicates=len(by_pred), variant=r["variants"].get(kind),
                       wide_ms=statistics.median(x for x in (med(p.get("W"), "warm_ms") for p in by_pred.values()) if x is not None),
                       fact_ms=statistics.median(x for x in (med(p.get(FACT), "warm_ms") for p in by_pred.values()) if x is not None),
                       unequal=sum(1 for p in by_pred.values() for rec in p.values() if rec.get("set_equal") is False),
                       over_limit={plan: n for plan in ("W", FACT, "F+N.inv", "R.reconstruct") if (n := sum(1 for p in by_pred.values() if "error" in p.get(plan, {})))})
            # a configuration is publishable only with both plans complete, equal and in limit on all 30 held-out predicates
            row["complete"] = (len(by_pred) == 30 and not row["unequal"] and not row["over_limit"].get("W") and not row["over_limit"].get(FACT)
                               and all(p.get(plan, {}).get("set_equal") is True for p in by_pred.values() for plan in ("W", FACT)))
            for key in ("warm_ms", "cold_ms"):
                x = ratios(by_pred, "W", FACT, key)
                lo, hi = boot(x.values())
                row[key] = dict(median=statistics.median(x.values()), min=min(x.values()), max=max(x.values()), ci=(lo, hi), wide_faster=sum(v < 1 for v in x.values()),
                                n=len(x), worst=min(x, key=x.get))
            for plan in ("F+N.inv", "R.reconstruct", "F+A.unnest"):
                x = ratios(by_pred, "W", plan, "warm_ms")
                if x:
                    row[plan] = dict(median=statistics.median(x.values()), min=min(x.values()), n=len(x))
            x = ratios(by_pred, "F+N.inv", FACT, "warm_ms")
            if x:
                row["N.inv/I.expand"] = dict(median=statistics.median(x.values()), min=min(x.values()), max=max(x.values()))
            out.append(row)
    return res, out


def ready(res, policy):
    """(wall s, CPU s) per side without an index, and per index kind the build (wall, CPU) per side."""
    L, r = res["load"], res["index"][policy]
    g = res["convert"]["policies"][policy].get("ingest")
    if not g or "cpu_s" not in r["load"][f"wx_{policy}"]:
        return None
    steps = [r["load"][f"mx_{policy}"]] + [L[f"{k}_{policy}"] for k in ("d", "ckpt", "i", "a")]
    fact = (g["wall_s"] + sum(t["insert_s"] + t["optimize_s"] for t in steps), g["cpu_s"] + sum(t["cpu_s"] for t in steps))
    w = r["load"][f"wx_{policy}"]
    idx = {k: ((v["wide"]["build_s"], v["wide"].get("build_cpu_s", 0.0)), (v["factored"]["build_s"], v["factored"].get("build_cpu_s", 0.0))) for k, v in r["variants"].items()}
    return dict(ingest=(g["wall_s"], g["cpu_s"]), factored=fact, wide=(w["insert_s"] + w["optimize_s"], w["cpu_s"]), index=idx)


def main():
    f1 = lambda v: f"{v:.2f}" if v < 10 else f"{v:.0f}"
    per_pred = "--per-predicate" in sys.argv
    incomplete = []
    for path in [x for x in sys.argv[1:] if not x.startswith("--")]:
        res, rows = summarize(path)
        m = res["meta"]
        print(f"\n**{path.split('/')[-1]}** — ClickHouse {m['clickhouse']}, {m['threads']} threads, {m['cold']} cold + {m['warm']} warm runs per plan, host {m['host']}\n")
        print("| Policy | Index | Wide ms | Factored ms | Speedup warm: median [95% CI] (min–max) | Wide faster (warm) | Speedup cold: median (min) | Wide faster (cold) | "
              "Build s wide / dict | Index MB wide / dict | Chosen (wide; dict) | With normalized membership: median (min) | Reconstruct first: median |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            w, c, v = r["warm_ms"], r["cold_ms"], r["variant"]
            norm = f"{f1(r['F+N.inv']['median'])}× ({f1(r['F+N.inv']['min'])})" if "F+N.inv" in r else "—"
            rec = f"{f1(r['R.reconstruct']['median'])}× (n={r['R.reconstruct']['n']})" if "R.reconstruct" in r else ("over limit" if r["over_limit"].get("R.reconstruct") else "—")
            print(f"| {r['policy']} | {r['index']} | {r['wide_ms']:.0f} | {r['fact_ms']:.0f} | {f1(w['median'])}× [{f1(w['ci'][0])}, {f1(w['ci'][1])}] ({f1(w['min'])}–{f1(w['max'])}) | "
                  f"{w['wide_faster']} of {w['n']} | {f1(c['median'])}× ({f1(c['min'])}) | {c['wide_faster']} of {c['n']} | "
                  f"{v['wide']['build_s']:.0f} / {v['factored']['build_s']:.1f} | {v['wide']['index_bytes'] / 1e6:.0f} / {v['factored']['index_bytes'] / 1e6:.0f} | "
                  f"param {v['wide']['param']}; granularity {v['factored']['granularity']}, param {v['factored']['param']} | {norm} | {rec} |")
        bad = [(r["index"], r["unequal"], r["over_limit"]) for r in rows if r["unequal"] or r["over_limit"]]
        print(f"\nUnequal sets or plans over the time limit: {bad or 'none'}")
        incomplete += [(path.split("/")[-1], r["policy"], r["index"]) for r in rows if not r["complete"]]
        for policy in res["index"]:
            t = ready(res, policy)
            if t:
                print(f"\nFrom raw requests to a queryable store ({policy}; wall s / CPU s): wide {t['wide'][0]:.0f} / {t['wide'][1]:.0f}; "
                      f"factored {t['factored'][0]:.0f} / {t['factored'][1]:.0f} (of which the ingester {t['ingest'][0]:.0f} / {t['ingest'][1]:.0f}). "
                      "Index build added, wide vs dictionary: " + "; ".join(f"{k} {w[0]:.0f} / {w[1]:.0f} vs {d[0]:.1f} / {d[1]:.1f}" for k, (w, d) in t["index"].items() if k != "none"))
        if per_pred:
            print("\n| Policy | Index | Predicate | String | Matched messages | Wide warm ms | Factored warm ms | Speedup |\n|---|---|---|---|---|---|---|---|")
            for policy, r in res["index"].items():
                for rec in r["records"]:
                    if rec["plan"] == "W" and rec["pred"].startswith("eval"):
                        fr = next(x for x in r["records"] if x["plan"] == FACT and x["pred"] == rec["pred"] and x["index"] == rec["index"])
                        a, b = med(rec, "warm_ms"), med(fr, "warm_ms")
                        print(f"| {policy} | {rec['index']} | {rec['pred']} | `{r['predicates'][rec['pred']]}` | {rec['matched_messages']} | {a:.1f} | {b:.1f} | {a / b:.2f}× |")
    if incomplete:
        sys.exit(f"INCOMPLETE (no publishable summary): {incomplete}")


if __name__ == "__main__":
    main()
