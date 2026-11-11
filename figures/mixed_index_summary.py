"""The running example (K = 3) and ctx_count over the 30 held-out predicates with a 3-gram text index on the dictionary and on the
wide table (mixedidx2 runs of bench.py stage mixed; reduced protocol: K = 3 only, 2 cold and 3 warm runs).

Gates per run: the run finished (no top-level mismatch, stage time recorded, only its own policy); mixed_index is ngrams3 and both
text tables hold the text index; the 30 predicates are exactly those of the same sample's index-free mixed run (and in its order);
exactly one record per (query, plan, predicate), 210 in all; every record validated (set_equal for residence, rows_differ == 0
for ctx_count) with no time-limit or error entry. A failed gate stops the script for that run and it is listed as excluded.

Reported: per-predicate speedups W / plan on warm medians (and cold medians), by predicate class, for W and W.distinct as the wide
reference and for I and N (I.expand and N.inv for ctx_count); the same sample's index-free speedup; index builds and bytes. Store fields are
taken before the index is built (index bytes are separate); occ_store is the interval table.
Ranges are over seeds 1-3 and are printed only for complete sets (pre-registered rule): a (corpus, policy) with a missing or invalid
run gets no range. --paper prints what Section 6 draws on: per corpus, append only and the managed policies as two groups.
usage: .venv/bin/python figures/mixed_index_summary.py [--across | --paper | --merge]
"""
import statistics as st
import sys
from collections import defaultdict

from style import CORPUS, RESULTS, load

R = RESULTS / "r1_2026-11-01"
R24 = RESULTS / "r1_2026-10-24"
PLANS = [("residence", "W"), ("residence", "W.distinct"), ("residence", "I"), ("residence", "N"),
         ("ctx_count", "W"), ("ctx_count", "I.expand"), ("ctx_count", "N.inv")]
RUNS = {"gym": ["prefix", "autogen_ht", "autogen_buf32", "lc_trim32k"], "swea": ["prefix", "swea_last5", "swea_window"],
        "smith": ["prefix", "swea_cache2", "swea_last5"], "oh": ["prefix", "autogen_ht", "autogen_buf32", "lc_trim32k"]}
CLASSES = (("sampled substrings", "eval:sub"), ("regular expressions", "eval:re"), ("fixed needles", "eval:t"))
COMPARE = (("residence", "W", "I"), ("residence", "W.distinct", "I"), ("residence", "W", "N"), ("residence", "W.distinct", "N"),
           ("ctx_count", "W", "I.expand"), ("ctx_count", "W", "N.inv"))


def wm(x, key="warm_ms"):
    return st.median(x[key])


def check(c, s, p):
    """One run's table {(query, plan): {pred: record}} and metadata, after every gate; raises AssertionError on a failed gate."""
    path = R / f"cap_{c}_s{s}__mixedidx2_{p}.json"
    d = load(path)
    assert "mismatch" not in d and "mixed_s" in d["meta"] and list(d["mixed"]) == [p], f"{path.name}: run did not finish"
    m = d["mixed"][p]
    assert d["load"][f"wm_{p}"]["granularity"] == 64 and d["load"][f"mm_{p}"]["granularity"] == 64, f"{path.name}: granularity"
    assert all(x.get("K") in (3, None) and len(x["cold_ms"]) == 2 and len(x["warm_ms"]) == 3 for x in m["records"]), f"{path.name}: protocol"
    assert m["mixed_index"] == "ngrams3" and all("text(tokenizer = ngrams(3))" in v for v in m["text_indexes"].values()) and len(m["text_indexes"]) == 2, f"{path.name}: index"
    ref = load(R24 / f"cap_{c}_s{s}__mixed_{p}.json")["mixed"][p]
    assert list(m["predicates"].items()) == list(ref["predicates"].items()) and len(m["predicates"]) == 30, f"{path.name}: predicates differ from the index-free run"
    assert m["spans"] == ref["spans"], f"{path.name}: calls differ from the index-free run"
    t = defaultdict(dict)
    for x in m["records"]:
        assert "error" not in x and "unvalidated" not in x, f"{path.name}: {x['pred']} {x['plan']} over the limit"
        assert x.get("set_equal") is True if x["query"] == "residence" else x.get("rows_differ") == 0, f"{path.name}: unvalidated record"
        assert x["pred"] not in t[(x["query"], x["plan"])], f"{path.name}: duplicate"
        t[(x["query"], x["plan"])][x["pred"]] = x
    assert len(m["records"]) == 210 and all(set(t[q]) == set(m["predicates"]) for q in PLANS), f"{path.name}: records incomplete"
    base = defaultdict(dict)                                     # the same sample's earlier index-free run (other code version and layout), K = 3
    for x in ref["records"]:
        if x.get("K") in (3, None):
            assert "error" not in x and "unvalidated" not in x and x["pred"] not in base[(x["query"], x["plan"])], f"{path.name}: reference record"
            base[(x["query"], x["plan"])][x["pred"]] = x
    assert all(set(base[q]) == set(m["predicates"]) for q in PLANS), f"{path.name}: reference incomplete"
    return t, base, m, d


def ratios(t, q, a, b, prefix="eval", key="warm_ms"):
    return {k: wm(t[(q, a)][k], key) / wm(t[(q, b)][k], key) for k in t[(q, a)] if k.startswith(prefix) and k in t[(q, b)]}


def one(c, s, p):
    t, base, m, d = check(c, s, p)
    out = dict(c=c, s=s, p=p, spans=m["spans"], mins=d["meta"]["mixed_s"] / 60, cmp={}, cold={}, cls={}, base={}, lat={})
    for q, a, b in COMPARE:
        r = ratios(t, q, a, b)
        out["cmp"][(q, a, b)] = (st.median(r.values()), min(r.values()), max(r.values()), sum(v < 1 for v in r.values()))
        out["cold"][(q, a, b)] = st.median(ratios(t, q, a, b, key="cold_ms").values())
        out["cls"][(q, a, b)] = {n: (st.median((rr := ratios(t, q, a, b, pre)).values()), sum(v < 1 for v in rr.values()), len(rr)) for n, pre in CLASSES}
        rb = ratios(base, q, a, b)
        out["base"][(q, a, b)] = (st.median(rb.values()), len(rb)) if rb else None
    both = {k: min(wm(t[("residence", "W")][k]), wm(t[("residence", "W.distinct")][k])) / wm(t[("residence", "I")][k]) for k in t[("residence", "I")]}
    out["best"] = (st.median(both.values()), sum(v < 1 for v in both.values()))
    # observed change against the earlier index-free run (not an isolated index effect: other code version, layout and K set)
    self_gain = lambda q, a: st.median(wm(base[(q, a)][k]) / wm(t[(q, a)][k]) for k in t[(q, a)])
    out["gain"] = {"W": self_gain("residence", "W"), "W.distinct": self_gain("residence", "W.distinct"), "I": self_gain("residence", "I"), "N": self_gain("residence", "N"),
                   "count W": self_gain("ctx_count", "W"), "count I": self_gain("ctx_count", "I.expand"), "count N": self_gain("ctx_count", "N.inv")}
    out["min_ratio"] = {k: out["cmp"][k][1] for k in COMPARE}
    out["slower"] = {k: out["cmp"][k][3] for k in COMPARE}
    out["lat"] = {q: st.median(wm(x) for x in t[q].values()) for q in PLANS}
    out["read"] = {q: st.median(x["read_rows"] for x in t[q].values()) for q in PLANS}
    M, W = f"mm_{p}", f"wm_{p}"
    ib = m["index_build"]
    out["cost"] = dict(dict_s=ib[M]["build_s"], dict_cpu=ib[M]["build_cpu_s"], dict_b=ib[M]["index_bytes"], wide_s=ib[W]["build_s"], wide_cpu=ib[W]["build_cpu_s"], wide_b=ib[W]["index_bytes"],
                       dict_store=d["load"][M]["bytes_on_disk"], wide_store=d["load"][W]["bytes_on_disk"], occ_store=d["load"][f"i_{p}"]["bytes_on_disk"], norm_store=d["load"].get(f"ni_{p}", {}).get("bytes_on_disk"))
    return out


def merge_sensitivity():
    """Seed-1 runs with both text tables merged into one part before the index builds (mixedidx2m, code_v10), against the same runs as
    loaded (mixedidx2): per-run median speedups, merge state, and index builds."""
    worst = {}
    for c, p in (("gym", "autogen_ht"), ("gym", "prefix"), ("swea", "swea_last5"), ("swea", "prefix")):
        a, b = load(R / f"cap_{c}_s1__mixedidx2_{p}.json"), load(R / f"cap_{c}_s1__mixedidx2m_{p}.json")
        ma, mb = a["mixed"][p], b["mixed"][p]
        assert "mismatch" not in b and len(mb["records"]) == 210 and mb["predicates"] == ma["predicates"] and mb["merge_state"]["mixed_merge"]
        assert mb["merge_state"]["before"] == mb["merge_state"]["after"] and all(v.startswith("['") and "','" not in v.split("\t")[0] for v in mb["merge_state"]["before"].values())
        assert all(x.get("set_equal") is True if x["query"] == "residence" else x.get("rows_differ") == 0 for x in mb["records"])
        ib = mb["index_build"]
        print(f"{CORPUS[c][0]} {p}: wide build {ib[f'wm_{p}']['build_s']} s, merge {b['load'][f'wm_{p}']['optimize_s']} s")
        for q, x, y in COMPARE:
            f = lambda m: st.median(wm(u[(x)]) / wm(u[(y)]) for u in pivot(m, q).values())
            ra, rb = f(ma), f(mb)
            worst[(q, x, y)] = max(worst.get((q, x, y), 0), abs(rb / ra - 1))
            print(f"  {q} {x}/{y}: as loaded {ra:.2f}, merged {rb:.2f}, change {100 * (rb / ra - 1):+.1f}%")
    print("largest absolute change per comparison: " + "; ".join(f"{q} {x}/{y} {100 * v:.1f}%" for (q, x, y), v in worst.items()))


def pivot(m, q):
    t = defaultdict(dict)
    for x in m["records"]:
        if x["query"] == q:
            t[x["pred"]][x["plan"]] = x
    return t


def main(argv):
    if "--merge" in argv:
        return merge_sensitivity()
    done, excluded = [], []
    for c, ps in RUNS.items():
        for p in ps:
            for s in (1, 2, 3):
                if not (R / f"cap_{c}_s{s}__mixedidx2_{p}.json").exists():
                    excluded.append(f"{c} s{s} {p}: missing")
                    continue
                try:
                    done.append(one(c, s, p))
                except AssertionError as e:
                    excluded.append(f"{c} s{s} {p}: {e}")
    if "--paper" in argv:
        groups = defaultdict(list)
        for o in done:
            groups[(o["c"], o["p"])].append(o)
        for c, ps in RUNS.items():
            for label, sel in (("append only", ["prefix"]), ("managed", ps[1:])):
                if not all(len(groups[(c, p)]) == 3 for p in sel):
                    print(f"{CORPUS[c][0]} {label}: INCOMPLETE ({sum(len(groups[(c, p)]) for p in sel)} of {3 * len(sel)} runs)")
                    continue
                g = [o for p in sel for o in groups[(c, p)]]
                print(f"\n{CORPUS[c][0]} {label}: {len(g)} runs, {len(g) * 30} predicate rows")
                rg = lambda f: (lambda v: f"{min(v):.2f}-{max(v):.2f}")([f(o) for o in g])
                for k in COMPARE:
                    print(f"  {k[0]} {k[1]}/{k[2]}: warm {rg(lambda o: o['cmp'][k][0])}; lowest per-predicate {min(o['cmp'][k][1] for o in g):.2f}; wide faster on {sum(o['cmp'][k][3] for o in g)}; cold {rg(lambda o: o['cold'][k])}; index-free warm {rg(lambda o: o['base'][k][0])}; "
                          + "; ".join(f"{n} {rg(lambda o: o['cls'][k][n][0])}" for n, _ in CLASSES))
                print(f"  best wide formulation per predicate / I: {rg(lambda o: o['best'][0])}; wide faster on {sum(o['best'][1] for o in g)}")
                print("  change against the earlier index-free run: " + "; ".join(f"{a} {rg(lambda o: o['gain'][a])}" for a in g[0]["gain"]))
                print("  warm median ms: " + "; ".join(f"{q[0][:3]} {q[1]} {rg(lambda o: o['lat'][q])}" for q in PLANS))
                print("  cost: " + "; ".join(f"{a} {min(o['cost'][a] for o in g) / (1e6 if a.endswith(('_b', '_store')) else 1):.1f}-{max(o['cost'][a] for o in g) / (1e6 if a.endswith(('_b', '_store')) else 1):.1f}" for a in g[0]["cost"]))
    elif "--across" in argv:
        groups = defaultdict(list)
        for o in done:
            groups[(o["c"], o["p"])].append(o)
        print("| corpus | policy | seeds | K=3 W/I | W.distinct/I | W/N | W.distinct/N | count W/I | count W/N | I-free W/I | I-free count W/I |\n|" + "---|" * 11)
        rg = lambda g, f: (lambda v: f"{min(v):.2f}-{max(v):.2f}")([f(o) for o in g])
        for (c, p), g in groups.items():
            if len(g) != 3:
                print(f"| {CORPUS[c][0]} | {p} | {len(g)} | INCOMPLETE |")
                continue
            cells = [rg(g, lambda o, k=k: o["cmp"][k][0]) for k in COMPARE]
            cells += [rg(g, lambda o, k=COMPARE[i]: o["base"][k][0]) for i in (0, 4)]
            print(f"| {CORPUS[c][0]} | {p} | {len(g)} | " + " | ".join(cells) + " |")
    else:
        for o in done:
            print(f"\n**{CORPUS[o['c']][0]} s{o['s']} {o['p']}**: {o['spans']:,} calls, {o['mins']:.0f} min")
            for k in COMPARE:
                med, lo, hi, n = o["cmp"][k]
                b = o["base"][k]
                cl = "; ".join(f"{n2} {v[0]:.2f}x ({v[1]}/{v[2]} slower)" for n2, v in o["cls"][k].items())
                print(f"  {k[0]} {k[1]}/{k[2]}: warm {med:.2f}x ({lo:.2f}-{hi:.1f}), slower on {n}/30; cold {o['cold'][k]:.2f}x; index-free {b[0] if b else float('nan'):.2f}x; {cl}")
            print("  warm median ms: " + ", ".join(f"{q[0][:3]} {q[1]} {v:.0f}" for q, v in o["lat"].items()))
            print("  median read rows: " + ", ".join(f"{q[0][:3]} {q[1]} {v:,.0f}" for q, v in o["read"].items()))
            print("  cost: " + ", ".join(f"{k} {v}" for k, v in o["cost"].items()))
    print(f"\n{len(done)} runs valid; excluded or missing: {len(excluded)}")
    for e in excluded:
        print(" ", e)


if __name__ == "__main__":
    main(sys.argv[1:])
