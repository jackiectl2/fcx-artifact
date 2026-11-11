"""Manager table (Table 3 in the PDF; file name kept): content predicates against the indexed wide table under each context manager (seed-1 capture samples).

The rule of fcx/index_report.py: held-out predicates only; per predicate, warm speedup = median wide latency (plan W) /
median FCX latency (plan F+I.expand); per index kind, the median over predicates. The superscript counts the predicates on
which the wide table is faster. Inputs are the fixed list RUNS: the seed-1 index run of every manager plus the seed-1
append-only run of the same corpus for reference. Every run must use 16 threads with both read thresholds at zero and hold
the same sample (store, traces, spans) as the append-only run of its corpus. A cell is printed only if every held-out
predicate of that kind has both plans, no plan over the time limit, and equal sets; otherwise the script stops.
Index kinds: the five measured in every run; sparse grams were measured for the append-only runs only (fixed in
cluster/r1_jobs.sh before the managed runs were submitted) and are left out here.
Since the final review: all three samples per pair; cell = mean of the three per-sample medians, subscript = their sample SD, bold = largest mean in the row, grey row = append only.
usage: .venv/bin/python figures/tab4_managers.py
"""
import statistics
import sys
from pathlib import Path

from style import LATE, RESULTS, load

OUT = Path(__file__).resolve().parent.parent / "paper" / "figures" / "tab4_managers.tex"
R24 = RESULTS / "r1_2026-10-24"
FACT, KINDS = "F+I.expand", ("none", "token", "ngrams3", "ngrams4", "bloom4")
MANAGERS = {"prefix": "append only", "swea_last5": "last 5 obs.", "swea_window": "closed window",
            "swea_cache2": "cache marks", "autogen_ht": "head+tail", "autogen_buf32": "buffer 32",
            "lc_trim32k": "trim 32k"}   # same suffixes as the figure labels
RUNS = [("swea", "SWE-agent", [("prefix", LATE), ("swea_last5", R24), ("swea_window", R24)]),
        ("smith", "SWE-smith", [("prefix", R24), ("swea_cache2", R24), ("swea_last5", R24)]),
        ("oh", "SWE-rebench", [("prefix", LATE), ("autogen_ht", R24), ("autogen_buf32", R24), ("lc_trim32k", R24)]),
        ("gym", "SWE-Gym", [("prefix", R24), ("autogen_ht", R24), ("autogen_buf32", R24), ("lc_trim32k", R24)])]


def config(res):
    s = res["meta"].get("settings", {})
    return (res["meta"]["threads"], s.get("merge_tree_min_rows_for_concurrent_read"), s.get("merge_tree_min_bytes_for_concurrent_read"),
            Path(res["convert"]["src"]).name, res["convert"]["traces"], res["convert"]["spans"])


def stats(res, policy, kind, path):
    """(median speedup over predicates, predicates on which the wide table is faster, predicates); stops on any gap."""
    r = res["index"][policy]
    preds = [p for p in r["predicates"] if p.startswith("eval")]
    by = {}
    for rec in r["records"]:
        if rec["index"] == kind and rec["pred"].startswith("eval"):
            by.setdefault(rec["pred"], {})[rec["plan"]] = rec
    sp = []
    for p in preds:
        plans = by.get(p, {})
        bad = [k for k in ("W", FACT) if k not in plans or "error" in plans[k] or plans[k].get("set_equal") is not True]
        if bad:
            sys.exit(f"{path.name} {kind} {p}: plans {bad} missing, over the time limit or unequal")
        sp.append(statistics.median(plans["W"]["warm_ms"]) / statistics.median(plans[FACT]["warm_ms"]))
    if not sp:
        sys.exit(f"{path.name} {kind}: no held-out predicates")
    return statistics.median(sp), sum(x < 1 for x in sp), len(sp)


def cell(sts, best):
    """Mean over the three samples of the per-sample median speedup, with its sample SD as a subscript; bold marks the row's largest mean."""
    meds = [x[0] for x in sts]
    m, sd = statistics.fmean(meds), statistics.stdev(meds)
    assert all(x[2] == 30 for x in sts) and len(sts) == 3
    return (rf"\textbf{{{m:.1f}}}" if best else f"{m:.1f}") + rf"\sdsub{{{sd:.1f}}}"


def main():
    table, lines = {}, []
    for c, name, runs in RUNS:
        for m, d in runs:
            sts = []
            for seed in (1, 2, 3):
                ref_dir = d if seed == 1 else R24
                path = ref_dir / f"cap_{c}_s{seed}__index_{m}.json"
                base = (runs[0][1] if seed == 1 else R24) / f"cap_{c}_s{seed}__index_prefix.json"
                res, cf, rf = load(path), None, None
                cf, rf = config(res), config(load(base))
                assert cf[:3] == (16, 0, 0), f"{path.name}: threads or read thresholds {cf[:3]}"
                assert cf[3:] == rf[3:], f"{path.name}: sample {cf[3:]} differs from the append-only run {rf[3:]}"
                sts.append([stats(res, m, k, path) for k in KINDS])
            table[(c, m)] = [[sts[s][k] for s in range(3)] for k in range(len(KINDS))]
    for c, name, runs in RUNS:
        for i, (m, _) in enumerate(runs):
            cs = table[(c, m)]
            means = [statistics.fmean(x[0] for x in v) for v in cs]
            print(c, m, [(round(statistics.fmean(x[0] for x in v), 2), round(statistics.stdev(x[0] for x in v), 2), sum(x[1] for x in v)) for v in cs])
            shade = r"\rowcolor{black!7}" if m == "prefix" else ""
            lines.append(f"    {shade}{name if i == 0 else '':<12} & {MANAGERS[m]:<17} & " + " & ".join(cell(v, x == max(means)) for v, x in zip(cs, means)) + r" \\")
        lines.append(r"    \midrule")
    lines[-1] = r"    \bottomrule"
    OUT.write_text("\n".join(lines) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
