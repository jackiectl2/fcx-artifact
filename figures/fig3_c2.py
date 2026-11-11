"""Figure 3: content predicates against a wide table that carries the same text index (claim C2).

Per held-out predicate, warm speedup = wide latency / factored latency (plan F+I.expand, median of the warm runs), per index
kind and corpus; hollow diamonds are the medians of the cold runs. Index build cost is in Figure 5. Reads the fixed list RUNS of seed-1 index results (both read thresholds at zero); a run on a
delta-ingested store (cap_<c>_s1d) counts as a repetition of seed 1. Samples drawn with other seeds are reported in the text,
not here. Stops if any index kind lacks a complete, equal, in-limit result for each of the 30 held-out predicates.
usage: .venv/bin/python figures/fig3_c2.py
"""
import statistics

import numpy as np

from style import CORPUS, COLORS, LATE, RESULTS, TEXT_IN, load, save, setup

FACT, KINDS = "F+I.expand", ["none", "token", "ngrams3", "ngrams4", "sparsegrams", "bloom4"]
LABEL = {"none": "no index", "token": "token", "ngrams3": "3-gram", "ngrams4": "4-gram", "sparsegrams": "sparse grams", "bloom4": "Bloom"}
MARK = {"sub": "o", "re": "^", "t": "s"}


R24 = RESULTS / "r1_2026-10-24"
RUNS = [("swea", "1", LATE / "cap_swea_s1__index_prefix.json"), ("swea", "1d", R24 / "cap_swea_s1d__index_prefix.json"),
        ("smith", "1", R24 / "cap_smith_s1__index_prefix.json"),
        ("oh", "1", LATE / "cap_oh_s1__index_prefix.json"), ("oh", "1d", R24 / "cap_oh_s1d__index_prefix.json"),
        ("gym", "1", R24 / "cap_gym_s1__index_prefix.json")]


def runs():
    """(corpus, seed tag, result) for the fixed list RUNS; each must use both read thresholds at zero."""
    out = []
    for corpus, seed, path in RUNS:
        res = load(path)
        s = res["meta"]["settings"]
        assert s["merge_tree_min_rows_for_concurrent_read"] == 0 and s["merge_tree_min_bytes_for_concurrent_read"] == 0, path
        out.append((corpus, seed, res))
    return out


def per_predicate(res, kind):
    """pred -> (warm ratio, cold ratio) for the 30 held-out predicates of one index kind; stops on a missing, unequal or
    timed-out plan, so that no summary is drawn from a subset."""
    by = {}
    for rec in res["index"]["prefix"]["records"]:
        if rec["index"] == kind and rec["pred"].startswith("eval") and rec["plan"] in ("W", FACT):
            assert "error" not in rec and rec.get("set_equal") is True, (kind, rec["pred"], rec["plan"])
            by.setdefault(rec["pred"], {})[rec["plan"]] = rec
    assert len(by) == 30 and all(len(p) == 2 for p in by.values()), (kind, len(by))
    return {pred: (statistics.median(p["W"]["warm_ms"]) / statistics.median(p[FACT]["warm_ms"]),
                   statistics.median(p["W"]["cold_ms"]) / statistics.median(p[FACT]["cold_ms"])) for pred, p in by.items()}


def main():
    setup()
    import matplotlib.pyplot as plt
    R = runs()
    corpora = ["swea", "smith", "oh", "gym"]
    fig, ax = plt.subplots(figsize=(TEXT_IN, 1.75))
    width = 0.8 / len(corpora)
    rng = np.random.default_rng(7)
    for ci, c in enumerate(corpora):
        name, col = CORPUS[c]
        for ki, kind in enumerate(KINDS):
            x0 = ki - 0.4 + width * (ci + 0.5)
            warm, cold = [], []
            for corpus, seed, res in R:
                if corpus != c:
                    continue
                for pred, (w, cd) in per_predicate(res, kind).items():
                    k = pred.split(":")[1]
                    k = "sub" if k.startswith("sub") else k
                    ax.scatter(x0 + rng.uniform(-width * 0.32, width * 0.32), w, s=5, marker=MARK[k], color=col, alpha=0.55, lw=0)
                    warm.append(w); cold.append(cd)
            if warm:
                ax.boxplot([warm], positions=[x0], widths=width * 0.7, showfliers=False, whis=(0, 100),
                           medianprops=dict(color="black", lw=1.0), boxprops=dict(lw=0.6), whiskerprops=dict(lw=0.5), capprops=dict(lw=0.5))
                ax.scatter([x0], [statistics.median(cold)], marker="D", s=12, facecolor="white", edgecolor=col, lw=0.8, zorder=5)
    ax.axhline(1, color=COLORS["grey"], lw=0.6, ls="--")
    ax.set_yscale("log")
    ax.set_xticks(range(len(KINDS)), [LABEL[k] for k in KINDS])
    ax.set_ylabel("speedup over wide table (wide / FCX)")
    
    handles = [plt.Line2D([], [], marker="s", ls="", color=CORPUS[c][1], label=CORPUS[c][0]) for c in corpora]
    handles += [plt.Line2D([], [], marker=m, ls="", color="black", ms=3, label=l) for m, l in (("o", "sampled substring"), ("^", "regex"), ("s", "fixed needle"))]
    handles += [plt.Line2D([], [], marker="D", ls="", mfc="white", mec="black", mew=0.8, ms=3, label="median cold")]
    ax.legend(handles=handles, ncol=len(handles), loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, handletextpad=0.2, columnspacing=0.8)

    save(fig, "fig3_c2")


if __name__ == "__main__":
    main()
