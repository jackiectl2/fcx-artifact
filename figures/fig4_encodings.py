"""Figure 4: what the membership encoding contributes once messages are stored once (claim C3).

SWE-agent sample (3,000 random trajectories, seed 1, 16 threads, read thresholds at zero), four context managers:
(a) ctx_any warm latency per filter kind, geometric mean over the matched-set regimes of the per-regime median (as in
fcx/r1_report.py), for the FCX default plan and the normalized membership table (best of its two sort orders);
(b) lifecycle queries L1-L4 on interval rows and on the normalized table; (c) rows written per call: transitions against
normalized rows, three capture samples; (d) reconstruction of 1,000 calls per encoding.
usage: .venv/bin/python figures/fig4_encodings.py
"""
import math
import statistics

import matplotlib.pyplot as plt

from fig2_workload import MANAGERS
from style import CORPUS, COLORS, LATE, TEXT_IN, ink, load, save, setup

CHECKS = LATE / "cap_swea_s1__checks.json"
POL = ["prefix", "swea_last5", "swea_window", "autogen_ht"]
MARK = {"prefix": "o", "swea_last5": "s", "swea_window": "^", "autogen_ht": "D"}
med = lambda r: statistics.median(r["warm_ms"]) if "error" not in r and r.get("warm_ms") else None
gm = lambda xs: math.exp(statistics.fmean(map(math.log, xs)))


def main():
    setup()
    plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 6.8,
                         "ytick.labelsize": 6.8, "legend.fontsize": 6.8})     # four panels in one text width: larger type
    res = load(CHECKS)
    # an empty spacer column before (c) holds its long manager labels clear of the markers of (b)
    fig, axes = plt.subplots(1, 5, figsize=(TEXT_IN, 2.0), gridspec_kw=dict(wspace=1.0, width_ratios=[1, 1, 0.55, 1.1, 0.8]))
    axes[2].axis("off")
    axes = [axes[0], axes[1], axes[3], axes[4]]

    ax = axes[0]                                                    # (a) ctx_any
    filters = list(res["filters"])
    for i, flt in enumerate(filters):
        default = "A.exists" if flt == "trace" else "I.expand"
        for j, p in enumerate(POL):
            by = {}
            for r in res["any"]:
                if r["policy"] == p and r["filter"] == flt and med(r) is not None:
                    by.setdefault(r["matched"], {})[r["plan"]] = med(r)
            fcx = gm([v[default] for v in by.values() if default in v])
            nor = gm([min(v[x] for x in ("N.inv", "N.fwd") if x in v) for v in by.values() if "N.inv" in v or "N.fwd" in v])
            y = i + (j - 1.5) * 0.17
            ax.plot([fcx, nor], [y, y], color=COLORS["grey"], lw=0.6)
            ax.scatter([fcx], [y], marker=MARK[p], s=9, color=COLORS["blue"], zorder=3)
            ax.scatter([nor], [y], marker=MARK[p], s=9, facecolor="white", edgecolor=COLORS["purple"], lw=0.8, zorder=3)
    ax.set_yticks(range(len(filters)), [{"none": "no filter", "repo": "repository", "trace": "one trace"}.get(f, f) for f in filters])
    ax.invert_yaxis(); ax.set_xlabel("ctx_any warm ms"); ax.set_title("(a) context membership", loc="left")

    ax = axes[1]                                                    # (b) lifecycle
    queries = sorted({r["query"] for r in res["life"]})
    for i, q in enumerate(queries):
        for j, p in enumerate(POL):
            v = {r["plan"]: med(r) for r in res["life"] if r["policy"] == p and r["query"] == q}
            y = i + (j - 1.5) * 0.17
            ax.plot([v["I"], v["N"]], [y, y], color=COLORS["grey"], lw=0.6)
            ax.scatter([v["I"]], [y], marker=MARK[p], s=9, color=COLORS["blue"], zorder=3)
            ax.scatter([v["N"]], [y], marker=MARK[p], s=9, facecolor="white", edgecolor=COLORS["purple"], lw=0.8, zorder=3)
    ax.set_yticks(range(len(queries)), [q.split(" ")[0] + " " + {"L1": "volume", "L2": "evictions", "L3": "re-entries", "L4": "residence"}[q.split(" ")[0]] for q in queries])
    ax.set_xscale("log"); ax.invert_yaxis(); ax.set_xlabel("warm ms"); ax.set_title("(b) lifecycle queries", loc="left")

    ax = axes[2]                                                    # (c) rows written per call
    rows = []
    for c in ("swea", "smith", "oh"):
        for p, s in load(LATE / f"cap_{c}_s1__ingest_cmp.json")["delta0"]["policies"].items():
            rows.append((c, p, s["transitions_per_span"], s["mean_context_len"]))
    rows.sort(key=lambda r: r[3] / r[2])
    for i, (c, p, t, n) in enumerate(rows):
        ax.plot([t, n], [i, i], color=CORPUS[c][1], lw=0.6)
        ax.scatter([t], [i], s=8, color=COLORS["blue"], zorder=3)
        ax.scatter([n], [i], s=8, facecolor="white", edgecolor=COLORS["purple"], lw=0.8, zorder=3)
    short = {"swea": "agent", "smith": "smith", "oh": "rebench"}                # the row label names the corpus, so rows stay apart in greyscale
    ax.set_yticks(range(len(rows)), [f"{short[c]}: " + MANAGERS[p].replace("SWE-agent ", "SWE-ag. ").replace(" obs.", "").replace("closed ", "").replace(" 32k", "").replace(" marks", "").replace(" head+tail", " h+t").replace(" buffer 32", " buf.") for c, p, t, n in rows], fontsize=6.8)
    for lab, (c, *_ ) in zip(ax.get_yticklabels(), rows):
        lab.set_color(ink(CORPUS[c][1]))
    ax.set_xscale("log"); ax.set_xticks([2, 5, 10, 20, 50], ["2", "5", "10", "20", "50"]); ax.minorticks_off()
    ax.set_xlabel("rows per call"); ax.set_title("(c) rows written (3 corpora)", loc="left")

    ax = axes[3]                                                    # (d) reconstruction of 1,000 calls
    plans = [("A", "arrays"), ("N", "normalized"), ("I", "intervals")]
    for j, p in enumerate(POL):
        for i, (plan, _) in enumerate(plans):
            v = [med(r) for r in res["point"] if r["policy"] == p and r["n"] == 1000 and r["plan"] == plan]
            if v:
                ax.scatter(v[0], i + (j - 1.5) * 0.17, marker=MARK[p], s=9, color=COLORS["black"])
    ax.set_yticks(range(len(plans)), [n for _, n in plans]); ax.invert_yaxis()
    ax.set_xlabel("warm ms"); ax.set_title("(d) rebuild 1,000 calls", loc="left")

    for ax, log in zip(axes, (False, True, True, False)):          # keep markers off the axis spines
        xs = [p[0] for col in ax.collections for p in col.get_offsets()]
        lo, hi = min(xs), max(xs)
        if log:
            ax.set_xlim(lo / 1.25, hi * 1.25)
        else:
            pad = 0.07 * (hi - lo)
            ax.set_xlim(lo - pad, hi + pad)
    handles = [plt.Line2D([], [], marker="o", ls="", color=COLORS["blue"], ms=3, label="FCX (intervals / arrays)"),
               plt.Line2D([], [], marker="o", ls="", mfc="white", mec=COLORS["purple"], ms=3, label="normalized membership")]
    handles += [plt.Line2D([], [], marker=MARK[p], ls="", color="black", ms=3, label=MANAGERS[p]) for p in POL]
    fig.legend(handles=handles, ncol=6, loc="lower center", bbox_to_anchor=(0.5, -0.2), frameon=False, columnspacing=0.8)
    save(fig, "fig4_encodings")


if __name__ == "__main__":
    main()
