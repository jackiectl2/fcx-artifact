"""Figure 5: cost of getting raw requests into a queryable, indexed store (claim C4).

(a) ingestion throughput of one process, full parsing against delta parsing of the same requests into the same store, per
context manager and corpus (3,000-trajectory capture samples, 64 interleaved traces), with the share of request bytes delta
parsing had to parse; (b) per-request derivation latency online on the same 17 streams (64 interleaved traces): median,
99th and 99.9th percentile from results/online_v2_summary.json (key "v2" with FCX_INGEST_V2=1, else "v1"); (c) wall-clock seconds (16 threads) to build each text index on the wide request column and on the message dictionary
(seed-1 index runs, policy prefix). The panels come from different runs and are not added up.
E-FAST (FCX_INGEST_V2=1): panel (a) from results/fast_cmp_v2/cap_<c>_s1__ingest_cmp.json (fcx/ingest_cmp.py with the final
ingestion code on the same four capture samples), only after the frozen validator figures/fast_summary.ingest_cmp() passed;
panel (b) unchanged. Runs in the package layout (see tab3_cost.py).
usage: [FCX_INGEST_V2=1] .venv/bin/python figures/fig5_ingest.py
"""
import os
import matplotlib.pyplot as plt

from fig2_workload import MANAGERS
from fig3_c2 import KINDS, LABEL, runs
from style import CORPUS, COLORS, LATE, MARK, RESULTS, TEXT_IN, ink, load, save, setup


def main():
    setup()
    fig = plt.figure(figsize=(TEXT_IN, 2.0))
    outer = fig.add_gridspec(1, 2, width_ratios=[2.0, 1], wspace=0.5)
    left = outer[0].subgridspec(1, 2, width_ratios=[1.3, 0.62], wspace=0.07)
    ax = fig.add_subplot(left[0]); ox = fig.add_subplot(left[1], sharey=ax); bx = fig.add_subplot(outer[1])
    rows = []
    v2 = os.environ.get("FCX_INGEST_V2") == "1"
    if v2:
        import fast_summary
        gate = fast_summary.ingest_cmp()
        assert gate["status"] == "ok" and gate["streams"] == 17, f"ingest_cmp validator: {gate}"
    for c, base in (("swea", LATE), ("smith", LATE), ("oh", LATE), ("gym", RESULTS / "r1_2026-10-24")):
        d = load((RESULTS / "fast_cmp_v2" if v2 else base) / f"cap_{c}_s1__ingest_cmp.json")
        assert not d["files_that_differ"], c
        assert not v2 or d.get("ingest_impl") == "v2", c
        for p, s in d["delta1"]["policies"].items():
            full = d["delta0"]["policies"][p]["ingest"]["logical_mb_per_s"]
            rows.append((c, p, full, s["ingest"]["logical_mb_per_s"], 100 * s["parsed_bytes"] / s["wide_chars"]))
    rows.sort(key=lambda r: (r[0], r[3]))
    for i, (c, p, full, delta, share) in enumerate(rows):
        col = CORPUS[c][1]
        ax.annotate("", xy=(delta, i), xytext=(full, i), arrowprops=dict(arrowstyle="->", color=col, lw=0.8))
        ax.scatter([full], [i], s=9, marker=MARK[c], facecolor="white", edgecolor=col, lw=0.8, zorder=3)
        ax.annotate(f"{share:.1f}%", (delta, i), xytext=(4, 0), textcoords="offset points", va="center", fontsize=6.8, color=ink(col))
    ax.set_yticks(range(len(rows)), [MANAGERS[p] for c, p, *_ in rows], fontsize=6.8)
    for lab, (c, *_) in zip(ax.get_yticklabels(), rows):
        lab.set_color(ink(CORPUS[c][1]))
    ax.invert_yaxis(); ax.set_xlim(40, 450 if v2 else 330)                 # v2 delta parsing reaches 390 MB/s
    ax.set_xlabel("ingestion MB/s per process\n(open: full parse, arrow: delta)")
    ax.set_title("(a) delta parsing; label: share of bytes parsed", loc="left")

    online = load(RESULTS / "online_v2_summary.json")["v2" if v2 else "v1"]          # (b) the same streams online
    assert online["status"] == "ok"
    for i, (c, p, *_) in enumerate(rows):
        s = online["per_stream"][f"cap_{c}_s1/{p}/k64"]
        col = CORPUS[c][1]
        ox.plot([s["p99_us"], s["p999_us"]], [i, i], color=col, lw=0.6, solid_capstyle="butt")
        ox.plot([s["p50_us"], s["p99_us"]], [i, i], color=col, lw=1.8, solid_capstyle="butt")
        ox.scatter([s["p50_us"]], [i], s=9, marker=MARK[c], color=col, edgecolor="white", lw=0.4, zorder=3)
    ox.set_xscale("log"); ox.set_xlim(30, 6000)
    ox.set_xticks([100, 1000], ["100", "1,000"]); ox.tick_params(labelleft=False)
    ox.set_xlabel("µs per request\n(marker p50, bar p99, line p99.9)")
    ox.set_title("(b) online latency", loc="left")

    R = [r for r in runs() if r[1] == "1"]
    kinds = [k for k in KINDS if k != "none"]
    for ci, (c, _, res) in enumerate(R):
        col = CORPUS[c][1]
        v = res["index"]["prefix"]["variants"]
        for ki, kind in enumerate(kinds):
            wide, fac = v[kind]["wide"]["build_s"], v[kind]["factored"]["build_s"]
            y = ki + (ci - (len(R) - 1) / 2) * 0.25
            bx.plot([fac, wide], [y, y], color=col, lw=0.8)
            bx.scatter([wide], [y], s=10, marker=MARK[c], color=col)
            bx.scatter([fac], [y], s=10, marker=MARK[c], facecolor="white", edgecolor=col, lw=0.8)
            bx.annotate(f"{wide / fac:.0f}×", (wide, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=6.8, color=ink(col))
    bx.set_xscale("log"); bx.set_xlim(0.5, 1e4)
    bx.set_yticks(range(len(kinds)), [LABEL[k] for k in kinds]); bx.invert_yaxis()
    bx.set_xlabel("index build wall s, 16 threads (open: dictionary, filled: wide)")
    bx.set_title("(c) text index build, ratio wide / dictionary", loc="left")
    handles = [plt.Line2D([], [], marker=MARK[c], ls="", color=CORPUS[c][1], ms=3, label=CORPUS[c][0]) for c in ("swea", "smith", "oh", "gym")]
    fig.legend(handles=handles, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.22), frameon=False)
    save(fig, "fig5_ingest")


if __name__ == "__main__":
    main()
