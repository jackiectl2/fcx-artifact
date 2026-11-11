"""Figure 2: how much a per-call context log repeats, and how context managers change contexts.

(a) logged context characters / distinct message characters, within a trace and across the sample (K2 pilot, the largest
sample per corpus); (b-d) per context manager and corpus on the 3,000-trajectory capture samples: mean context length in
messages, transitions (message occurrences opened or closed) per call, and mean residence of an occurrence in calls.
usage: .venv/bin/python figures/fig2_workload.py
"""
import matplotlib.pyplot as plt

from style import CORPUS, COLORS, LATE, MARK, ROOT, TEXT_IN, load, save, setup

K2 = {"swea": ("k2_pilot_v2_6000.json", "sweagent"), "oh": ("k2_pilot_v2_3000.json", "openhands_rebench"),
      "gym": ("k2_pilot_v2_6000.json", "openhands_swegym")}
MANAGERS = {"prefix": "append only", "swea_last5": "SWE-agent last 5 obs.", "swea_window": "SWE-agent closed window",
            "swea_cache2": "SWE-agent cache marks", "autogen_ht": "AutoGen head+tail", "autogen_buf32": "AutoGen buffer 32",
            "lc_trim32k": "LangChain trim 32k"}


def managers():
    """corpus -> manager -> stats, from the capture-time stores (the full-parse run of ingest_cmp, or the capture's own stats)."""
    out = {}
    for c in ("swea", "smith", "oh"):
        out[c] = load(LATE / f"cap_{c}_s1__ingest_cmp.json")["delta0"]["policies"]
    out["gym"] = load(LATE / "cap_gym_s1__convert_stats.json")["policies"]
    return out


def main():
    setup()
    fig = plt.figure(figsize=(TEXT_IN, 1.7))
    outer = fig.add_gridspec(1, 2, width_ratios=[1, 3.1], wspace=0.95)
    right = outer[1].subgridspec(1, 3, wspace=0.12)
    axes = [fig.add_subplot(outer[0])] + [fig.add_subplot(right[k]) for k in range(3)]
    ax = axes[0]
    for i, (c, (f, key)) in enumerate(K2.items()):
        r = load(ROOT / "results" / "k2_trace" / f)[key]
        w, x = r["redundancy_within_trace_x"], r["redundancy_cross_trace_x"]
        col = CORPUS[c][1]
        ax.plot([w, x], [i, i], color=col, lw=1)
        ax.scatter([x], [i], s=14, marker=MARK[c], color=col, zorder=3)
        # the open (within-trace) marker is drawn last so that its outline stays visible where the two values nearly meet
        ax.scatter([w], [i], s=14, marker=MARK[c], facecolor="white", edgecolor=col, lw=0.9, zorder=4)
        ax.annotate(f"{w:.1f}×", (w, i), xytext=(-3.5, 0), textcoords="offset points", va="center", ha="right", fontsize=6)
        ax.annotate(f"{x:.2f}×", (x, i), xytext=(3.5, 0), textcoords="offset points", va="center", fontsize=6)
    ax.set_yticks(range(len(K2)), [f"{CORPUS[c][0]}\n({r} traj.)" for c, r in zip(K2, ("6,000", "3,000", "6,000"))])
    ax.set_xlim(0, 52); ax.set_ylim(-0.6, len(K2) - 0.4); ax.invert_yaxis()
    ax.set_xlabel("logged / distinct characters")
    ax.set_title("(a) redundancy (open: within trace,\nfilled: across sample)", loc="left")

    M = managers()
    rows = [m for m in MANAGERS if any(m in M[c] for c in M)]
    panels = [("mean_context_len", "(b) context size"), ("transitions_per_span", "(c) transitions / call"),
              ("mean_run_len_spans", "(d) residence, calls")]
    for ax, (field, title) in zip(axes[1:], panels):
        for j, c in enumerate(M):
            for i, m in enumerate(rows):
                if m in M[c]:
                    ax.scatter(M[c][m][field], i + (j - 1.5) * 0.22, s=7, marker=MARK[c], color=CORPUS[c][1], lw=0)   # spaced so equal values stay apart
        ax.set_yticks(range(len(rows)), [MANAGERS[m] for m in rows] if ax is axes[1] else [""] * len(rows))
        ax.set_ylim(-0.6, len(rows) - 0.4); ax.invert_yaxis()
        ax.set_title(title, loc="left")
        ax.grid(axis="x", color=COLORS["grey"], lw=0.3, alpha=0.5)
        vals = [M[c][m][field] for c in M for m in rows if m in M[c]]
        pad = 0.08 * (max(vals) - min(vals))                        # keep markers off the axis spine
        ax.set_xlim(min(vals) - pad, max(vals) + pad)
    handles = [plt.Line2D([], [], marker=MARK[c], ls="", color=CORPUS[c][1], ms=3, label=CORPUS[c][0]) for c in M]
    axes[2].legend(handles=handles, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.42), frameon=False, columnspacing=0.8)
    save(fig, "fig2_workload")


if __name__ == "__main__":
    main()
