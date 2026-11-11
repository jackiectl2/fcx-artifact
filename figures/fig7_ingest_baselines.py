"""Figure: ingestion baselines and ablations on the same raw requests (experiment E-ING, fcx/ingest_ablate.py).

Rows are the 17 (corpus, request configuration) pairs; inside a row the three capture samples are drawn top to bottom (seed 1,
2, 3), so every one of the 51 streams is shown. Inputs: results/ing_s1_v1 (seed 1), results/ing_s1_rep2 and ing_s1_rep3 (two
more runs of seed 1, timing only) and results/ing_s23_v1 (seeds 2 and 3), the streams of cluster/tasks/s1.txt and s23.txt.
(a) Share of requests whose replayed context differs from the request, for the content-addressed set difference and the
    multiset difference, and the share of requests that hold a message twice. Every exact variant (FCX at every tested edit
    budget, FCX without delta parsing, without order keys, per-call hash arrays, positional delta) reproduces every request in
    every file; the script stops otherwise.
(b) Modeled payload per call relative to FCX (transitions plus checkpoints, fixed-width uncompressed rows), for per-call hash
    arrays and for FCX without order keys (dense positions, same alignment, same checkpoint cadence).
(c) Derivation CPU time relative to FCX (delta parsing, edit budget 64), for FCX without delta parsing and per-call hash
    arrays (full parsing): seed 1 as the median with the min-max of three runs in fixed variant order, seeds 2 and 3 as single
    runs. Derivation = parsing, hashing, dictionary lookup, differencing and building the rows in memory.
The numbers in the text come from figures/ing_ext_summary.py (the frozen aggregator); this script draws the same quantities.
With --impl v2 (E-FAST, refine-logs/E-FAST-PROTOCOL.md) the inputs are results/ing_s1_v2, ing_s1_v2r2, ing_s1_v2r3 and
ing_s23_v2 (the final ingestion code; every file must record ingest_impl "v2"), and panel (c) adds per-call hash arrays derived
with FCX's delta parser and dictionary (hash_array_delta, crosses). Numbers in the text then come from figures/fast_summary.py.
usage: .venv/bin/python figures/fig7_ingest_baselines.py [--summary] [--impl v2]
"""
import statistics, sys

import matplotlib.pyplot as plt
import numpy as np

from style import CORPUS, COLORS, MANAGERS, MARK, RESULTS, TEXT_IN, load, save, setup

ROOT = RESULTS.parent
ORDER = ["swea", "smith", "oh", "gym"]
EXACT = ["fcx_b0", "fcx_b2", "fcx_b8", "fcx_b64", "fcx_b256", "fcx_full", "nokey", "hash_array", "pos_diff"]
OFFSET = {1: 0.27, 2: 0.0, 3: -0.27}                              # vertical position of a sample inside its row
IMPL = sys.argv[sys.argv.index("--impl") + 1] if "--impl" in sys.argv else "v1"
RUNS = {"v1": (["ing_s1_v1", "ing_s1_rep2", "ing_s1_rep3"], ["ing_s23_v1"]), "v2": (["ing_s1_v2", "ing_s1_v2r2", "ing_s1_v2r3"], ["ing_s23_v2"])}[IMPL]
CPU = ("full_cpu", "hash_cpu") + (("hash_delta_cpu",) if IMPL == "v2" else ())


def ratios(d):
    v, n = d["variants"], d["requests"]
    bad = {k: v[k]["wrong"] for k in EXACT if v[k]["wrong"]}
    assert not bad, (d["data"], d["manager"], bad)
    fcx_bytes = v["fcx_b64"]["payload_bytes"] + v["fcx_b64"]["checkpoint_bytes"]
    extra = {}
    if IMPL == "v2":
        assert d.get("ingest_impl") == "v2" and v["hash_array_delta"]["wrong"] == 0, (d["data"], d["manager"])
        extra["hash_delta_cpu"] = v["hash_array_delta"]["cpu_s"] / v["fcx_b64"]["cpu_s"]
    return dict(**extra, dup=d["slice_requests"]["dup"] / n, set_wrong=v["set_diff"]["wrong"] / n, mset_wrong=v["mset_lifo"]["wrong"] / n,
                hash_bytes=v["hash_array"]["payload_bytes"] / fcx_bytes,
                nokey_bytes=(v["nokey"]["payload_bytes"] + v["nokey"]["checkpoint_bytes"]) / fcx_bytes,
                full_cpu=v["fcx_full"]["cpu_s"] / v["fcx_b64"]["cpu_s"], hash_cpu=v["hash_array"]["cpu_s"] / v["fcx_b64"]["cpu_s"])


def rows():
    pairs = [l.split() for l in open(ROOT / "cluster" / "tasks" / "s1.txt") if l.strip()]
    out = []
    for data, m in pairs:
        corpus = data.split("_")[1]
        r = dict(corpus=corpus, manager=m, samples={})
        for seed in (1, 2, 3):
            name = f"{data[:-3]}_s{seed}__{m}.json"
            runs = RUNS[0] if seed == 1 else RUNS[1]
            rr = [ratios(load(RESULTS / run / name)) for run in runs]
            s = dict(rr[0])                                       # deterministic quantities: the first run (all runs agree)
            for k in CPU:
                vals = [x[k] for x in rr]
                s[k], s[k + "_lo"], s[k + "_hi"] = statistics.median(vals), min(vals), max(vals)
            r["samples"][seed] = s
        out.append(r)
    out.sort(key=lambda r: (ORDER.index(r["corpus"]), r["manager"] != "prefix", MANAGERS[r["manager"]]))
    return out


def main():
    R = rows()
    if "--summary" in sys.argv:
        for r in R:
            for seed, s in r["samples"].items():
                print("%-6s s%d %-24s dup %5.1f%% set %5.1f%% mset %5.1f%% | bytes hash %.2fx nokey %.2fx | cpu full %.2fx [%.2f-%.2f] hash %.2fx [%.2f-%.2f]"
                      % (r["corpus"], seed, MANAGERS[r["manager"]], 100 * s["dup"], 100 * s["set_wrong"], 100 * s["mset_wrong"], s["hash_bytes"],
                         s["nokey_bytes"], s["full_cpu"], s["full_cpu_lo"], s["full_cpu_hi"], s["hash_cpu"], s["hash_cpu_lo"], s["hash_cpu_hi"]))
        return
    setup()
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_IN, 0.098 * len(R) + 0.5), sharey=True, gridspec_kw=dict(wspace=0.12))
    y = np.arange(len(R))[::-1]
    for ax in axes:
        for i, r in enumerate(R):                                 # light bands mark the corpora
            if ORDER.index(r["corpus"]) % 2 == 0:
                ax.axhspan(y[i] - 0.5, y[i] + 0.5, color="#f2f2f2", lw=0, zorder=0)
    ms, tick = 6, 4
    ax = axes[0]
    for i, r in enumerate(R):
        col, mk = CORPUS[r["corpus"]][1], MARK[r["corpus"]]
        for seed, s in r["samples"].items():
            yy = y[i] + OFFSET[seed]
            ax.plot([100 * s["dup"]], [yy], marker="|", ms=tick, mew=0.9, color=COLORS["black"], zorder=3)
            ax.scatter([100 * s["set_wrong"]], [yy], s=ms, marker=mk, color=col, lw=0, zorder=4)
            ax.scatter([100 * s["mset_wrong"]], [yy], s=ms, marker=mk, facecolor="white", edgecolor=col, lw=0.6, zorder=4)
    ax.set_xlim(-3, 101)                                            # markers at 0 % stay off the spine
    ax.set_xlabel("requests rebuilt wrongly (%)")
    ax.set_title("(a) content-addressed differences", loc="left")
    h = [plt.Line2D([], [], marker="o", ls="", color="0.3", ms=3.5, label="set difference"),
         plt.Line2D([], [], marker="o", ls="", mfc="white", mec="0.3", ms=3.5, label="multiset difference"),
         plt.Line2D([], [], marker="|", ls="", color="black", ms=5, mew=1.0, label="requests with a repeated message")]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.17 * 11 / len(R) - 0.12), ncol=2, frameon=False,
              handletextpad=0.2, columnspacing=0.8)
    ax.set_yticks(y, [f"{CORPUS[r['corpus']][0]}, {MANAGERS[r['manager']]}" for r in R])
    for k, (key_a, key_b, title, lab_a, lab_b, xlab) in enumerate((
            ("hash_bytes", "nokey_bytes", "(b) modeled payload per call", "per-call hash arrays", "FCX without order keys", "relative to FCX (log scale)"),
            ("hash_cpu", "full_cpu", "(c) derivation CPU time", "per-call hash arrays", "FCX without delta parsing", "relative to FCX (log scale)"))):
        ax = axes[k + 1]
        ax.axvline(1, color="0.4", lw=0.6, ls="--", zorder=1)
        for i, r in enumerate(R):
            col, mk = CORPUS[r["corpus"]][1], MARK[r["corpus"]]
            for seed, s in r["samples"].items():
                yy = y[i] + OFFSET[seed]
                if k == 1 and seed == 1:                          # min-max of the three seed-1 runs
                    for key in (key_a, key_b) + (("hash_delta_cpu",) if IMPL == "v2" else ()):
                        ax.plot([s[key + "_lo"], s[key + "_hi"]], [yy, yy], color=col, lw=0.6, zorder=3)
                ax.scatter([s[key_a]], [yy], s=ms, marker=mk, color=col, lw=0, zorder=4)
                ax.scatter([s[key_b]], [yy], s=ms, marker=mk, facecolor="white", edgecolor=col, lw=0.6, zorder=4)
                if k == 1 and IMPL == "v2":
                    ax.scatter([s["hash_delta_cpu"]], [yy], s=ms * 1.4, marker="x", color=col, lw=0.7, zorder=5)
        ax.set_xscale("log")
        ax.set_xlabel(xlab)
        ax.set_title(title, loc="left")
        h = [plt.Line2D([], [], marker="o", ls="", color="0.3", ms=3.5, label=lab_a),
             plt.Line2D([], [], marker="o", ls="", mfc="white", mec="0.3", ms=3.5, label=lab_b)]
        if k == 1 and IMPL == "v2":
            h[0].set_label("hash arrays, full parsing")
            h.append(plt.Line2D([], [], marker="x", ls="", color="0.3", ms=3.5, mew=0.8, label="hash arrays, delta parsing"))
        ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.17 * 11 / len(R) - 0.12), ncol=1, frameon=False,
                  handletextpad=0.2)
        ax.tick_params(axis="y", length=0)
    keys = ("hash_bytes", "nokey_bytes", "hash_cpu_lo", "hash_cpu_hi", "full_cpu_lo", "full_cpu_hi") + (("hash_delta_cpu_lo", "hash_delta_cpu_hi") if IMPL == "v2" else ())
    vals = [s[k] for r in R for s in r["samples"].values() for k in keys]
    lo, hi = min(vals), max(vals)
    ticks = [t for t in (0.25, 0.5, 1, 2, 4, 8, 16) if lo / 1.3 <= t <= hi * 1.3]
    for ax in axes[1:]:
        ax.set_xlim(lo / 1.3, hi * 1.3)
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.set_xticks(ticks, [f"{t:g}×" for t in ticks])
    for ax in axes:
        ax.set_ylim(-0.6, len(R) - 0.4)
    save(fig, "fig7_ingest_baselines")


if __name__ == "__main__":
    main()
