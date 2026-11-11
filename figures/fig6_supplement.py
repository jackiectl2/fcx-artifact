"""Figure 6: the two supplementary experiments of Section 6.
(a) Q_3 with a 3-gram text index on both sides (mixedidx2, gates of mixed_index_summary.py): one dot per predicate and run, warm speedup
W / I; black bar: median per-predicate speedup of each run; open circle: the earlier index-free run of the same sample, per run. SWE-rebench
has one sample per row, the other corpora three.
(b) Substrings across a message boundary (fallback2, seeds 2 and 3, gates of boundary_summary.py): one dot per needle, warm W / plan, with
P's guard query added to P; black bar: median over needles.
usage: .venv/bin/python figures/fig6_supplement.py
"""
import statistics as st

import numpy as np

from fig2_workload import MANAGERS
from mixed_index_summary import R, RUNS, check, ratios
from style import COLORS, CORPUS, MARK, TEXT_IN, load, save, setup

ORDER = ("swea", "smith", "oh", "gym")
SHORT = {"swea": "SWE-agent", "smith": "SWE-smith", "oh": "SWE-rebench", "gym": "SWE-Gym"}


def indexed():
    """(corpus, policy) -> list of (per-predicate ratios, run median, earlier index-free median)"""
    out = {}
    for c in ORDER:
        for p in RUNS[c]:
            for s in (1, 2, 3):
                if (R / f"cap_{c}_s{s}__mixedidx2_{p}.json").exists():
                    try:
                        t, base, _, _ = check(c, s, p)
                    except AssertionError as e:                  # an unfinished or invalid run stays out, by name
                        print("left out:", e)
                        continue
                    r = ratios(t, "residence", "W", "I")
                    out.setdefault((c, p), []).append((list(r.values()), st.median(r.values()), st.median(ratios(base, "residence", "W", "I").values())))
    return out


def boundary(c, plan):
    xs = []
    for s in (2, 3):
        d = load(R / f"cap_{c}_s{s}__fallback2_prefix.json")
        assert not d.get("mismatch")
        needles = {}
        for x in d["fallback"]["prefix"]["records"]:
            assert x.get("set_equal") is True and "error" not in x, x
            needles.setdefault(x["needle"], {})[x["plan"]] = x
        assert len(needles) == 12
        for v in needles.values():
            t = st.median(v[plan]["warm_ms"]) + (v[plan]["guard_ms"] if plan == "P" else 0)
            xs.append(st.median(v["W"]["warm_ms"]) / t)
    return xs


def main():
    setup()
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 6.8})
    fig, (a, b) = plt.subplots(1, 2, figsize=(TEXT_IN, 1.95), gridspec_kw=dict(wspace=0.42, width_ratios=[1.15, 1]))
    rng = np.random.default_rng(7)

    data, labels, y = indexed(), [], 0
    for c in ORDER:
        for p in RUNS[c]:
            runs = data.get((c, p))
            if not runs:
                continue
            for pts, med, old in runs:
                a.scatter(pts, y + rng.uniform(-0.25, 0.25, len(pts)), s=2.5, marker=MARK[c], color=CORPUS[c][1], alpha=0.4, lw=0)
            for pts, med, old in runs:
                a.plot([med, med], [y - 0.36, y + 0.36], color="black", lw=0.9, zorder=4)
                a.scatter([old], [y], s=9, facecolor="white", edgecolor="black", lw=0.6, zorder=5)
            labels.append((f"{SHORT[c]}: {MANAGERS[p].replace('SWE-agent ', 'SWE-ag. ').replace(' obs.', '').replace('closed ', '')}", c))
            y += 1
    a.axvline(1, color=COLORS["grey"], lw=0.6, ls="--")
    a.set_xscale("log"); a.set_yticks(range(len(labels)), [l for l, _ in labels]); a.invert_yaxis(); a.set_xlim(0.5, 200)
    a.set_xlabel("$Q_3$ speedup, wide / intervals, 3-gram index on both sides"); a.set_title("(a) running example with a text index", loc="left")

    labels, y = [], 0
    for c in ORDER:
        for plan in ("R", "F", "P"):
            pts = boundary(c, plan)
            b.scatter(pts, y + rng.uniform(-0.25, 0.25, len(pts)), s=4, marker=MARK[c], color=CORPUS[c][1], alpha=0.6, lw=0)
            m = st.median(pts)
            b.plot([m, m], [y - 0.38, y + 0.38], color="black", lw=0.9, zorder=4)
            labels.append(f"{SHORT[c]}: {plan}")
            y += 1
    b.axvline(1, color=COLORS["grey"], lw=0.6, ls="--")
    b.set_xscale("log"); b.set_yticks(range(len(labels)), labels); b.invert_yaxis()
    b.set_xlabel("speedup, wide / plan (left of 1: wide is faster)"); b.set_title("(b) substrings across a message boundary", loc="left")
    save(fig, "fig6_supplement")


if __name__ == "__main__":
    main()
