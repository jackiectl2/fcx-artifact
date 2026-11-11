"""Break-even query count (text of Section 6.5): per seed-1 append-only index run of the four corpora (the seed-1 runs of
fig3_c2.RUNS, gated like Figure 3) and per configuration (no index, token index, 3-gram index), the extra wall time FCX needs
to make the sample queryable (tab3_cost.totals: ingestion, loads and index build, against insert and index build of the wide
table; summed stage wall times) divided by the per-query saving for the same index kind in the same run, rounded up. Three
saving models: the median over the 30 held-out predicates of (wide - F+I.expand) warm latency (a typical query), the mean of
the same differences (a uniform mix of the 30 predicates), and the median of the cold differences. A configuration in which
FCX is not slower to make queryable needs no queries (0); one whose saving is not positive never breaks even. Ingestion is the one recorded in that run: full
parsing except where the run's ingest record says delta (printed).
With FCX_INGEST_V2=1 the ingestion record comes from the final ingestion code (see tab3_cost.ingest_record).
usage: [FCX_INGEST_V2=1] .venv/bin/python figures/breakeven.py   -> Markdown on stdout
"""
import math
import statistics

from fig3_c2 import FACT, per_predicate, runs
from tab3_cost import CONFIGS, ingest_record, totals
from style import CORPUS


def main():
    print("| Corpus | Ingestion | Configuration | Wall s wide / FCX | Warm saving ms: median / mean; cold median | W / FCX median warm ms "
          "| Break-even queries: warm median / warm mean / cold median |\n|---|---|---|---|---|---|---|")
    for corpus, seed, res in runs():                              # checks both read thresholds at zero
        if seed != "1":
            continue
        out, ing = totals(res)
        for k, name in CONFIGS:
            per_predicate(res, k)                                 # stops unless all 30 held-out predicates are complete and equal
            by, cold = {}, {}
            for r in res["index"]["prefix"]["records"]:
                if r["index"] == k and r["pred"].startswith("eval") and r["plan"] in ("W", FACT):
                    by.setdefault(r["pred"], {})[r["plan"]] = statistics.median(r["warm_ms"])
                    cold.setdefault(r["pred"], {})[r["plan"]] = statistics.median(r["cold_ms"])
            assert len(by) == 30 and all(len(v) == 2 for v in by.values()), (corpus, k)
            d = [v["W"] - v[FACT] for v in by.values()]
            savings = (statistics.median(d), statistics.mean(d), statistics.median(v["W"] - v[FACT] for v in cold.values()))
            extra = out[k]["fcx"][0] - out[k]["wide"][0]
            n = ["0" if extra <= 0 else ("never" if x <= 0 else f"{math.ceil(1000 * extra / x):,}") for x in savings]
            print(f"| {CORPUS[corpus][0]} | {'delta' if ing.get('delta') else 'full'} | {name} | {out[k]['wide'][0]:,.1f} / {out[k]['fcx'][0]:,.1f} "
                  f"| {savings[0]:,.1f} / {savings[1]:,.1f}; {savings[2]:,.1f} | {statistics.median(v['W'] for v in by.values()):,.0f} / "
                  f"{statistics.median(v[FACT] for v in by.values()):,.0f} | {' / '.join(n)} |")


if __name__ == "__main__":
    main()
