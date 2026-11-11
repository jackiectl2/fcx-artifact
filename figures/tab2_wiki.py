"""Table 2 row for the Wikipedia samples (E-WIKI), from the frozen summary only (results/wiki_summary.json, status ok).

The row sums the three samples' checks runs with the Table 2 counting rule (figures/wiki_summary.table2): calls, ctx_any
comparisons, aggregate comparisons, reconstruction results, requests compared bytewise, lifecycle results; configurations is 1
(the block sequence of each revision, no replayed policy).
usage: .venv/bin/python figures/tab2_wiki.py   -> paper/figures/tab2_wiki.tex
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
f = lambda n: f"{n:,}".replace(",", "{,}")


def main():
    s = json.load(open(ROOT / "results" / "wiki_summary.json"))
    assert s["status"] == "ok", s["status"]
    ex = [s["samples"][f"wiki_s{k}"]["exactness_checks"] for k in (1, 2, 3)]
    assert all(e["differences"] == 0 and e["any_over_limit"] == 0 for e in ex)
    tot = {k: sum(e[k] for e in ex) for k in ("calls", "any", "agg", "point", "bytes", "life")}
    assert {k: tot[k] for k in ("any", "agg", "point", "bytes", "life")} == s["ranges"]["exactness_totals"]
    row = f"    Wikipedia & 3 samples & 1 & {f(tot['calls'])} & {f(tot['any'])} & {f(tot['agg'])} & {f(tot['point'])} & {f(tot['bytes'])} & {f(tot['life'])} \\\\\n"
    out = ROOT / "paper" / "figures" / "tab2_wiki.tex"
    out.write_text(row)
    print(row)


if __name__ == "__main__":
    main()
