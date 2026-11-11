"""Table 2 row for the WildChat samples (E-CHAT), from the reviewed summary only (results/chat_summary.json, status ok).

The row sums the three samples' checks runs (four managers each) with the Table 2 counting rule (figures/wiki_summary.table2):
calls (once per sample, as in the capture-sample rows), ctx_any comparisons, aggregate comparisons, reconstruction results,
requests compared bytewise, lifecycle results; configurations is 4 (prefix and the three framework managers).
usage: .venv/bin/python figures/tab2_chat.py   -> paper/figures/tab2_chat.tex
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
f = lambda n: f"{n:,}".replace(",", "{,}")


def main():
    s = json.load(open(ROOT / "results" / "chat_summary.json"))
    assert s["status"] == "ok", s["status"]
    ex = [s["samples"][f"cap_chat_s{k}"]["exactness_checks"] for k in (1, 2, 3)]
    assert all(e["differences"] == 0 and e["any_over_limit"] == 0 for e in ex)
    tot = {k: sum(e[k] for e in ex) for k in ("calls", "any", "agg", "point", "bytes", "life")}
    assert tot == s["ranges"]["exactness_totals"]
    row = f"    WildChat & 3 samples & 4 & {f(tot['calls'])} & {f(tot['any'])} & {f(tot['agg'])} & {f(tot['point'])} & {f(tot['bytes'])} & {f(tot['life'])} \\\\\n"
    out = ROOT / "paper" / "figures" / "tab2_chat.tex"
    out.write_text(row)
    print(row)


if __name__ == "__main__":
    main()
