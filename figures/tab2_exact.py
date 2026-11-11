"""Exactness table (Table 2 in the PDF): what the checks of bench.py compared, per capture sample and full corpus.

Counting rule (reproduces the columns audited on 2026-10-23): ctx_any plan results other than the reference plan A.exists;
aggregate results other than the reference "A (reference)"; reconstruction plan results; logged requests rebuilt and compared
byte for byte (spans x policies with logged requests); lifecycle plan results. A ctx_any result over the time limit is not a
comparison: it is counted as unverified, not in the ctx_any column. Differences add unequal sets of every completed ctx_any
result (the reference included), differing aggregate rows,
unequal reconstruction and lifecycle results, requests not reproduced, differing rows of the byte checks and nonzero build
checks and a top-level mismatch; the script stops if any is nonzero, if an aggregate record lacks rows_differ, or if a
lifecycle result was verified on a subset of traces only. Inputs are the fixed list RUNS; a full corpus may span one file per policy.
usage: .venv/bin/python figures/tab2_exact.py   -> paper/figures/tab2_exact.tex
"""
from pathlib import Path

from style import LATE, RESULTS, load

OUT = Path(__file__).resolve().parent.parent / "paper" / "figures" / "tab2_exact.tex"
R24 = RESULTS / "r1_2026-10-24"
RUNS = [("SWE-agent", "sample", [LATE / "cap_swea_s1__checks.json"]),
        ("SWE-smith", "sample", [LATE / "cap_smith_s1__checks.json"]),
        ("SWE-rebench", "sample", [R24 / "cap_oh_s1__checks.json"]),
        ("SWE-Gym", "sample", [R24 / "cap_gym_s1__checks.json"]),
        ("SWE-agent", "full", [R24 / "full_swea__c1.json"]),
        ("SWE-smith", "full", [LATE / "full_smith__c1.json"]),
        ("SWE-rebench", "full", [R24 / f"full_oh__c1_{p}.json" for p in ("prefix", "window32", "amortized32", "elide5")]),
        ("SWE-Gym", "full", [R24 / "full_gym__c1.json"])]


R1101 = RESULTS / "r1_2026-11-01"
RECHECK = {"SWE-agent": [R1101 / "full_swea__c1v.json"], "SWE-rebench": [R1101 / f"full_oh__c1v_{p}.json" for p in ("prefix", "window32", "amortized32", "elide5")]}


def rechecked(path_list):
    """Records of a rerun with a longer limit for the equality check, by (policy, set, filter, plan)."""
    out = {}
    for path in path_list:
        r = load(path)
        assert not r.get("mismatch"), path
        for x in r["any"]:
            out[(x["policy"], x["matched"], x["filter"], x["plan"])] = x
    return out


def count(paths, again=()):
    t = dict(policies=set(), calls=set(), any=0, agg=0, point=0, bytes=0, life=0, over=0, diff=0, rechecked=0)
    redo = rechecked(again)
    for path in paths:
        r = load(path)
        t["calls"].add(r["spans"])
        assert not r.get("mismatch"), (path, r.get("mismatch"))
        for x in r["any"]:
            t["policies"].add(x["policy"])
            if "error" in x and x.get("set_equal") is not True:      # a plan that passed the check and then the time limit of its timing is verified
                y = redo.get((x["policy"], x["matched"], x["filter"], x["plan"]))
                if y is not None:                                    # the same comparison repeated on the same stored corpus with a longer limit
                    assert y["set_equal"] is True and y["cardinality"] == x["cardinality"] and y["n_matched"] == x["n_matched"], (x, y)
                    t["rechecked"] += 1
                    t["any"] += 1
                else:
                    t["over"] += x["plan"] != "A.exists"
                continue
            t["diff"] += x["set_equal"] is not True
            if x["plan"] != "A.exists":
                t["any"] += 1
        for x in r["agg"]:
            if x["plan"] != "A (reference)":
                t["agg"] += 1
                t["diff"] += x["rows_differ"]
        t["point"] += len(r["point"])
        t["diff"] += sum(not x["equal"] for x in r["point"])
        assert not any("equal_on_2000_traces" in x for x in r["life"]), f"{path}: lifecycle verified on 2,000 traces only"
        t["life"] += len(r["life"])
        t["diff"] += sum(not x["equal"] for x in r["life"])
        for b in r["bytes"].values():
            t["bytes"] += b["spans"]
            t["diff"] += b["requests_not_reproduced"] + sum(c["rows_differ"] for c in b["checks"].values())
        t["diff"] += sum(abs(v) for v in r["checks"].values())
    assert len(t["calls"]) == 1, f"files of one corpus hold different stores: {paths}"
    t["calls"] = t["calls"].pop()
    t["policies"] = len(t["policies"])
    return t


def num(v):
    return f"{v:,}".replace(",", "{,}")


def main():
    rows = []
    for corpus, kind, paths in RUNS:
        t = count(paths, RECHECK.get(corpus, ()) if kind == "full" else ())
        assert t["over"] == 0, f"{corpus} {kind}: {t['over']} unverified"
        assert t["diff"] == 0, f"{corpus} {kind}: {t['diff']} differences"
        if rows and kind == "full" and "sample" in rows[-1].split("&")[1]:
            rows.append(r"    \midrule")
        rows.append(f"    {corpus} & {kind} & {t['policies']} & {num(t['calls'])} & {num(t['any'])} & {num(t['agg'])} & "
                    f"{num(t['point'])} & {num(t['bytes'])} & {num(t['life'])} \\\\")
        print(corpus, kind, t)
    OUT.write_text("\n".join(rows) + "\n    \\bottomrule\n")


if __name__ == "__main__":
    main()
