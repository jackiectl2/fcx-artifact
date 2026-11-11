"""Cost table (Table 4 in the PDF; file name kept): cost of making one capture sample queryable, wide table against FCX, per configuration (no index, token index,
3-gram index), from one seed-1 index run per corpus (policy prefix, 16 threads). With FCX_INGEST_V2 unset, every total adds up records of one result file (with it, see E-FAST below).

Wide: insert of the requests (`index.prefix.load.wx_prefix`) plus, for an indexed configuration, the index build on the wide
column (`index.prefix.variants.<kind>.wide`). FCX: ingestion (`convert.policies.prefix.ingest`, one Python process), the
loads of the log, checkpoints, intervals and arrays (`load.{d,ckpt,i,a}_prefix`), the dictionary as loaded for that
configuration (`index.prefix.load.mx_prefix/<kind>`, at the granularity the queries of that configuration use) and, for an
indexed configuration, the index build on it (`index.prefix.variants.<kind>.factored`). Wall time per table is insert plus
optimize. MB = 10^6 bytes on disk, index included.

Delta rows (FCX columns): the no-index totals, wall and CPU, from the index run on the copy of the same sample that was
ingested with delta parsing (`cap_<c>_s1d__index_prefix.json`, PAIRS below). The pairing is checked: threads, read settings,
traces, spans, distinct messages, requests, rows of every FCX table, and the dictionary granularity and index parameter of
the no-index configuration (calibration chooses them per run; indexed configurations are not compared because their
choices can differ between the two runs). A dash where that run does not exist yet, or where its no-index layout differs from
the full-parse run's (its total is then not comparable; the ingestion times of both runs are still printed).
E-FAST (FCX_INGEST_V2=1 in the environment; refine-logs/E-FAST-PROTOCOL.md): FCX's ingestion is measured separately with the
final implementation on the same sample (results/fast_cmp_v2/<sample>__ingest_cmp.json, fcx/ingest_cmp.py: full parsing
`delta0` for a full-parse run, delta parsing `delta1` for a delta-parsed run, policy prefix, the first manager ingested), and
the load, index-build and size components come from the corresponding index run; full and delta parsing produced identical
logical output tables. The totals are componentwise, not one measured end-to-end run. The replacement requires the frozen
validator figures/fast_summary.ingest_cmp() to pass and checks the same source sample, traces, spans, distinct messages,
requests and parse mode as the run. breakeven.py, which imports totals(), follows the same switch.
This script, fig5_ingest.py and breakeven.py run in the package layout (figures/ with the merged style.py, fig2_workload.py
and fig3_c2.py, and the earlier runs under results/r1_*); tools/smoke_v2_figures.py checks both modes there.
usage: [FCX_INGEST_V2=1] .venv/bin/python figures/tab3_cost.py
"""
import os
from pathlib import Path

from style import LATE, RESULTS, load

INGEST_V2 = os.environ.get("FCX_INGEST_V2") == "1"
_GATE = None


def ingest_record(d):
    """The ingestion record (wall_s, cpu_s, ...) of index run d: its own, or with FCX_INGEST_V2=1 the final code's (checked)."""
    ing = d["convert"]["policies"]["prefix"]["ingest"]
    if not INGEST_V2:
        return ing
    global _GATE
    if _GATE is None:
        import fast_summary
        _GATE = fast_summary.ingest_cmp()
    assert _GATE["status"] == "ok", f"ingest_cmp validator: {_GATE}"
    sample = Path(d["convert"]["src"]).name
    cmp_ = load(RESULTS / "fast_cmp_v2" / f"{sample}__ingest_cmp.json")
    assert list(cmp_["delta0"]["policies"])[0] == "prefix" == list(d["convert"]["policies"])[0], f"{sample}: prefix is not the first manager"
    key = "delta1" if ing.get("delta") else "delta0"
    st = cmp_[key]
    new = st["policies"]["prefix"]["ingest"]
    assert cmp_.get("ingest_impl") == "v2" and not cmp_["files_that_differ"], f"{sample}: not a clean v2 comparison"
    assert Path(st["src"]).name == sample, (sample, st["src"])
    for k in ("traces", "spans", "distinct_messages"):
        assert st[k] == d["convert"][k], (sample, k, st[k], d["convert"][k])
    assert new["requests"] == ing["requests"] and int(new.get("delta", 0)) == int(ing.get("delta", 0)), (sample, new, ing)
    return new

OUT = Path(__file__).resolve().parent.parent / "paper" / "figures" / "tab3_cost.tex"
FCX_TABLES = ("d_prefix", "ckpt_prefix", "i_prefix", "a_prefix")
CONFIGS = (("none", "no index"), ("token", "token index"), ("ngrams3", "3-gram index"))
CORPORA = ("swea", "oh")
PAIRS = {c: (LATE / f"cap_{c}_s1__index_prefix.json", RESULTS / "r1_2026-10-24" / f"cap_{c}_s1d__index_prefix.json") for c in CORPORA}


def wall(t):
    return t["insert_s"] + t.get("optimize_s", 0.0)


def totals(d):
    """{config: dict(wide=(wall, cpu, mb), fcx=(wall, cpu, mb))} for one index run, and its ingestion record."""
    ix = d["index"]["prefix"]
    ing = ingest_record(d)
    w = ix["load"]["wx_prefix"]
    base = [d["load"][t] for t in FCX_TABLES]
    out = {}
    for k, _ in CONFIGS:
        fcx = base + [ix["load"][f"mx_prefix/{k}"]]
        v = ix["variants"][k]
        wb, fb = (v["wide"], v["factored"]) if k != "none" else (dict(build_s=0.0, build_cpu_s=0.0, index_bytes=0),) * 2
        out[k] = dict(wide=(wall(w) + wb["build_s"], w["cpu_s"] + wb["build_cpu_s"], (w["bytes_on_disk"] + wb["index_bytes"]) / 1e6),
                      fcx=(ing["wall_s"] + sum(map(wall, fcx)) + fb["build_s"], ing["cpu_s"] + sum(t["cpu_s"] for t in fcx) + fb["build_cpu_s"],
                           (sum(t["bytes_on_disk"] for t in fcx) + fb["index_bytes"]) / 1e6))
    return out, ing


def check_pair(a, b, path):
    """The delta-parsed copy must hold the same sample under the same settings. Returns False (no delta cells for that corpus)
    when only the no-index layout differs, since calibration chose it per run; stops on any other difference."""
    checks = {"source": lambda d: Path(d["convert"]["src"]).name, "threads": lambda d: d["meta"]["threads"], "settings": lambda d: d["meta"].get("settings"),
              "traces": lambda d: d["convert"]["traces"], "spans": lambda d: d["convert"]["spans"],
              "distinct": lambda d: d["convert"]["distinct_messages"], "requests": lambda d: d["convert"]["policies"]["prefix"]["ingest"]["requests"],
              "rows": lambda d: [d["load"][t]["rows"] for t in FCX_TABLES] + [d["index"]["prefix"]["load"][f"mx_prefix/{k}"]["rows"] for k, _ in CONFIGS]}
    checks["no-index layout"] = lambda d: (d["index"]["prefix"]["load"]["mx_prefix/none"]["granularity"], d["index"]["prefix"]["variants"]["none"]["factored"].get("param"))
    bad = [n for n, f in checks.items() if f(a) != f(b)]
    assert bad in ([], ["no-index layout"]), f"{path} differs from the full-parse run in {bad}"
    assert b["convert"]["policies"]["prefix"]["ingest"].get("delta") == 1, f"{path} was not ingested with delta parsing"
    assert not a["convert"]["policies"]["prefix"]["ingest"].get("delta"), "the reference run must use full parsing"
    if bad:
        print(path.name, "no-index layout differs (granularity, param):", checks["no-index layout"](a), "vs", checks["no-index layout"](b))
    return not bad


def num(x):
    return f"{x:,.1f}".replace(",", "{,}")


def main():
    cols, delta = [], []
    for c in CORPORA:
        full, dpath = PAIRS[c]
        a = load(full)
        t, ing = totals(a)
        cols += [{k: t[k]["wide"] for k, _ in CONFIGS}, {k: t[k]["fcx"] for k, _ in CONFIGS}]
        if dpath.exists():
            b = load(dpath)
            same = check_pair(a, b, dpath)
            td, ding = totals(b)
            delta += [None, {k: td[k]["fcx"] for k, _ in CONFIGS} if same else None]
            print(c, "ingestion wall/cpu s: full", ing["wall_s"], ing["cpu_s"], "delta", ding["wall_s"], ding["cpu_s"])
        else:
            delta += [None, None]
            print(c, "no delta-parsed run yet:", dpath)
    def row(group, metric, vals):
        """Bold marks the lower value of each wide / FCX pair; FCX columns carry a light identity shade."""
        out = []
        for i in range(0, 4, 2):
            w, f = vals[i], vals[i + 1]
            ws, fs = (num(w) if w is not None else "n/a"), (num(f) if f is not None else "--")   # wide: no delta parsing; FCX: not comparable
            if w is not None and f is not None and ws != fs:
                ws, fs = (rf"\textbf{{{ws}}}", fs) if w < f else (ws, rf"\textbf{{{fs}}}")
            out += [ws, rf"\cellcolor{{blue!8}}{fs}"]
        return f"    {group:<16} & {metric:<6} & " + " & ".join(out) + r" \\"
    lines = []
    for k, name in CONFIGS:
        for j, metric in enumerate(("wall s", "CPU s", "MB")):
            lines.append(row(name if j == 0 else "", metric, [x[k][j] for x in cols]))
        lines.append(r"    \addlinespace[2pt]")
    for j, metric in enumerate(("wall s", "CPU s")):
        lines.append(row("no index, delta" if j == 0 else "", metric, [x["none"][j] if x else None for x in delta]))
    body = "\n".join(lines)
    OUT.write_text(body + "\n    \\bottomrule\n")                # inside the input: a rule after \input would be misplaced
    for c, x in zip(("swea wide", "swea fcx", "oh wide", "oh fcx"), cols):
        print(c, {k: tuple(round(y, 1) for y in v) for k, v in x.items()})
    print("wrote", OUT)


if __name__ == "__main__":
    main()
