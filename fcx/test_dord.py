"""Synthetic test of the index stage's two dictionary layouts (bench.dict_load, bench.dict_checks) on a job-local ClickHouse
server (run through cluster/fcx_bench.sbatch with PYSCRIPT=fcx/test_dord.py; the driver flags it passes are ignored).
Interleaved traces with contiguous span ranges, messages shared across traces, re-entries (several intervals of one message in
one trace) and intervals that start at or after the cutoff S. For both layouts the loaded table must hold exactly the messages
open in a span below S with their texts, and for `first` the first span of each message, with sorting key (first_span, msg_id);
dict_checks must pass on it and must catch each mutation: a missing row, an extra row, a wrong first span, a changed text."""
import argparse, json, random, time

import bench

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True); ap.add_argument("--store", required=True); ap.add_argument("--out", required=True)
a, _ = ap.parse_known_args()
ch = bench.CH(a.port, 2)
for _ in range(120):
    try:
        ch.val("SELECT 1"); break
    except (OSError, RuntimeError):
        time.sleep(1)
else:
    raise SystemExit("ClickHouse server did not come up")

rng = random.Random(5)
msgs = {m: f"msg-{m}-" + "".join(rng.choice("abcde") for _ in range(rng.randint(5, 60))) for m in range(300)}
iv, lo, occ = [], 0, 0
for t in range(8):                                                # trace t owns spans lo .. lo + calls - 1
    calls = rng.randint(20, 60)
    for _ in range(rng.randint(30, 80)):
        m = rng.randrange(300)
        f = lo + rng.randrange(calls); e = rng.randint(f, lo + calls - 1)
        iv.append((m, t, occ, occ * 1000, f, e)); occ += 1
        if rng.random() < 0.3 and e + 2 < lo + calls:            # the same message re-enters the same trace later
            f2 = rng.randint(e + 2, lo + calls - 1); iv.append((m, t, occ, occ * 1000, f2, rng.randint(f2, lo + calls - 1))); occ += 1
    lo += calls
S = int(lo * 0.8)
first = {}
for m, t, o, k, f, e in iv:
    if f < S:
        first[m] = min(first.get(m, f), f)
ch.q("DROP TABLE IF EXISTS msg SYNC"); ch.q("DROP TABLE IF EXISTS i_t SYNC")
ch.q("CREATE TABLE msg (msg_id UInt32, text String) ENGINE = MergeTree ORDER BY msg_id")
ch.q("INSERT INTO msg FORMAT JSONEachRow\n" + "\n".join(json.dumps(dict(msg_id=m, text=x)) for m, x in msgs.items()))
cols, order = bench.DDL["i"]
ch.q(f"CREATE TABLE i_t ({cols}) ENGINE = MergeTree ORDER BY ({order})")
ch.q("INSERT INTO i_t FORMAT JSONEachRow\n" + "\n".join(json.dumps(dict(msg_id=m, trace_id=t, occ=o, ord=k, span_from=f, span_to=e)) for m, t, o, k, f, e in iv))
assert any(f >= S for _, _, _, _, f, _ in iv) and any(sum(1 for x in iv if x[0] == m and x[4] < S) > 1 for m in first), "the fixture lacks a case"

out = dict(messages=len(first), intervals=len(iv), S=S)
for layout in ("id", "first"):
    kind, sel = bench.dict_load("t", S, layout)
    bench.load(ch, a.store, "mx_t", kind, sel, 8, optimize=False)
    c = bench.dict_checks(ch, "mx_t", "t", S, layout)
    assert c["ok"], (layout, c)
    if layout == "first":
        got = {int(r[0]): (r[1], int(r[2])) for r in (l.split("\t") for l in ch.val("SELECT msg_id, text, first_span FROM mx_t").splitlines())}
        assert got == {m: (msgs[m], first[m]) for m in first}, "first: wrong relation or first spans"
        key = ch.val("SELECT sorting_key FROM system.tables WHERE database = currentDatabase() AND name = 'mx_t'")
        assert key.replace(" ", "") == "first_span,msg_id", key
        stored = [int(x) for x in ch.val("SELECT msg_id FROM mx_t ORDER BY first_span, msg_id").split()]
        assert stored == sorted(first, key=lambda m: (first[m], m))
    else:
        got = {int(r[0]): r[1] for r in (l.split("\t") for l in ch.val("SELECT msg_id, text FROM mx_t").splitlines())}
        assert got == {m: msgs[m] for m in first}, "id: wrong relation"
    victim = min(first)
    caught = {}
    fs = "first_span, " if layout == "first" else ""
    for name, bad in (("missing", f"SELECT * FROM ({sel}) WHERE msg_id != {victim}"),       # damaged copies of the same SELECT (a sorting-key
                      ("text_differs", f"SELECT msg_id, {fs}if(msg_id = {victim}, concat(text, 'x'), text) AS text FROM ({sel})"),   # column cannot be updated)
                      ("wrong_first_span", f"SELECT msg_id, if(msg_id = {victim}, first_span + 1, first_span) AS first_span, text FROM ({sel})")):
        if name == "wrong_first_span" and layout == "id":
            continue
        bench.load(ch, a.store, "mx_t", kind, bad, 8, optimize=False)
        c = bench.dict_checks(ch, "mx_t", "t", S, layout)
        caught[name] = (c[name], c["ok"])
        assert c[name] >= 1 and not c["ok"], (layout, name, c)
    bench.load(ch, a.store, "mx_t", kind, sel, 8, optimize=False)
    extra = max(msgs) + 1
    ch.q(f"INSERT INTO mx_t SELECT {extra}, " + ("0, " if layout == "first" else "") + "'stray'")
    c = bench.dict_checks(ch, "mx_t", "t", S, layout)
    assert c["extra"] == 1 and not c["ok"], (layout, c)
    caught["extra"] = (c["extra"], c["ok"])
    out[layout] = dict(kind=kind, checks_pass=True, mutations_caught=caught)
json.dump(out, open(a.out, "w"), indent=1)
print("TEST_DORD_OK", json.dumps(out))
