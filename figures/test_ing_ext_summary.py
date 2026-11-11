"""Test of ing_ext_summary.py on synthetic trees built only from results/ing_s1_v1 (seed-1 repetition 1, already reported).

Repetitions 2 and 3 are copies of repetition 1 and the seed-2/3 files are copies renamed to the seed-2/3 capture names, so the
clean tree must pass; each damaged tree must produce the status and the blocked claims the protocol requires.
usage: .venv/bin/python figures/test_ing_ext_summary.py
"""
import copy, json, shutil, sys, tempfile
from pathlib import Path

import ing_ext_summary as agg

REAL = Path(__file__).resolve().parent.parent


def build(tmp):
    root = Path(tmp)
    (root / "cluster" / "tasks").mkdir(parents=True)
    for f in ("s1.txt", "s23.txt"):
        shutil.copy(REAL / "cluster" / "tasks" / f, root / "cluster" / "tasks" / f)
    agg.ROOT, agg.RES = root, root / "results"
    res = root / "results"
    entries = []
    for run in agg.SEED1_RUNS + [agg.S23_RUN]:
        (res / run).mkdir(parents=True)
    for data, m, _ in agg.manifest("s1.txt", []):
        d = json.load(open(REAL / "results" / "ing_s1_v1" / f"{data}__{m}.json"))
        for run in agg.SEED1_RUNS:
            json.dump(d, open(res / run / f"{data}__{m}.json", "w"))
            entries.append(dict(run=run, data=data, manager=m, attempt=1, state="COMPLETED", exit_code="0:0", failure_class=None))
    for data, m, s in agg.manifest("s23.txt", []):
        d = json.load(open(REAL / "results" / "ing_s1_v1" / f"{data[:-3]}_s1__{m}.json"))
        d["data"] = str(Path(d["data"]).parent / data)
        json.dump(d, open(res / agg.S23_RUN / f"{data}__{m}.json", "w"))
        entries.append(dict(run=agg.S23_RUN, data=data, manager=m, attempt=1, state="COMPLETED", exit_code="0:0", failure_class=None))
    json.dump(dict(entries=entries), open(res / "ing_ext_attempts.json", "w"))
    return root


def run(root):
    agg.ROOT, agg.RES = root, root / "results"
    sys.argv = ["ing_ext_summary.py", "--out", str(root / "out" / "summary.json")]
    try:
        agg.main()
        code = 0
    except SystemExit as e:
        code = 1 if e.code else 0
    return code, json.load(open(root / "out" / "summary.json"))


def edit(root, run_name, name, fn):
    p = root / "results" / run_name / name
    d = json.load(open(p))
    fn(d)
    json.dump(d, open(p, "w"))


def status(out):
    return {c: v["status"] for c, v in out["claims"].items()}


def main():
    one = "cap_swea_s1__prefix.json"
    with tempfile.TemporaryDirectory() as t:                     # clean tree
        code, out = run(build(t))
        assert code == 0 and out["status"] == "ok" and set(status(out).values()) == {"ok"}, (code, out["status"], status(out))
        assert len(out["c5_per_stream"]) == 51 and out["claims"]["C2"]["append_set_wrong_equals_dup_requests"] is True
        e = out["c5_per_stream"][0]["full_cpu"]
        assert e["kind"] == "three fixed-order descriptive measurements" and len(e["values"]) == 3 and e["min"] == e["max"] == e["median"]
        assert out["claims"]["C6"]["append_b0_fallback_requests_zero"] is True
    with tempfile.TemporaryDirectory() as t:                     # a missing file
        root = build(t)
        (root / "results" / "ing_s1_rep3" / one).unlink()
        code, out = run(root)
        assert code == 1 and out["status"] == "incomplete" and set(status(out).values()) == {"blocked"} and len(out["missing"]) == 1
        assert all(set(v) == {"status", "reasons"} for v in out["claims"].values())
    with tempfile.TemporaryDirectory() as t:                     # a non-timing difference inside one variant of repetition 2
        root = build(t)
        edit(root, "ing_s1_rep2", one, lambda d: d["variants"]["nokey"].update(records=d["variants"]["nokey"]["records"] + 1))
        code, out = run(root)
        st = status(out)
        assert code == 1 and out["status"] == "blocked" and st["C1"] == st["C4"] == "blocked" and st["C2"] == st["C3"] == st["C5"] == st["C6"] == "ok", st
        assert out["nondeterministic"][0]["paths"] == [".variants.nokey.records"]
    with tempfile.TemporaryDirectory() as t:                     # a non-timing difference outside the variants blocks everything
        root = build(t)
        edit(root, "ing_s1_rep3", one, lambda d: d.update(distinct_messages=d["distinct_messages"] + 1))
        code, out = run(root)
        assert code == 1 and set(status(out).values()) == {"blocked"}
    with tempfile.TemporaryDirectory() as t:                     # timing fields inside a list are ignored at any depth
        root = build(t)
        for r, x in (("ing_s1_v1", 1.0), ("ing_s1_rep2", 2.0), ("ing_s1_rep3", 3.0)):
            edit(root, r, one, lambda d, x=x: d.update(extra=[{"cpu_s": x, "n": 5, "inner": [{"wall_s": x}]}]))
        code, out = run(root)
        assert code == 0 and out["status"] == "ok", out["problems"]
        edit(root, "ing_s1_rep2", one, lambda d: d["extra"][0].update(n=6))
        code, out = run(root)
        assert code == 1 and out["nondeterministic"][0]["paths"] == [".extra[0].n"]
    with tempfile.TemporaryDirectory() as t:                     # a correctness failure in repetition 3 only
        root = build(t)
        edit(root, "ing_s1_rep3", one, lambda d: d["variants"]["hash_array"].update(wrong=1))
        code, out = run(root)
        st = status(out)
        assert code == 1 and st["C1"] == st["C3"] == st["C5"] == "blocked" and st["C2"] == st["C4"] == st["C6"] == "ok", st
        assert any(e.get("status") == "blocked" for e in out["c5_per_stream"]) and len(out["c5_per_stream"]) == 51
    with tempfile.TemporaryDirectory() as t:                     # a rerun after a completed or failed attempt is not allowed
        root = build(t)
        led = json.load(open(root / "results" / "ing_ext_attempts.json"))
        first = copy.deepcopy(led["entries"][0]); first.update(state="FAILED", exit_code="1:0", failure_class="failed")
        led["entries"][0]["attempt"] = 2
        led["entries"].append(first)
        json.dump(led, open(root / "results" / "ing_ext_attempts.json", "w"))
        code, out = run(root)
        assert code == 1 and out["status"] == "incomplete" and any("rerun after a failed" in p for p in out["ledger_problems"])
        first.update(state="NODE_FAIL", exit_code="0:0", failure_class="infrastructure")
        led["entries"][-1] = first
        json.dump(led, open(root / "results" / "ing_ext_attempts.json", "w"))
        code, out = run(root)
        assert code == 0 and out["status"] == "ok", out.get("ledger_problems")
    with tempfile.TemporaryDirectory() as t:                     # a file copied under the wrong name
        root = build(t)
        edit(root, agg.S23_RUN, "cap_oh_s2__prefix.json", lambda d: d.update(data=str(Path(d["data"]).parent / "cap_oh_s3")))
        code, out = run(root)
        assert code == 1 and set(status(out).values()) == {"blocked"} and out["c5_per_stream"] == []
    with tempfile.TemporaryDirectory() as t:                     # a whole result directory missing
        root = build(t)
        shutil.rmtree(root / "results" / "ing_s1_rep2")
        code, out = run(root)
        assert code == 1 and out["status"] == "incomplete" and len(out["missing"]) == 17
    with tempfile.TemporaryDirectory() as t:                     # a partial result file left in a result directory
        root = build(t)
        (root / "results" / "ing_s23_v1" / "cap_oh_s2__prefix.json.part").write_text("{")
        code, out = run(root)
        assert code == 1 and out["status"] == "incomplete" and out["partial_files"] == ["ing_s23_v1/cap_oh_s2__prefix.json.part"]
    for damage in (lambda d: d.clear() or d.update(x=[1]),                                   # not the expected object
                   lambda d: d.pop("requests"),                                               # missing top-level count
                   lambda d: d["variants"]["hash_array"].update(fallback_requests="7"),       # field a non-log variant does not have
                   lambda d: d["variants"]["nokey"].update(records=1.5),                      # count that is not an integer
                   lambda d: d["slice_requests"].update(dup=True),                            # boolean as a count
                   lambda d: d["variants"]["fcx_b64"].update(cpu_s=float("inf"))):            # non-finite timing (written as Infinity)
        with tempfile.TemporaryDirectory() as t:
            root = build(t)
            edit(root, "ing_s1_rep2", one, damage)
            code, out = run(root)
            assert code == 1 and out["status"] == "blocked" and set(status(out).values()) == {"blocked"} and out["c5_per_stream"] == [], out.get("data_errors")
            assert list(out["data_errors"]) == ["ing_s1_rep2/cap_swea_s1__prefix"]
    with tempfile.TemporaryDirectory() as t:                     # an fcx_b64 failure also blocks C2 (its slices come from that replay)
        root = build(t)
        edit(root, "ing_s1_rep3", one, lambda d: d["variants"]["fcx_b64"].update(wrong=1))
        code, out = run(root)
        st = status(out)
        assert code == 1 and set(st.values()) == {"blocked"}, st
    with tempfile.TemporaryDirectory() as t:                     # a nokey-only difference leaves C5 and every stream's timing usable
        root = build(t)
        edit(root, "ing_s1_rep3", one, lambda d: d["variants"]["nokey"].update(wrong=1))
        code, out = run(root)
        st = status(out)
        assert code == 1 and st["C1"] == st["C4"] == "blocked" and st["C5"] == "ok" and all(e["status"] == "ok" for e in out["c5_per_stream"]), st
    with tempfile.TemporaryDirectory() as t:                     # a malformed ledger entry
        root = build(t)
        led = json.load(open(root / "results" / "ing_ext_attempts.json"))
        led["entries"][3]["attempt"] = "1"
        json.dump(led, open(root / "results" / "ing_ext_attempts.json", "w"))
        code, out = run(root)
        assert code == 1 and out["status"] == "incomplete" and any("malformed" in p for p in out["ledger_problems"])
    with tempfile.TemporaryDirectory() as t:                     # a malformed manifest
        root = build(t)
        with open(root / "cluster" / "tasks" / "s23.txt", "a") as f:
            f.write("cap_oh_s4 prefix extra\n")
        code, out = run(root)
        assert code == 1 and out["status"] == "incomplete" and out["manifest_errors"]
    print("TEST_ING_EXT_SUMMARY_OK")


if __name__ == "__main__":
    main()
