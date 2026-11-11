"""Synthetic test of duck_bench.py: random traces whose contexts evolve (appends, drops, re-entries, duplicates), ingested with the
paper's Ingester into transitions; truth counts by brute force over the serialized requests. Every plan must equal the truth, a
wrong truth must stop the run. usage: python test_duck_bench.py  (prints TEST_DUCK_BENCH_OK)"""
import json, os, random, subprocess, sys, tempfile

import orjson, pyarrow as pa, pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ingest import Dictionary, Ingester                          # noqa: E402

U32 = pa.uint32()


def main():
    rnd = random.Random(7)
    words = ["alpha beta", "gamma delta", "Traceback (most", "error 42 here", "no match", "delta gamma", "x = 1"]
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(f"{d}/cap_test_s1/prefix")
        dic = Dictionary(f"{d}/dict.bin"); ing = Ingester(dic)
        wide, trans, traces, span = [], [], [], 0
        for t in range(40):
            ctx, lo = [], span
            for k in range(rnd.randint(1, 12)):
                op = rnd.random()
                if op < 0.5 or not ctx:
                    ctx.append({"content": rnd.choice(words) + f" #{rnd.randint(0, 30)}", "role": rnd.choice(["user", "assistant"])})
                elif op < 0.7:
                    ctx.pop(rnd.randrange(len(ctx)))
                elif op < 0.85:
                    ctx.append(dict(rnd.choice(ctx)))                 # duplicate message
                else:
                    ctx.insert(0, ctx.pop())                          # move
                msgs = [orjson.dumps(m, option=orjson.OPT_SORT_KEYS) for m in ctx]
                text = b"[" + b",".join(msgs) + b"]"
                _, ids, _, _, tr, _ = ing.request(t, msgs)
                wide.append((span, text.decode()))
                trans += [(t, span, op_, occ, msg, ord_) for op_, occ, msg, ord_ in tr]
                span += 1
            ing.end(t); traces.append((t, lo, span - 1))
        pq.write_table(pa.table({"span_id": pa.array([w[0] for w in wide], U32), "input": pa.array([w[1] for w in wide])}), f"{d}/cap_test_s1/prefix/wide.parquet")
        cols = list(zip(*trans))
        pq.write_table(pa.table({"trace_id": pa.array(cols[0], U32), "span_id": pa.array(cols[1], U32), "op": pa.array(cols[2], pa.int8()),
                                 "occ": pa.array(cols[3], U32), "msg_id": pa.array(cols[4], U32), "ord": pa.array(cols[5], pa.uint64())}),
                       f"{d}/cap_test_s1/prefix/transitions.parquet")
        pq.write_table(pa.table({"trace_id": pa.array([x[0] for x in traces], U32), "span_lo": pa.array([x[1] for x in traces], U32),
                                 "span_hi": pa.array([x[2] for x in traces], U32)}), f"{d}/cap_test_s1/traces.parquet")
        pq.write_table(pa.table({"msg_id": pa.array(range(len(dic)), U32), "text": pa.array([dic.read(x).decode() for x in range(len(dic))])}),
                       f"{d}/cap_test_s1/messages.parquet")
        import re
        preds = {"eval:sub8:0": ("like", "gamma de"), "eval:sub8:1": ("like", "Traceback (most"), "eval:sub8:2": ("like", "zzzz"),
                 "eval:re:0": ("match", "error [0-9]+ here"), "eval:re:1": ("match", "#[12][0-9]")}
        truth = {k: sum(1 for _, w in wide if (s in w if kind == "like" else re.search(s, w))) for k, (kind, s) in preds.items()}
        P = dict(sample="cap_test_s1", manager="prefix", source="synthetic", source_md5="-",
                 predicates={k: dict(kind=kind, s=s, truth_spans=truth[k]) for k, (kind, s) in preds.items()})
        json.dump(P, open(f"{d}/preds.json", "w"))
        cmd = [sys.executable, os.path.join(HERE, "duck_bench.py"), "--data", f"{d}/cap_test_s1", "--manager", "prefix", "--preds", f"{d}/preds.json",
               "--threads", "2", "--mem-gb", "2", "--cold", "1", "--warm", "2"]
        os.makedirs(f"{d}/w1")
        r = subprocess.run(cmd + ["--work", f"{d}/w1", "--out", f"{d}/out.json"], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        out = json.load(open(f"{d}/out.json"))
        assert len(out["records"]) == 15 and all(x["equal"] for x in out["records"]), out["records"]
        assert out["checks"] == dict(iv_rows_minus_opens=0, wide_rows_minus_distinct_spans=0, bad_intervals=0)
        P["predicates"]["eval:sub8:0"]["truth_spans"] += 1                # a wrong truth must stop the run
        json.dump(P, open(f"{d}/preds_bad.json", "w"))
        os.makedirs(f"{d}/w2")
        r = subprocess.run(cmd[:7] + [f"{d}/preds_bad.json"] + cmd[8:] + ["--work", f"{d}/w2", "--out", f"{d}/bad.json"], capture_output=True, text=True)
        assert r.returncode != 0 and "mismatch" in (r.stdout + r.stderr), "a wrong truth was accepted"
    print("TEST_DUCK_BENCH_OK", truth)


if __name__ == "__main__":
    main()
