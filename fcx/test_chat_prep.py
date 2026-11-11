"""Synthetic tests of chat_prep.py: eligibility rules, exclusion counts, disjoint samples, output schema (role and content only),
row order, determinism and refusal to overwrite. usage: python test_chat_prep.py  (prints TEST_CHAT_PREP_OK)"""
import json, os, subprocess, sys, tempfile

import pyarrow as pa, pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
TURN = pa.struct([("content", pa.string()), ("role", pa.string()), ("hashed_ip", pa.string()), ("country", pa.string())])


def conv(n, tag, bad=None):
    msgs = []
    for k in range(2 * n):
        role = "user" if k % 2 == 0 else "assistant"
        msgs.append(dict(content=f"{tag} message {k}", role=role, hashed_ip="SECRET", country="X"))
    if bad == "role":
        msgs[1]["role"] = "user"
    if bad == "content":
        msgs[2]["content"] = None
    if bad == "short":
        msgs = msgs[:-1]
    return msgs


def write(path, rows):
    pq.write_table(pa.table({"conversation_hash": pa.array([r[0] for r in rows]), "turn": pa.array([r[1] for r in rows], pa.int64()),
                             "conversation": pa.array([r[2] for r in rows], pa.list_(TURN)), "hashed_ip": pa.array(["SECRET"] * len(rows))}), path)


def run(src, out, *extra):
    return subprocess.run([sys.executable, os.path.join(HERE, "chat_prep.py"), "--src", src, "--out", out, *extra], capture_output=True, text=True)


def main():
    with tempfile.TemporaryDirectory() as d:
        good = [(f"h{i}", 10 + i % 3, conv(10 + i % 3, f"c{i}")) for i in range(30)]
        bad = [("few", 9, conv(9, "few")), ("role", 10, conv(10, "role", "role")), ("content", 10, conv(10, "content", "content")),
               ("short", 10, conv(10, "short", "short")), ("dup", 10, conv(10, "dup1")), ("dup", 11, conv(11, "dup2"))]
        write(f"{d}/train-00000.parquet", good[:15] + bad[:3])
        write(f"{d}/train-00001.parquet", bad[3:] + good[15:])
        r = run(f"{d}/train-*.parquet", f"{d}/out", "--per-sample", "8", "--samples", "3")
        assert r.returncode == 0, r.stderr
        st = json.load(open(f"{d}/out/chat_prep_stats.json"))
        assert st["conversations"] == 36 and st["eligible"] == 30, st
        assert st["excluded"] == {"few_turns": 1, "role_order": 1, "content_not_string": 1, "length_mismatch": 1, "duplicate_hash": 2}, st["excluded"]
        seen, order = set(), {f"c{i}": i for i in range(30)}
        for s in (1, 2, 3):
            t = pq.read_table(f"{d}/out/chat_pool_s{s}.parquet")
            assert t.column_names == ["messages", "repo"] and t.num_rows == 8
            rows = t.column("messages").to_pylist()
            tags = [m[0]["content"].split()[0] for m in rows]
            assert not (set(tags) & seen), "samples overlap"; seen |= set(tags)
            assert [order[x] for x in tags] == sorted(order[x] for x in tags), "row order not kept"
            for m in rows:
                assert all(set(x) == {"role", "content"} for x in m), "extra fields leaked"
                assert [x["role"] for x in m] == ["user", "assistant"] * (len(m) // 2)
            assert set(t.column("repo").to_pylist()) == {"wildchat"}
        assert "SECRET" not in open(f"{d}/out/chat_pool_s1.parquet", "rb").read().decode("latin-1")
        r2 = run(f"{d}/train-*.parquet", f"{d}/out2", "--per-sample", "8", "--samples", "3")
        for s in (1, 2, 3):
            assert pq.read_table(f"{d}/out/chat_pool_s{s}.parquet").equals(pq.read_table(f"{d}/out2/chat_pool_s{s}.parquet")), "not deterministic"
        assert run(f"{d}/train-*.parquet", f"{d}/out", "--per-sample", "8").returncode != 0, "overwrote an existing sample"
        assert run(f"{d}/train-*.parquet", f"{d}/out3", "--per-sample", "11").returncode != 0, "too few eligible conversations accepted"
    print("TEST_CHAT_PREP_OK")


if __name__ == "__main__":
    main()
