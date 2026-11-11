"""E-CHAT evidence: pairwise overlap of the three WildChat samples. A conversation's fingerprint is xxh3_128 of its canonical
message list (role and content, sorted keys); prints the per-sample counts of conversations and distinct fingerprints and the
pairwise intersections as JSON. usage: chat_overlap.py <dir with chat_pool_s{1,2,3}.parquet> <out.json>"""
import itertools, json, sys
import orjson, pyarrow.parquet as pq, xxhash
d, out = sys.argv[1], sys.argv[2]
fp = {}
for s in (1, 2, 3):
    rows = pq.read_table(f"{d}/chat_pool_s{s}.parquet", columns=["messages"]).column("messages").to_pylist()
    fp[s] = [xxhash.xxh3_128_hexdigest(orjson.dumps(m, option=orjson.OPT_SORT_KEYS)) for m in rows]
res = dict(conversations={s: len(v) for s, v in fp.items()}, distinct={s: len(set(v)) for s, v in fp.items()},
           pairwise_overlap={f"{a}-{b}": len(set(fp[a]) & set(fp[b])) for a, b in itertools.combinations((1, 2, 3), 2)},
           fingerprint="xxh3_128 of the canonical JSON message list (role, content)")
json.dump(res, open(out, "w"), indent=1); print(json.dumps(res))
