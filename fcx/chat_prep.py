"""E-CHAT: draw three disjoint samples of multi-turn conversations from WildChat-1M as trajectories for capture.py.

A conversation is a trace; every assistant message is one LLM call, whose request capture.py builds from the conversation's
earlier messages with each context manager under test (the recorded assistant messages are the model's replies). Each message
keeps only its role and its text, as the OpenAI chat format sends them: {"role": "user"|"assistant", "content": <text>}. No
other field of the dataset (hashes, IP-derived fields, headers, moderation scores, timestamps) is read into the output.

Eligible conversations (fixed before any WildChat sample was converted):
  - `turn` >= --min-turns (default 10), and the message list has exactly 2 * turn messages;
  - roles alternate user, assistant, starting with user;
  - every content is a string.
Conversation identity is the dataset's `conversation_hash`; a hash that occurs more than once is excluded entirely.
Eligible conversations are ranked by xxh64(conversation_hash, seed=--seed); ranks [k*N, (k+1)*N) form sample k+1, so the
samples are disjoint. Each sample file keeps the dataset's row order (file order, then row order).

Output: <out>/chat_pool_s{1,2,3}.parquet with columns messages (list of {role, content}) and repo (the constant "wildchat"),
and <out>/chat_prep_stats.json with the exclusion counts and the turn histogram of eligible conversations.

usage: chat_prep.py --src '<dir>/train-*.parquet' --out <dir> [--min-turns 10] [--per-sample 3000] [--samples 3] [--seed 1]
"""
import argparse, glob, json, os
from collections import Counter

import pyarrow as pa, pyarrow.parquet as pq, xxhash

MSG = pa.struct([("role", pa.string()), ("content", pa.string())])


def check(conv, turn, min_turns):
    """Reason a conversation is not eligible, or None."""
    if turn is None or turn < min_turns:
        return "few_turns"
    if conv is None or len(conv) != 2 * turn:
        return "length_mismatch"
    for k, m in enumerate(conv):
        if m.get("role") != ("user" if k % 2 == 0 else "assistant"):
            return "role_order"
        if not isinstance(m.get("content"), str):
            return "content_not_string"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--min-turns", type=int, default=10); ap.add_argument("--per-sample", type=int, default=3000)
    ap.add_argument("--samples", type=int, default=3); ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    files = sorted(glob.glob(a.src))
    if not files:
        raise SystemExit(f"no input files match {a.src}")
    reasons, rows, hash_count, total = Counter(), [], Counter(), 0
    for fi, f in enumerate(files):
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=2048, columns=["conversation_hash", "turn", "conversation"]):
            hashes = batch.column("conversation_hash").to_pylist()
            turns = batch.column("turn").to_pylist()
            convs = batch.column("conversation").to_pylist()
            for h, t, c in zip(hashes, turns, convs):
                total += 1
                hash_count[h] += 1
                why = check(c, t, a.min_turns)
                if why:
                    reasons[why] += 1
                    continue
                rows.append((fi, len(rows), h, t, [{"role": m["role"], "content": m["content"]} for m in c]))
    dup = {h for h, n in hash_count.items() if n > 1}
    kept = [r for r in rows if r[2] not in dup]
    reasons["duplicate_hash"] = len(rows) - len(kept)
    need = a.per_sample * a.samples
    if len(kept) < need:
        raise SystemExit(f"only {len(kept)} eligible conversations; {need} needed")
    ranked = sorted(kept, key=lambda r: (xxhash.xxh64_intdigest(r[2].encode(), seed=a.seed), r[2]))
    os.makedirs(a.out, exist_ok=True)
    stats = dict(src=[os.path.basename(f) for f in files], conversations=total, min_turns=a.min_turns, seed=a.seed,
                 excluded=dict(reasons), eligible=len(kept),
                 eligible_turns=dict(sorted(Counter(r[3] for r in kept).items())), samples={})
    for k in range(a.samples):
        part = sorted(ranked[k * a.per_sample:(k + 1) * a.per_sample], key=lambda r: (r[0], r[1]))
        path = f"{a.out}/chat_pool_s{k + 1}.parquet"
        if os.path.exists(path):
            raise SystemExit(f"refusing to overwrite {path}")
        pq.write_table(pa.table({"messages": pa.array([r[4] for r in part], pa.list_(MSG)), "repo": pa.array(["wildchat"] * len(part), pa.string())}),
                       path, compression="zstd", compression_level=3)
        stats["samples"][f"s{k + 1}"] = dict(conversations=len(part), calls=sum(r[3] for r in part),
                                             messages=sum(len(r[4]) for r in part), content_chars=sum(len(m["content"]) for r in part for m in r[4]))
    json.dump(stats, open(f"{a.out}/chat_prep_stats.json", "w"), indent=1)
    print(json.dumps(stats, indent=1)); print("CHAT_PREP_OK")


if __name__ == "__main__":
    main()
