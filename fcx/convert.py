"""Convert agent trajectories into FCX's inputs by replaying context policies and ingesting the resulting requests.

For every trajectory and policy, replay.contexts() gives the request of each LLM call as a list of messages. The requests are
  (a) written as message-ID arrays (<policy>/arrays.parquet): the reference for every later check;
  (b) fed one by one to the differencer, whose transition log and checkpoints are what FCX stores
      (<policy>/transitions.parquet, <policy>/checkpoints.parquet);
  (c) for the first --wide-traj trajectories, serialized as a wide span table would hold them (<policy>/wide.parquet).
Also written: messages.parquet (one row per distinct serialized message), spans.parquet, traces.parquet.
Span IDs are dense in (trace, call order), so a trace is a contiguous span-ID range.
With --sample N the input is N trajectories drawn uniformly at random from the corpus (seed --seed) instead of all of it.
"""
import argparse, glob, json, os, time

import numpy as np, orjson, pyarrow as pa, pyarrow.parquet as pq

from ingest import Dictionary, Differencer
from replay import contexts, head_len, stub_text

U32, U64 = pa.uint32(), pa.uint64()


class Sink:
    """Parquet writer that buffers columns and writes one row group per flush. A list column is buffered as (lengths, values)."""

    def __init__(self, path, schema):
        self.w, self.schema, self.cols = pq.ParquetWriter(path, schema, compression="zstd", compression_level=3), schema, [[] for _ in schema]

    def add(self, *cols):
        for buf, c in zip(self.cols, cols):
            buf.append(c)

    def flush(self):
        if self.cols[0]:
            arrs = []
            for f, buf in zip(self.schema, self.cols):
                if pa.types.is_list(f.type):
                    off = np.concatenate([[0], np.cumsum(np.concatenate([b[0] for b in buf]))]).astype(np.int32)
                    arrs.append(pa.ListArray.from_arrays(pa.array(off, pa.int32()), pa.array(np.concatenate([b[1] for b in buf]), f.type.value_type)))
                elif isinstance(buf[0], np.ndarray):
                    arrs.append(pa.array(np.concatenate(buf), f.type))
                else:
                    arrs.append(pa.array([x for b in buf for x in b], f.type))
            self.w.write_table(pa.Table.from_arrays(arrs, schema=self.schema))
            self.cols = [[] for _ in self.schema]

    def close(self):
        self.flush(); self.w.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Parquet file, or a glob of files, with one trajectory per row")
    ap.add_argument("--out", required=True)
    ap.add_argument("--traj-col", default="trajectory"); ap.add_argument("--tools-col", default="tools")
    ap.add_argument("--assistant-role", default="assistant")
    ap.add_argument("--tool-role", default="tool", help="role of tool outputs after the head (SWE-agent logs them as user messages)")
    ap.add_argument("--resolved-col", default="resolved")
    ap.add_argument("--traj-json", action="store_true", help="the trajectory column holds the message list as a JSON string")
    ap.add_argument("--policies", default="prefix,window32,amortized32,elide5")
    ap.add_argument("--wide-policies", default="prefix,window32")
    ap.add_argument("--wide-traj", type=int, default=3000)
    ap.add_argument("--limit", type=int, default=0, help="convert only the first N trajectories (0 = all)")
    ap.add_argument("--work", default=os.environ.get("TMPDIR", "/tmp"), help="directory for the dictionary's byte file (node-local disk)")
    ap.add_argument("--sample", type=int, default=0, help="convert N trajectories drawn at random (0 = all)"); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--budget", type=int, default=64, help="edit budget of the differencer"); ap.add_argument("--ckpt", type=int, default=32)
    a = ap.parse_args()
    policies = a.policies.split(","); wide = set(a.wide_policies.split(",")) & set(policies)
    for p in policies:
        os.makedirs(f"{a.out}/{p}", exist_ok=True)
    files = sorted(glob.glob(a.src))
    total = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    keep = set(np.random.default_rng(a.seed).choice(total, a.sample, replace=False).tolist()) if a.sample else None
    meta = [c for c in ("repo", "instance_id", "exit_status", a.resolved_col) if c in pq.ParquetFile(files[0]).schema_arrow.names]

    msgs_out = Sink(f"{a.out}/messages.parquet", pa.schema([("msg_id", U32), ("role", pa.string()), ("n_chars", U32), ("text", pa.large_string())]))
    spans_out = Sink(f"{a.out}/spans.parquet", pa.schema([("span_id", U32), ("trace_id", U32), ("span_seq", U32), ("pos", U32), ("out_msg", U32)]))
    traces_out = Sink(f"{a.out}/traces.parquet", pa.schema([("trace_id", U32), ("span_lo", U32), ("span_hi", U32), ("n_msgs", U32),
                                                           ("repo", pa.string()), ("exit_status", pa.string()), ("resolved", pa.int8())]))
    tr_out = {p: Sink(f"{a.out}/{p}/transitions.parquet", pa.schema([("trace_id", U32), ("span_id", U32), ("op", pa.int8()), ("occ", U32), ("msg_id", U32), ("ord", U64)])) for p in policies}
    ck_out = {p: Sink(f"{a.out}/{p}/checkpoints.parquet", pa.schema([("span_id", U32), ("occs", pa.list_(U32)), ("ords", pa.list_(U64)), ("msg_ids", pa.list_(U32))])) for p in policies}
    arr_out = {p: Sink(f"{a.out}/{p}/arrays.parquet", pa.schema([("span_id", U32), ("msg_ids", pa.list_(U32))])) for p in policies}
    wide_out = {p: Sink(f"{a.out}/{p}/wide.parquet", pa.schema([("span_id", U32), ("input", pa.large_string())])) for p in wide}

    msg_id, new = Dictionary(f"{a.work}/fcx-dict-{os.getpid()}.bin"), ([], [], [], [])     # dense ids in first-seen order; bytes compared on a hash hit

    def intern(text, role):
        x, fresh = msg_id.intern(text)
        if fresh:
            new[0].append(x); new[1].append(role); new[3].append(text.decode()); new[2].append(len(new[3][-1]))
        return x

    stats = {p: dict(array_elems=0, opens=0, closes=0, fallbacks=0, checkpoint_elems=0, wide_chars=0) for p in policies}
    n_traces = n_spans = n_occ = n_checked = buffered = row_no = 0
    t0 = time.time()
    batches = (b for f in files for b in pq.ParquetFile(f).iter_batches(batch_size=64, columns=[a.traj_col] + ([a.tools_col] if a.tools_col else []) + meta))
    for batch in batches:
        if keep is not None:
            idx = [i for i in range(batch.num_rows) if row_no + i in keep]
            row_no += batch.num_rows
            batch = batch.take(pa.array(idx, pa.int64()))
        for row in batch.to_pylist():
            traj = orjson.loads(row[a.traj_col]) if a.traj_json else row[a.traj_col]
            raw = [orjson.dumps(m, option=orjson.OPT_SORT_KEYS) for m in traj]
            roles = ["assistant" if m["role"] == a.assistant_role else m["role"] for m in traj]
            if a.tools_col:                                       # tool definitions are sent with every call
                raw.insert(0, orjson.dumps({"role": "tools", "content": row[a.tools_col]}, option=orjson.OPT_SORT_KEYS)); roles.insert(0, "tools")
            if a.tool_role != "tool":                             # tool outputs logged under another role: every such message after the head
                h = head_len(roles)
                roles = ["tool" if r == a.tool_role and i >= h else r for i, r in enumerate(roles)]
            ks = [i for i, r in enumerate(roles) if r == "assistant"]
            if not ks:
                continue
            ids = [intern(m, roles[i]) for i, m in enumerate(raw)]
            t, s0, ns = n_traces, n_spans, len(ks)
            span_ids = np.arange(s0, s0 + ns, dtype=np.uint32)
            spans_out.add(span_ids, np.full(ns, t, np.uint32), np.arange(ns, dtype=np.uint32), np.array(ks, np.uint32), np.array([ids[k] for k in ks], np.uint32))
            repo = row.get("repo") or (row["instance_id"].rsplit("-", 1)[0] if row.get("instance_id") else None)
            traces_out.add([t], [s0], [s0 + ns - 1], [len(raw)], [repo], [row.get("exit_status")], [-1 if row.get(a.resolved_col) is None else int(row[a.resolved_col])])
            check = t < a.wide_traj or t % 50 == 0
            for p in policies:
                st, diff, stubs, active = stats[p], Differencer(a.budget, a.ckpt), {}, {}
                text_of = dict(zip(ids, raw)) if p in wide and t < a.wide_traj else None     # msg_id -> serialized message
                lens, flat, rows, texts = [], [], [], []
                for s, ctx in enumerate(contexts(roles, p)):
                    req = []
                    for item in ctx:
                        if isinstance(item, tuple):               # an elided tool output: a short stand-in message
                            j = item[1]
                            if j not in stubs:
                                stub = stub_text(len(raw[j])).encode()
                                stubs[j] = intern(stub, "tool")
                                if text_of is not None:
                                    text_of[stubs[j]] = stub
                            req.append(stubs[j])
                        else:
                            req.append(ids[item])
                    lens.append(len(req)); flat += req
                    trans, ck = diff.feed(req)
                    rows += [(s0 + s, op, occ, mid, key) for op, occ, mid, key in trans]
                    if ck is not None:
                        ck_out[p].add(np.array([s0 + s], np.uint32), ([len(ck[0])], ck[0].astype(np.uint32)), ([len(ck[1])], ck[1]), ([len(ck[2])], ck[2].astype(np.uint32)))
                        st["checkpoint_elems"] += len(ck[0])
                    if check:                                     # replay the log independently and compare with the request
                        for op, occ, mid, key in trans:
                            if op == 1:
                                active[occ] = (key, mid)
                            else:
                                assert active.pop(occ) == (key, mid), (t, s, p)
                        assert [m for _, m in sorted(active.values())] == req, (t, s, p)
                    if text_of is not None:
                        texts.append("[" + ",".join(text_of[m].decode() for m in req) + "]")
                r = np.array(rows, dtype=np.int64).reshape(-1, 5) if rows else np.empty((0, 5), np.int64)
                tr_out[p].add(np.full(len(r), t, np.uint32), r[:, 0].astype(np.uint32), r[:, 1].astype(np.int8), r[:, 2].astype(np.uint32),
                              r[:, 3].astype(np.uint32), np.array([x[4] for x in rows], np.uint64))
                arr_out[p].add(span_ids, (lens, np.array(flat, np.uint32)))
                st["array_elems"] += len(flat); st["opens"] += int((r[:, 1] == 1).sum()); st["closes"] += int((r[:, 1] == -1).sum()); st["fallbacks"] += diff.fallbacks
                if texts:
                    st["wide_chars"] += sum(map(len, texts)); wide_out[p].add(span_ids, texts); wide_out[p].flush()
            if new[0]:
                msgs_out.add(*new); buffered += sum(new[2]); new = ([], [], [], [])
            n_traces += 1; n_spans += ns; n_occ += len(raw); n_checked += check
            if buffered > 64 << 20:
                msgs_out.flush(); buffered = 0
            if n_traces % 256 == 0:
                for sink in (spans_out, traces_out, *tr_out.values(), *ck_out.values(), *arr_out.values()):
                    sink.flush()
            if n_traces % 4096 == 0:
                print(f"{n_traces} traces, {n_spans} spans, {len(msg_id)} messages, {time.time() - t0:.0f}s", flush=True)
            if n_traces == a.limit:
                break
        if n_traces == a.limit:
            break
    for sink in (msgs_out, spans_out, traces_out, *tr_out.values(), *ck_out.values(), *arr_out.values(), *wide_out.values()):
        sink.close()
    for st in stats.values():                                     # the segment statistics
        st.update(mean_context_len=round(st["array_elems"] / n_spans, 2), transitions=st["opens"] + st["closes"],
                  transitions_per_span=round((st["opens"] + st["closes"]) / n_spans, 3), mean_run_len_spans=round(st["array_elems"] / max(st["opens"], 1), 2))
    out = dict(src=a.src, corpus_trajectories=total, sample=a.sample, seed=a.seed, traces=n_traces, spans=n_spans, message_occurrences=n_occ, distinct_messages=len(msg_id), budget=a.budget, ckpt=a.ckpt,
               wide_traj=min(a.wide_traj, n_traces), checked_traj=n_checked, seconds=round(time.time() - t0, 1), policies=stats)
    os.close(msg_id.fd); os.remove(f"{a.work}/fcx-dict-{os.getpid()}.bin")
    json.dump(out, open(f"{a.out}/convert_stats.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
