"""M0 vertical slice, step 1: convert agent trajectories into factored context tables.

One span per LLM call (assistant message). A span's context is what the agent would send under
a context policy:
  prefix      every earlier message;
  window<N>   a fixed head (tool definitions, system prompt, first user message) plus the last N
              other messages (sliding-window eviction).
Outputs, as Parquet under --out:
  messages.parquet             msg_id, role, n_chars, text    one row per distinct serialized message
  spans.parquet                span_id, trace_id, span_seq, pos, out_msg
  traces.parquet               trace_id, span_lo, span_hi, n_msgs, repo, exit_status, resolved
  <policy>/intervals.parquet   trace_id, pos, msg_id, span_from, span_to   one row per message occurrence
  <policy>/arrays.parquet      span_id, poss, msg_ids         the context of every span, in order
  <policy>/wide.parquet        span_id, input                 serialized context, first --wide-traj trajectories
Span IDs are dense in (trace, span order), so a trace is a contiguous span-ID range. `pos`, the
position of a message in its trajectory, identifies an occurrence and is its order key.
"""
import argparse, json, os, time

import numpy as np, orjson, pyarrow as pa, pyarrow.parquet as pq, xxhash

ZSTD3 = dict(compression="zstd", compression_level=3)
U32 = pa.uint32()


def context_index(ks, h, window):
    """Message positions of every span's context, concatenated, plus the per-span lengths.
    ks: positions of the assistant messages (one span each); h: head length; window: None or N."""
    head = np.minimum(h, ks)
    tail0 = np.full_like(ks, h) if window is None else np.maximum(h, ks - window)
    lens = head + np.maximum(ks - tail0, 0)
    local = np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens)
    hrep = np.repeat(head, lens)
    return np.where(local < hrep, local, np.repeat(tail0, lens) + local - hrep), lens


def intervals(n, ks, h, window):
    """For each message position, the range [lo, hi] of span indexes whose context holds it."""
    j = np.arange(n)
    lo = np.searchsorted(ks, j, side="right")                    # first span after the message
    hi = np.full(n, len(ks) - 1)
    if window is not None:                                       # in context while j < k <= j + window
        hi = np.where(j < h, hi, np.searchsorted(ks, j + window, side="right") - 1)
    keep = lo <= hi
    return j[keep], lo[keep], hi[keep]


def simulate(roles, window):
    """The context of every span as a list of message positions, by replaying the agent loop message by message.
    This is the oracle: it shares no code with context_index() and intervals(), and the wide table is built from it."""
    first_user = roles.index("user") if "user" in roles else None
    seen, out = [], []
    for i, r in enumerate(roles):
        if r == "assistant":
            if window is None:
                out.append(list(seen))
            else:
                if first_user is not None:
                    n_head = sum(1 for x in seen if x <= first_user)
                else:                                            # no user message: the head is the leading tools / system messages
                    n_head = next((n for n, x in enumerate(seen) if roles[x] not in ("tools", "system")), len(seen))
                tail = seen[n_head:]
                out.append(seen[:n_head] + tail[max(0, len(tail) - window):])
        seen.append(i)
    return out


def check(sim, flat, lens, iv):
    """Compare the vectorized arrays and intervals with the simulated contexts."""
    off = np.concatenate([[0], np.cumsum(lens)]).tolist()
    assert [flat[off[s]:off[s + 1]].tolist() for s in range(len(lens))] == sim
    assert {(j, s) for s, ctx in enumerate(sim) for j in ctx} == {(j, s) for j, lo, hi in zip(*(x.tolist() for x in iv)) for s in range(lo, hi + 1)}


class Sink:
    """Parquet writer that buffers numpy columns and writes one row group per flush."""

    def __init__(self, path, schema):
        self.w, self.schema, self.cols = pq.ParquetWriter(path, schema, **ZSTD3), schema, [[] for _ in schema]

    def add(self, *cols):
        for buf, c in zip(self.cols, cols):
            buf.append(c)

    def flush(self):
        if self.cols[0]:
            arrs = []
            for f, buf in zip(self.schema, self.cols):
                if pa.types.is_list(f.type):                     # buf holds (lengths, values) pairs
                    lens = np.concatenate([b[0] for b in buf]); vals = np.concatenate([b[1] for b in buf])
                    off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
                    arrs.append(pa.ListArray.from_arrays(pa.array(off, pa.int32()), pa.array(vals, U32)))
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
    ap.add_argument("--src", required=True, help="Parquet file with columns trajectory, tools, repo, exit_status, resolved")
    ap.add_argument("--out", required=True)
    ap.add_argument("--policies", default="prefix,window32,window128")
    ap.add_argument("--wide-policies", default="prefix,window32")
    ap.add_argument("--wide-traj", type=int, default=3000)
    ap.add_argument("--limit", type=int, default=0, help="convert only the first N trajectories (0 = all)")
    a = ap.parse_args()
    policies = {p: (None if p == "prefix" else int(p.removeprefix("window"))) for p in a.policies.split(",")}
    wide = set(a.wide_policies.split(",")) & set(policies)
    for p in policies:
        os.makedirs(f"{a.out}/{p}", exist_ok=True)

    lst = pa.list_(U32)
    msgs_out = Sink(f"{a.out}/messages.parquet", pa.schema([("msg_id", U32), ("role", pa.string()), ("n_chars", U32), ("text", pa.large_string())]))
    spans_out = Sink(f"{a.out}/spans.parquet", pa.schema([("span_id", U32), ("trace_id", U32), ("span_seq", U32), ("pos", U32), ("out_msg", U32)]))
    traces_out = Sink(f"{a.out}/traces.parquet", pa.schema([("trace_id", U32), ("span_lo", U32), ("span_hi", U32), ("n_msgs", U32),
                                                           ("repo", pa.string()), ("exit_status", pa.string()), ("resolved", pa.int8())]))
    iv_out = {p: Sink(f"{a.out}/{p}/intervals.parquet", pa.schema([("trace_id", U32), ("pos", U32), ("msg_id", U32), ("span_from", U32), ("span_to", U32)])) for p in policies}
    arr_out = {p: Sink(f"{a.out}/{p}/arrays.parquet", pa.schema([("span_id", U32), ("poss", lst), ("msg_ids", lst)])) for p in policies}
    wide_out = {p: Sink(f"{a.out}/{p}/wide.parquet", pa.schema([("span_id", U32), ("input", pa.large_string())])) for p in wide}

    # 128-bit content hash -> dense id, in first-seen order. This offline converter trusts the hash;
    # the ingestion path adds the byte comparison on a bucket hit.
    msg_id = {}
    stats = {p: dict(interval_rows=0, array_elems=0, closes=0, wide_chars=0) for p in policies}
    n_traces = n_spans = n_occ = n_checked = buffered = 0
    t0 = time.time()
    pf = pq.ParquetFile(a.src)
    for batch in pf.iter_batches(batch_size=64, columns=["trajectory", "tools", "repo", "exit_status", "resolved"]):
        for row in batch.to_pylist():
            raw = [orjson.dumps({"role": "tools", "content": row["tools"]}, option=orjson.OPT_SORT_KEYS)]   # tool definitions are sent with every call
            raw += [orjson.dumps(m, option=orjson.OPT_SORT_KEYS) for m in row["trajectory"]]
            roles = ["tools"] + [m["role"] for m in row["trajectory"]]
            ks = np.array([i for i, r in enumerate(roles) if r == "assistant"], np.int64)
            if not ks.size:
                continue
            ids = np.empty(len(raw), np.uint32)
            new = ([], [], [], [])
            for i, m in enumerate(raw):
                d = xxhash.xxh3_128_digest(m)
                x = msg_id.get(d)
                if x is None:
                    x = msg_id[d] = len(msg_id)
                    new[0].append(x); new[1].append(roles[i]); new[3].append(m.decode()); new[2].append(len(new[3][-1]))
                ids[i] = x
            if new[0]:
                msgs_out.add(*new); buffered += sum(new[2])
            # head: everything up to the first user message (tool definitions, system prompt, task statement)
            h = next((i + 1 for i, r in enumerate(roles) if r == "user"), None)
            if h is None:
                h = next((i for i, r in enumerate(roles) if r not in ("tools", "system")), len(roles))
            t, s0, ns = n_traces, n_spans, ks.size
            span_ids = np.arange(s0, s0 + ns, dtype=np.uint32)
            spans_out.add(span_ids, np.full(ns, t, np.uint32), np.arange(ns, dtype=np.uint32), ks.astype(np.uint32), ids[ks])
            traces_out.add([t], [s0], [s0 + ns - 1], [len(raw)], [row["repo"]], [row["exit_status"]], [-1 if row["resolved"] is None else row["resolved"]])
            for p, window in policies.items():
                flat, lens = context_index(ks, h, window)
                iv = intervals(len(raw), ks, h, window)
                sim = None
                if t < a.wide_traj or t % 50 == 0:                # the wide subset and a sample across the corpus
                    sim = simulate(roles, window)
                    check(sim, flat, lens, iv)
                j, lo, hi = iv
                iv_out[p].add(np.full(j.size, t, np.uint32), j.astype(np.uint32), ids[j], (s0 + lo).astype(np.uint32), (s0 + hi).astype(np.uint32))
                arr_out[p].add(span_ids, (lens, flat.astype(np.uint32)), (lens, ids[flat]))
                st = stats[p]
                st["interval_rows"] += int(j.size); st["array_elems"] += int(flat.size); st["closes"] += int((hi < ns - 1).sum())
                if p in wide and t < a.wide_traj:
                    texts = ["[" + ",".join(raw[q].decode() for q in ctx) + "]" for ctx in sim]
                    st["wide_chars"] += sum(map(len, texts))
                    wide_out[p].add(span_ids, texts); wide_out[p].flush()
            n_traces += 1; n_spans += ns; n_occ += len(raw); n_checked += sim is not None
            if buffered > 64 << 20:
                msgs_out.flush(); buffered = 0
            if n_traces % 256 == 0:
                for sink in (spans_out, traces_out, *iv_out.values(), *arr_out.values()):
                    sink.flush()
            if n_traces % 4096 == 0:
                print(f"{n_traces} traces, {n_spans} spans, {len(msg_id)} messages, {time.time() - t0:.0f}s", flush=True)
            if n_traces == a.limit:
                break
        if n_traces == a.limit:
            break
    for sink in (msgs_out, spans_out, traces_out, *iv_out.values(), *arr_out.values(), *wide_out.values()):
        sink.close()
    for p, st in stats.items():                                   # the segment statistics the cost model reads
        tau = st["interval_rows"] + st["closes"]
        st.update(mean_context_len=round(st["array_elems"] / n_spans, 2), transitions=tau,
                  mean_run_len_spans=round(st["array_elems"] / st["interval_rows"], 2), churn=round(tau / st["array_elems"], 5))
    out = dict(src=a.src, traces=n_traces, spans=n_spans, message_occurrences=n_occ, distinct_messages=len(msg_id),
               wide_traj=min(a.wide_traj, n_traces), checked_traj=n_checked, seconds=round(time.time() - t0, 1), policies=stats)
    json.dump(out, open(f"{a.out}/convert_stats.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
