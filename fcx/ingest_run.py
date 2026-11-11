"""Ingest raw requests into an FCX store: from a request log to the files the benchmark driver loads.

Input (written by capture.py): <data>/<manager>/wide.parquet with one row per LLM call, (span_id, input), where `input` is the
serialized request (a JSON array of messages) and a row group holds one trace; <data>/spans.parquet maps a span to its trace
and its call number. With --delta 1 (default) a request is first compared, byte by byte, with the previous request of its
trace, and only the part that differs is parsed; with --delta 0 every request is parsed in full. Each parsed message is looked
up in the dictionary (128-bit hash, bytes compared on a hit), and the Ingester turns the request into transitions. With
--interleave K the requests of K traces at a time arrive interleaved, in random order across the K traces and in call order
within each.

Output: messages.parquet (shared by the managers) and, per manager, transitions.parquet, checkpoints.parquet and
arrays.parquet (the message IDs of every request as parsed: the reference of the later checks); variants.parquet (messages that
are small edits of a message they replace, as splices of that message); convert_stats.json with counts, timings, CPU time
and peak memory. Buffers are flushed every --flush requests, so memory does not grow with the log.
"""
import argparse, json, os, random, resource, time

import numpy as np, orjson, pyarrow as pa, pyarrow.parquet as pq

from convert import Sink
from ingest import Dictionary, Ingester

U32, U64 = pa.uint32(), pa.uint64()


def arrivals(src, trace_of, k, rng):
    """Yield (trace, span_id, request bytes, last request of the trace?) with the requests of up to k traces interleaved."""
    groups, pool = iter(range(src.num_row_groups)), []

    def refill():
        while len(pool) < k:
            g = next(groups, None)
            if g is None:
                return
            tab = src.read_row_group(g)
            spans = tab["span_id"].to_numpy()
            assert len(set(trace_of[spans].tolist())) == 1, "a row group must hold one trace"
            pool.append([int(trace_of[spans[0]]), spans.tolist(), tab["input"].cast(pa.large_binary()).to_pylist(), 0])

    refill()
    while pool:
        e = pool[rng.randrange(len(pool))] if k > 1 else pool[0]
        i = e[3]; e[3] += 1
        last = e[3] == len(e[1])
        if last:
            pool.remove(e)
        yield e[0], e[1][i], e[2][i], last
        if last:
            refill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--managers", required=True)
    ap.add_argument("--out", default="", help="directory for the output files (default: --data)")
    ap.add_argument("--delta", type=int, default=1, help="1: compare with the previous request of the trace and parse only what differs; 0: parse every request in full")
    ap.add_argument("--work", default=os.environ.get("TMPDIR", "/tmp"), help="directory for the dictionary's byte file (node-local disk)")
    ap.add_argument("--interleave", type=int, default=1, help="number of traces whose requests arrive interleaved")
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--flush", type=int, default=2000)
    ap.add_argument("--budget", type=int, default=64); ap.add_argument("--ckpt", type=int, default=32)
    a = ap.parse_args()
    managers = a.managers.replace(":", ",").split(",")
    out_dir = a.out or a.data
    os.makedirs(out_dir, exist_ok=True)
    sp = pq.read_table(f"{a.data}/spans.parquet")
    trace_of, seq_of = sp["trace_id"].to_numpy(), sp["span_seq"].to_numpy()       # span_id is the row number
    n_spans, n_traces = len(trace_of), int(trace_of.max()) + 1
    dic = Dictionary(f"{a.work}/fcx-dict-{os.getpid()}.bin")
    msgs_out = Sink(f"{out_dir}/messages.parquet", pa.schema([("msg_id", U32), ("role", pa.string()), ("n_chars", U32), ("text", pa.large_string())]))
    B = pa.large_binary()
    var_out = Sink(f"{out_dir}/variants.parquet", pa.schema([("msg_id", U32), ("base_id", U32), ("p", U32), ("s", U32), ("ins", B), ("wv", B), ("wb", B)]))
    stats = {}
    for p in managers:
        ing = Ingester(dic, a.budget, a.ckpt)
        os.makedirs(f"{out_dir}/{p}", exist_ok=True)
        tr_out = Sink(f"{out_dir}/{p}/transitions.parquet", pa.schema([("trace_id", U32), ("span_id", U32), ("op", pa.int8()), ("occ", U32), ("msg_id", U32), ("ord", U64)]))
        ck_out = Sink(f"{out_dir}/{p}/checkpoints.parquet", pa.schema([("span_id", U32), ("occs", pa.list_(U32)), ("ords", pa.list_(U64)), ("msg_ids", pa.list_(U32))]))
        arr_out = Sink(f"{out_dir}/{p}/arrays.parquet", pa.schema([("span_id", U32), ("msg_ids", pa.list_(U32))]))
        st = dict(array_elems=0, opens=0, closes=0, checkpoint_elems=0, wide_chars=0, parsed_bytes=0, new_messages=0, new_message_bytes=0, variants=0, variant_bytes=0, max_active_traces=0)
        t = dict(read=0.0, ingest=0.0, write=0.0)
        src = pq.ParquetFile(f"{a.data}/{p}/wide.parquet")
        cpu0, wall0, n, t0 = time.process_time(), time.perf_counter(), 0, time.perf_counter()
        for trace, span, text, last in arrivals(src, trace_of, a.interleave, random.Random(a.seed)):
            t1 = time.perf_counter(); t["read"] += t1 - t0
            st["max_active_traces"] = max(st["max_active_traces"], len(ing.active) + (trace not in ing.active))
            if a.delta:
                seq, ids, new, variants, trans, ck, roles, n_parsed = ing.request_text(trace, text)
            else:
                msgs = orjson.loads(text)
                raw = [orjson.dumps(m, option=orjson.OPT_SORT_KEYS) for m in msgs]
                assert b"[" + b",".join(raw) + b"]" == text, "the request is not in canonical form"
                seq, ids, new, variants, trans, ck = ing.request(trace, raw)
                role = {id(b): m.get("role", "") for b, m in zip(raw, msgs)}
                roles, n_parsed = [role[id(b)] for _, b in new], len(text)
            assert seq == seq_of[span], (trace, span, seq)
            if last:
                ing.end(trace)
            t3 = time.perf_counter(); t["ingest"] += t3 - t1
            st["parsed_bytes"] += n_parsed
            if new:
                txt = [b.decode() for _, b in new]
                msgs_out.add([x for x, _ in new], roles, [len(s) for s in txt], txt)
                st["new_message_bytes"] += sum(len(b) for _, b in new)
            if variants:
                var_out.add(*map(list, zip(*variants)))
                size = dict(new)
                st["variants"] += len(variants); st["variant_bytes"] += sum(len(size[v[0]]) for v in variants)
            if trans:
                r = np.array([x[:3] for x in trans], dtype=np.int64)
                tr_out.add(np.full(len(r), trace, np.uint32), np.full(len(r), span, np.uint32), r[:, 0].astype(np.int8), r[:, 1].astype(np.uint32),
                           r[:, 2].astype(np.uint32), np.array([x[3] for x in trans], np.uint64))
                st["opens"] += int((r[:, 0] == 1).sum()); st["closes"] += int((r[:, 0] == -1).sum())
            if ck is not None:
                ck_out.add(np.array([span], np.uint32), ([len(ck[0])], ck[0].astype(np.uint32)), ([len(ck[1])], ck[1]), ([len(ck[2])], ck[2].astype(np.uint32)))
                st["checkpoint_elems"] += len(ck[0])
            arr_out.add(np.array([span], np.uint32), ([len(ids)], np.array(ids, np.uint32)))
            st["array_elems"] += len(ids); st["wide_chars"] += len(text); st["new_messages"] += len(new); n += 1
            if n % a.flush == 0:
                for sink in (msgs_out, var_out, tr_out, ck_out, arr_out):
                    sink.flush()
            t0 = time.perf_counter(); t["write"] += t0 - t3
        t1 = time.perf_counter()
        for sink in (tr_out, ck_out, arr_out):
            sink.close()
        msgs_out.flush()
        t["write"] += time.perf_counter() - t1
        assert n == n_spans and not ing.active, (p, n, n_spans)
        wall = time.perf_counter() - wall0
        st.update(fallbacks=ing.fallbacks, mean_context_len=round(st["array_elems"] / n, 2), transitions=st["opens"] + st["closes"],
                  transitions_per_span=round((st["opens"] + st["closes"]) / n, 3), mean_run_len_spans=round(st["array_elems"] / max(st["opens"], 1), 2),
                  ingest=dict(requests=n, interleave=a.interleave, delta=a.delta, wall_s=round(wall, 1), cpu_s=round(time.process_time() - cpu0, 1), seconds={k: round(v, 1) for k, v in t.items()},
                              requests_per_s=round(n / wall, 1), logical_mb_per_s=round(st["wide_chars"] / wall / 1e6, 1),
                              peak_rss_mb_so_far=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1), distinct_messages_so_far=len(dic)))
        stats[p] = st
        print(p, json.dumps(st), flush=True)
    msgs_out.close(); var_out.close()
    out = dict(src=a.data, traces=n_traces, spans=n_spans, distinct_messages=len(dic), dictionary_bytes=dic.size, budget=a.budget, ckpt=a.ckpt,
               interleave=a.interleave, peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1), policies=stats)
    json.dump(out, open(f"{out_dir}/convert_stats.json", "w"), indent=1)
    os.close(dic.fd); os.remove(f"{a.work}/fcx-dict-{os.getpid()}.bin")
    print("INGEST_OK")


if __name__ == "__main__":
    main()
