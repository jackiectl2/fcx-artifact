"""E-ONLINE: the online cost of FCX's derivation per request: latency, state of the active traces, and how both depend on the
number of traces whose requests arrive interleaved.

One pass of the paper's ingestion (Ingester.request_text, delta parsing, edit budget 64, checkpoint interval 32) over the request
stream of one (capture directory, manager), in the arrival order of ingest_run.arrivals with --interleave K and arrival seed 1.
Nothing is written but the result JSON; the dictionary's byte file lives in --work and is deleted.

Per request, timed with perf_counter around request_text only (reading the input is not timed): wall latency in microseconds.
Reported: count, the 50th, 90th, 99th and 99.9th percentiles and the maximum, the mean, and the same percentiles of latency per
KB of request text; total wall and CPU seconds and MB/s of request text.

Active-trace payload, in bytes, after every request and after a completed trace has been dropped (retained state between
requests; Python object overhead and the few scalar counters per trace are excluded, so it is a lower bound on resident memory):
per active trace, the previous request kept for delta parsing (its bytes, 8 B per message ID, the separator offsets array) plus
the open occurrences (occurrence ID, order key, message ID arrays: 24 B each). Reported: the maximum and the request-weighted
mean of the total over active traces, the maximum number of active traces, and the mean payload per active trace. Global state
is reported separately: the dictionary (distinct messages; bytes of its byte file, on disk; entries of its in-memory hash index
and location list, one each per distinct message) and the manager-wide variant map (entries; it outlives traces). harness peak
RSS (ru_maxrss, KiB on Linux) covers the whole process: the arrival buffer (every request of the active traces), the replay
check's copy of the open occurrences and the latency lists; it is not FCX's ingestion memory.

Correctness: requests of a trace arrive in call order (the ingester's call number equals the span's span_seq) and the stream
is complete (requests = spans); every request's message IDs, rebuilt from the log of this pass by replaying the transitions, must equal the IDs
the ingester parsed (the same check as ingest_ablate's validation, on IDs); a difference stops the run.

usage: ingest_online.py --data <capture dir> --manager <m> --interleave K --work <node-local dir> --out <json>
"""
import argparse, json, os, random, resource, time

import numpy as np, pyarrow.parquet as pq

import ingest
from ingest import Dictionary, Ingester
from ingest_run import arrivals

PCT = (50, 90, 99, 99.9)


def quantiles(x):
    x = np.asarray(x, np.float64)
    return dict({f"p{p}": float(np.percentile(x, p)) for p in PCT}, max=float(x.max()), mean=float(x.mean()), n=int(x.size))


def trace_state(ing, trace):
    """Exact payload bytes of one active trace's state (see the module docstring)."""
    b = 0
    d = ing.active.get(trace)
    if d is not None:
        b += d.state_bytes()                                      # 8 B per occurrence, key and message ID
    pv = ing.prev.get(trace)
    if pv is not None:
        b += len(pv[0]) + 8 * len(pv[1]) + 8 * len(pv[2])
    return b


class Replay:
    """Rebuild each request's message IDs from the transitions alone (opens add (ord, occ, msg); closes remove occ)."""

    def __init__(self):
        self.open = {}

    def apply(self, trace, trans):
        ctx = self.open.setdefault(trace, {})
        for op, occ, msg, ord_ in trans:
            if op == -1:
                if occ not in ctx:
                    raise AssertionError("close of an occurrence that is not open")
                del ctx[occ]
            else:
                if occ in ctx:
                    raise AssertionError("open of an occurrence that is already open")
                ctx[occ] = (int(ord_), msg)
        return [m for _, m in sorted(ctx.values())]

    def end(self, trace):
        self.open.pop(trace, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--manager", required=True)
    ap.add_argument("--interleave", type=int, required=True); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--budget", type=int, default=64); ap.add_argument("--ckpt", type=int, default=32)
    ap.add_argument("--work", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    spans = pq.read_table(f"{a.data}/spans.parquet", columns=["span_id", "trace_id", "span_seq"])
    sid = spans["span_id"].to_numpy()
    trace_of = np.zeros(int(sid.max()) + 1, np.int64); seq_of = np.full(int(sid.max()) + 1, -1, np.int64)
    trace_of[sid] = spans["trace_id"].to_numpy(); seq_of[sid] = spans["span_seq"].to_numpy()
    src = pq.ParquetFile(f"{a.data}/{a.manager}/wide.parquet")
    path = os.path.join(a.work, f"online_{os.getpid()}.dict")
    dic = Dictionary(path)
    ing, rep = Ingester(dic, a.budget, a.ckpt), Replay()
    lat, per_kb, state_of, total = [], [], {}, 0
    st = dict(requests=0, logical_bytes=0, parsed_bytes=0, wall_s=0.0, cpu_s=0.0, max_active=0, max_state_bytes=0, state_sum=0, active_sum=0)
    for trace, span, text, last in arrivals(src, trace_of, a.interleave, random.Random(a.seed)):
        c0, t0 = time.process_time(), time.perf_counter()
        call, ids, _, _, trans, _, _, n_parsed = ing.request_text(trace, text)
        t1 = time.perf_counter(); c1 = time.process_time()
        us = (t1 - t0) * 1e6
        lat.append(us); per_kb.append(us / max(len(text) / 1024, 1e-9))
        st["wall_s"] += t1 - t0; st["cpu_s"] += c1 - c0
        st["requests"] += 1; st["logical_bytes"] += len(text); st["parsed_bytes"] += n_parsed
        if call != seq_of[span]:
            raise AssertionError(f"span {span}: call {call} of its trace, expected span_seq {seq_of[span]}")
        if rep.apply(trace, trans) != list(ids):
            raise AssertionError(f"replayed message IDs differ from the parsed ones at span {span}")
        if last:                                                  # a completed trace is dropped before the state is sampled
            ing.end(trace); rep.end(trace)
            total -= state_of.pop(trace, 0)
        else:                                                     # update the running total for this trace only
            new = trace_state(ing, trace)
            total += new - state_of.get(trace, 0); state_of[trace] = new
        st["max_state_bytes"] = max(st["max_state_bytes"], total)
        st["max_active"] = max(st["max_active"], len(state_of))
        st["state_sum"] += total; st["active_sum"] += len(state_of)
    if ing.active or state_of or total != 0:
        raise AssertionError("traces left open after the stream")
    if st["requests"] != len(sid):
        raise AssertionError(f"{st['requests']} requests, {len(sid)} spans")
    n = st["requests"]
    out = dict(data=os.path.basename(os.path.normpath(a.data)), manager=a.manager, interleave=a.interleave, seed=a.seed, budget=a.budget, ckpt=a.ckpt,
               ingest_impl=getattr(ingest, "IMPL", "v1"),
               requests=n, logical_bytes=st["logical_bytes"], parsed_bytes=st["parsed_bytes"], wall_s=st["wall_s"], cpu_s=st["cpu_s"],
               mb_per_s=st["logical_bytes"] / 1e6 / st["wall_s"], latency_us=quantiles(lat), latency_us_per_kb=quantiles(per_kb),
               max_active_traces=st["max_active"], max_active_payload_bytes=st["max_state_bytes"], mean_active_payload_bytes=st["state_sum"] / n,
               mean_payload_bytes_per_active_trace=st["state_sum"] / max(st["active_sum"], 1),
               distinct_messages=len(dic), dictionary_file_bytes=dic.size, dictionary_index_entries=len(dic.index), dictionary_loc_entries=len(dic.loc),
               variant_map_entries=len(ing.base), harness_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, fallbacks=ing.fallbacks,
               replay_check="all requests equal, call order and completeness checked")
    os.close(dic.fd); os.remove(path)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1)); print("INGEST_ONLINE_OK")


if __name__ == "__main__":
    main()
