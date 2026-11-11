"""E-FAST equivalence check: the sped-up ingest.py against the reference copy ingest_v1.py on real request streams. Both
ingesters see the same requests in the same arrival order; every return value of request_text (call number, message IDs, new
messages, variants, transitions, checkpoint arrays with their dtypes, roles of new messages, parsed bytes) and of request (the
full-parse path of the fcx_full ablation) must be identical, and so must the per-trace fallback counts. Also reports the CPU
time of both versions on the same requests (development aid; the paper's timings come from ingest_ablate.py).
usage: check_ingest_v2.py --data <capture dir> --manager <m> --work <node-local dir> [--n 0 (all requests; with --n, a prefix of the arrival order)]"""
import argparse, json, os, random, time

import numpy as np, orjson, pyarrow.parquet as pq

import ingest, ingest_v1
from ingest_run import arrivals

SORT = orjson.OPT_SORT_KEYS


def same(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape and bool((a == b).all())
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--manager", required=True); ap.add_argument("--work", required=True)
    ap.add_argument("--n", type=int, default=0)
    a = ap.parse_args()
    trace_of = pq.read_table(f"{a.data}/spans.parquet", columns=["trace_id"])["trace_id"].to_numpy()

    def stream():                                                  # the paper's arrival order (64 interleaved traces, seed 1), read anew per pass
        for i, (trace, span, text, last) in enumerate(arrivals(pq.ParquetFile(f"{a.data}/{a.manager}/wide.parquet"), trace_of, 64, random.Random(1))):
            if a.n and i >= a.n:
                return
            yield trace, text, last

    out = dict(data=os.path.basename(a.data.rstrip("/")), manager=a.manager)
    for mode in ("delta", "full"):
        ings = {name: mod.Ingester(mod.Dictionary(f"{a.work}/{name}-{mode}.bin"), 64, 32) for name, mod in (("v1", ingest_v1), ("v2", ingest))}
        cpu, bad, fb = dict(v1=0.0, v2=0.0), 0, []
        for i, (trace, text, last) in enumerate(stream()):
            res = {}
            for name in ("v1", "v2") if i % 2 == 0 else ("v2", "v1"):      # alternate which version runs first
                ing = ings[name]
                t = time.process_time()
                res[name] = ing.request_text(trace, text) if mode == "delta" else ing.request(trace, [orjson.dumps(m, option=SORT) for m in orjson.loads(text)])
                cpu[name] += time.process_time() - t
            if not same(res["v1"], res["v2"]):
                bad += 1
                if bad <= 3:
                    print("DIFF", mode, i, trace, repr(res["v1"])[:300], repr(res["v2"])[:300], flush=True)
            if last:
                f = [ings[n].active[trace].fallbacks for n in ("v1", "v2")]
                fb.append(f[0] == f[1])
                for ing in ings.values():
                    ing.end(trace)
        for name, ing in ings.items():
            os.close(ing.dict.fd); os.remove(f"{a.work}/{name}-{mode}.bin")
        out[mode] = dict(requests=i + 1, requests_differing=bad, traces_fallbacks_differing=fb.count(False), cpu_v1_s=round(cpu["v1"], 3), cpu_v2_s=round(cpu["v2"], 3),
                         speedup=round(cpu["v1"] / max(cpu["v2"], 1e-9), 3), totals_v1=dict(fallbacks=ings["v1"].fallbacks), totals_v2=dict(fallbacks=ings["v2"].fallbacks))
        print(mode, json.dumps(out[mode]), flush=True)
    ok = all(out[m]["requests_differing"] == 0 and out[m]["traces_fallbacks_differing"] == 0 and out[m]["totals_v1"] == out[m]["totals_v2"] for m in ("delta", "full"))
    print(("CHECK_V2_OK " if ok else "CHECK_V2_FAIL ") + json.dumps(out), flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
