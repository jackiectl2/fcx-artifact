"""Profile FCX's derivation (delta parsing + differencer) on the first N requests of one stream; development aid, not a result.
usage: prof_derive.py --data <capture dir> --manager <m> --work <dir> [--n 20000]"""
import argparse, cProfile, io, pstats, random, time

import pyarrow.parquet as pq

from ingest import Dictionary, Ingester
from ingest_run import arrivals

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--manager", required=True); ap.add_argument("--work", required=True)
ap.add_argument("--n", type=int, default=20000)
a = ap.parse_args()
sp = pq.read_table(f"{a.data}/spans.parquet", columns=["trace_id"])
reqs = []
for trace, span, text, last in arrivals(pq.ParquetFile(f"{a.data}/{a.manager}/wide.parquet"), sp["trace_id"].to_numpy(), 64, random.Random(1)):
    reqs.append((trace, text, last))
    if len(reqs) >= a.n:
        break


def run():
    ing = Ingester(Dictionary(f"{a.work}/d.bin"), 64, 32)
    for trace, text, last in reqs:
        ing.request_text(trace, text)
        if last:
            ing.end(trace)


t = time.process_time(); run(); base = time.process_time() - t
pr = cProfile.Profile(); pr.enable(); run(); pr.disable()
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(22)
print(f"PROFILE {a.data.rstrip('/').split('/')[-1]} {a.manager}: {len(reqs)} requests, {base:.2f} s CPU without profiler, {1e6 * base / len(reqs):.1f} us/request")
print(s.getvalue())
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(14)
print(s.getvalue())
