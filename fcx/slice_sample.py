"""Copy the first K traces of a converted sample (capture Parquet format) into a new directory, for a small sanity run.

Copies spans.parquet, traces.parquet and <manager>/wide.parquet restricted to traces 0..K-1 (span IDs stay dense from 0 because
traces are numbered in order and their spans are contiguous). Refuses to overwrite the destination; writes to a staging
directory next to it and renames it into place only when complete.
usage: slice_sample.py --src <dir> --dst <dir> --traces K [--manager wiki]
"""
import argparse, os, shutil

import pyarrow.compute as pc, pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--dst", required=True)
    ap.add_argument("--traces", type=int, required=True); ap.add_argument("--manager", default="wiki")
    a = ap.parse_args()
    if os.path.exists(a.dst):
        raise SystemExit(f"refusing to overwrite {a.dst}")
    tr = pq.read_table(f"{a.src}/traces.parquet")
    tr = tr.filter(pc.less(tr["trace_id"], a.traces))
    assert tr.num_rows == a.traces and tr["trace_id"].to_pylist() == list(range(a.traces)), "traces are not numbered 0..K-1"
    hi = max(tr["span_hi"].to_pylist())
    sp = pq.read_table(f"{a.src}/spans.parquet")
    sp = sp.filter(pc.less_equal(sp["span_id"], hi))
    assert sp["span_id"].to_pylist() == list(range(hi + 1)) and max(sp["trace_id"].to_pylist()) == a.traces - 1
    w = pq.ParquetFile(f"{a.src}/{a.manager}/wide.parquet")
    stage = f"{a.dst.rstrip('/')}.staging-{os.getpid()}"
    try:
        os.makedirs(f"{stage}/{a.manager}")
        pq.write_table(tr, f"{stage}/traces.parquet"); pq.write_table(sp, f"{stage}/spans.parquet")
        with pq.ParquetWriter(f"{stage}/{a.manager}/wide.parquet", w.schema_arrow, compression="zstd") as out:
            n = 0
            for g in range(a.traces):                             # one row group per trace
                t = w.read_row_group(g)
                assert t["span_id"].to_pylist() == list(range(n, n + t.num_rows))
                out.write_table(t)
                n += t.num_rows
        assert n == hi + 1
        if os.path.exists(a.dst):
            raise SystemExit(f"refusing to overwrite {a.dst}")
        os.rename(stage, a.dst)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    print("SLICE_OK", a.traces, "traces", n, "spans")


if __name__ == "__main__":
    main()
