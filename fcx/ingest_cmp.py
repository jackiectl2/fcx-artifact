"""Delta ingestion against full parsing on the same requests: both must write the same store, and their costs are reported.

Runs ingest_run.py twice (--delta 0, --delta 1) into a work directory, requires the message, transition, checkpoint and array
files of every manager to hold the same rows, and writes both runs' statistics to --out.
usage: ingest_cmp.py --data <capture directory> --managers a:b --work <node-local directory> --out stats.json [--interleave K]
"""
import argparse, json, subprocess, sys

import pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--managers", required=True); ap.add_argument("--work", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--interleave", default="64"); ap.add_argument("--seed", default="1")
a = ap.parse_args()
stats = {}
for delta in ("0", "1"):
    subprocess.run([sys.executable, "ingest_run.py", "--data", a.data, "--managers", a.managers, "--work", a.work, "--out", f"{a.work}/delta{delta}",
                    "--delta", delta, "--interleave", a.interleave, "--seed", a.seed], check=True)
    stats["delta" + delta] = json.load(open(f"{a.work}/delta{delta}/convert_stats.json"))
files = ["messages", "variants"] + [f"{m}/{f}" for m in a.managers.replace(":", ",").split(",") for f in ("transitions", "checkpoints", "arrays")]
differ = [f for f in files if not pq.read_table(f"{a.work}/delta0/{f}.parquet").equals(pq.read_table(f"{a.work}/delta1/{f}.parquet"))]
stats["files_compared"], stats["files_that_differ"] = len(files), differ
import ingest                                                     # noqa: E402  (the implementation ingest_run.py imported)
stats["ingest_impl"] = getattr(ingest, "IMPL", "v1")
json.dump(stats, open(a.out, "w"), indent=1)
for m, s in stats["delta1"]["policies"].items():
    f = stats["delta0"]["policies"][m]["ingest"]; d = s["ingest"]
    print(m, "full parse: %.0f MB/s, %.1f CPU s | delta: %.0f MB/s, %.1f CPU s, %.1f%% of the bytes parsed" % (
        f["logical_mb_per_s"], f["cpu_s"], d["logical_mb_per_s"], d["cpu_s"], 100 * s["parsed_bytes"] / s["wide_chars"]), d["seconds"], f["seconds"])
assert not differ, differ
print("INGEST_CMP_OK")
