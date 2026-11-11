"""Build the E-ING attempt ledger (results/ing_ext_attempts.json) from Slurm accounting records.

Input: results/ing_ext_sacct.txt, the columns JobID|State|ExitCode|Elapsed|Start|End of
  sacct -X -D -P -n -j 65657751,65664263,65664264,65664265
keeping only the rows of these array jobs whose Start is in 2026 (-D also lists older jobs that reused these IDs). In this
package the job-name and node-list columns of the accounting extract were removed (the rows were filtered on the job name
before), and a state such as "CANCELLED by <uid>" would have been cut to its first word.
Each array job ran one task file (cluster/ing_ablate.sbatch: element N runs line N+1). Every attempt of every element becomes one
ledger entry; a requeued element appears in sacct -D once per attempt. The failure class of a non-completed attempt is
"infrastructure" for NODE_FAIL / PREEMPTED / BOOT_FAIL, "incomplete" for TIMEOUT / CANCELLED / OUT_OF_MEMORY, and "failed"
otherwise (an application error, which the protocol treats as a correctness failure that may not be rerun).
usage: .venv/bin/python figures/ing_ext_ledger.py
"""
import json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JOBS = {"65657751": ("ing_s1_v1", "s1.txt"), "65664264": ("ing_s1_rep2", "s1.txt"), "65664265": ("ing_s1_rep3", "s1.txt"),
        "65664263": ("ing_s23_v1", "s23.txt")}
CLASS = {"NODE_FAIL": "infrastructure", "PREEMPTED": "infrastructure", "BOOT_FAIL": "infrastructure",
         "TIMEOUT": "incomplete", "CANCELLED": "incomplete", "OUT_OF_MEMORY": "incomplete"}


def main():
    tasks = {f: [l.split() for l in open(ROOT / "cluster" / "tasks" / f) if l.strip()] for f in {t for _, t in JOBS.values()}}
    entries = {}
    for line in open(ROOT / "results" / "ing_ext_sacct.txt"):
        job_id, state, exit_code, elapsed, start, end = line.rstrip("\n").split("|")
        parent, element = job_id.split("_")
        assert start.startswith("2026-"), line
        run, task_file = JOBS[parent]
        data, manager = tasks[task_file][int(element)]
        state = state.split()[0]                                   # "CANCELLED by <uid>" -> CANCELLED
        key = (run, data, manager)
        entries.setdefault(key, []).append(dict(job=parent, element=int(element), state=state, exit_code=exit_code, elapsed=elapsed,
                                                start=start, end=end, failure_class=None if state == "COMPLETED" and exit_code == "0:0"
                                                else CLASS.get(state, "failed")))
    out = []
    for (run, data, manager), atts in sorted(entries.items()):
        atts.sort(key=lambda x: x["start"])
        for k, x in enumerate(atts, start=1):
            out.append(dict(run=run, data=data, manager=manager, attempt=k, result=f"results/{run}/{data}__{manager}.json", **x))
    dst = ROOT / "results" / "ing_ext_attempts.json"
    tmp = dst.with_suffix(".json.tmp")
    json.dump(dict(source="results/ing_ext_sacct.txt", entries=out), open(tmp, "w"), indent=1)
    os.replace(tmp, dst)
    print(len(out), "attempts;", sum(x["failure_class"] is None for x in out), "completed")


if __name__ == "__main__":
    main()
