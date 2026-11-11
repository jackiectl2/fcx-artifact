# FCX supplementary material

Code, job scripts and derived results for the paper *FCX: Factored Querying of Evolving LLM-Agent Contexts*. With this package
you can (1) regenerate the paper's data figures and tables from the shipped derived results on a laptop, and (2) rerun the
pipeline from the public source datasets on a Slurm cluster with Singularity/Apptainer and ClickHouse.

## Layout

```
README.md                 this file
LICENSE                   MIT license of the code (the derived results keep the terms of their source datasets; see Data)
requirements.txt          Python packages of fcx/ and figures/
requirements-capture.txt  agent frameworks for fcx/capture.py only
.env.example              environment variables used by the job scripts (names and meanings only)
fcx/                      pipeline code and unit tests
cluster/                  Slurm job scripts (templated) and task lists (cluster/tasks/*.txt)
figures/                  figure, table and summary scripts; style.py holds the shared paths and plotting style
results/                  derived results (small JSON files) read by the scripts in figures/
paper/figures/            output directory of the scripts in figures/ (empty)
```

`fcx/`: `capture.py` (requests of framework context managers over recorded trajectories), `replay.py` and `convert.py`
(replayed policies over full corpora), `ingest.py` (differencer, dictionary), `ingest_run.py` (online ingestion with delta
parsing), `ingest_cmp.py` (delta against full parsing), `ingest_ablate.py` (ingestion baselines and ablations, E-ING),
`query.py` (SQL of every plan), `bench.py` (benchmark driver against ClickHouse), `index_report.py` (fixed analysis of the
index stage), `wiki_convert.py`, `slice_sample.py`, `wiki_pilot.py` (Simple English Wikipedia workload, E-WIKI),
`chat_prep.py` and `chat_overlap.py` (WildChat-1M samples and their pairwise overlap, E-CHAT), `ingest_online.py` (per-request latency and active-trace state of the online
ingestion, E-ONLINE), `duck_bench.py` (the content-query comparison on DuckDB, E-DUCK and E-DUCK2), and tests `test_*.py`
(`m0_convert.py` is included only as the oracle of `test_replay.py`). `ingest.py` is the final implementation; `ingest_v1.py`
is the earlier implementation of the same algorithm, kept as the reference of `check_ingest_v2.py` (side-by-side comparison on
real request streams) and `test_ingest_v2.py` (differential property tests). `test_dord.py` tests the two dictionary layouts
of the index stage on a job-local ClickHouse server (`bench.py --dict-order`, E-DORD); `prof_derive.py` is a profiling aid.

## Regenerate the figures and tables from the shipped results

Run from the package root:

```
python3 -m venv .venv
.venv/bin/pip install numpy matplotlib
.venv/bin/python figures/fig2_workload.py          # Figure 2
.venv/bin/python figures/fig3_c2.py                # Figure 3
.venv/bin/python figures/fig4_encodings.py         # Figure 4
.venv/bin/python figures/fig5_ingest.py            # Figure 5
.venv/bin/python figures/fig6_supplement.py        # Figure 6
.venv/bin/python figures/fig7_ingest_baselines.py  # Figure 7
.venv/bin/python figures/tab2_exact.py             # Table II body  -> paper/figures/tab2_exact.tex
.venv/bin/python figures/tab4_managers.py          # Table III body -> paper/figures/tab4_managers.tex
.venv/bin/python figures/tab3_cost.py              # Table IV body  -> paper/figures/tab3_cost.tex
```

| paper | script | inputs (under results/) |
|---|---|---|
| Fig. 2 | figures/fig2_workload.py | r1_2026-10-03_late, k2_trace |
| Fig. 3 | figures/fig3_c2.py | r1_2026-10-03_late, r1_2026-10-24 (index runs) |
| Fig. 4 | figures/fig4_encodings.py | r1_2026-10-03_late (checks run, ingest_cmp) |
| Fig. 5 | figures/fig5_ingest.py | r1_2026-10-03_late, r1_2026-10-24 (ingest_cmp, index runs) |
| Fig. 6 | figures/fig6_supplement.py (with figures/mixed_index_summary.py) | r1_2026-11-01, r1_2026-10-24 |
| Fig. 7 | figures/fig7_ingest_baselines.py | ing_s1_v1, ing_s1_rep2, ing_s1_rep3, ing_s23_v1 |
| Table II | figures/tab2_exact.py | r1_2026-10-03_late, r1_2026-10-24, r1_2026-11-01 |
| Table III | figures/tab4_managers.py | r1_2026-10-03_late, r1_2026-10-24 |
| Table IV | figures/tab3_cost.py | r1_2026-10-03_late, r1_2026-10-24 |

Figure 1 is a diagram and Table I is written directly in the LaTeX source; neither is generated here. The table file names keep their earlier
numbering (tab3_cost.py makes Table IV and tab4_managers.py Table III). `figures/tab2_wiki.py` writes a Wikipedia row for
Table II from `results/wiki_summary.json`, which is included. `figures/tab2_chat.py` writes the WildChat row from
`results/chat_summary.json`.

The E-ING numbers quoted in Section 6 (ingestion baselines and ablations) come from `figures/ing_ext_summary.py`, an aggregator
that was frozen and reviewed before it was run on the results; `results/ing_ext_summary.json` is its output, and Figure 7
draws the same quantities. `figures/wiki_summary.py`, `figures/chat_summary.py` and `figures/online_duck_summary.py` play
the same role for the Wikipedia workload, the WildChat workload, and the online-ingestion and DuckDB measurements;
`figures/scale_summary.py` compares the 10x Wikipedia sample and all 24,498 WildChat conversations (E-SCALE) with the
3,000-item samples they contain. `figures/dord_summary.py` summarizes the paired runs of the index stage with the dictionary
sorted by message ID and clustered by first call (E-DORD, `results/dord/`), and `figures/fast_summary.py` the ingestion timings
of the final implementation (E-FAST: `results/ing_*_v2*`, `online_v2`, `fast_cmp_v2`); with `FCX_INGEST_V2=1`,
`figures/fig5_ingest.py`, `figures/tab3_cost.py` and `figures/breakeven.py` take FCX's ingestion records from
`results/fast_cmp_v2` (checked by `fast_summary.ingest_cmp()`), and `figures/fig7_ingest_baselines.py --impl v2` draws Figure 7
from the v2 runs. They refer to protocol documents
(`refine-logs/...`) that are not part of this package. Tables II-IV are generated by the table scripts above. Other numbers
quoted in the text from the earlier runs were computed by analysis scripts that are not part of this package.

Checks that run on a laptop:

```
.venv/bin/python figures/test_ing_ext_summary.py   # aggregator on synthetic damaged trees built from results/ing_s1_v1
.venv/bin/python figures/test_wiki_summary.py      # E-WIKI summary on synthetic inputs
.venv/bin/python figures/test_chat_summary.py      # E-CHAT summary on synthetic inputs
.venv/bin/python figures/test_scale_summary.py     # E-SCALE summary on synthetic inputs
.venv/bin/python figures/test_dord_summary.py      # E-DORD summary on synthetic pairs
.venv/bin/python figures/test_fast_summary.py      # E-FAST summary on fixtures built from the v1 results
.venv/bin/python figures/chat_summary.py --out <somewhere>/chat_summary.json            # recompute E-CHAT
.venv/bin/python figures/online_duck_summary.py --out <somewhere>/online_duck_summary.json   # recompute E-ONLINE, E-DUCK, E-DUCK2
.venv/bin/python figures/ing_ext_summary.py --out <somewhere>/ing_ext_summary.json   # recompute and compare
```

The unit tests of `fcx/` need the pipeline packages (`pip install -r requirements.txt`) and are run from inside `fcx/`, e.g.
`cd fcx && python test_ingest.py` (also `test_ablate.py`, `test_replay.py`, `test_boundary.py`, `test_wiki_convert.py`,
`test_chat_prep.py`, and `test_duck_bench.py`, which needs `duckdb`).

## Environment for the pipeline

The paper's runs used three Singularity images, kept as `${PROJECT_ROOT}/envs/<name>.sif` (no image is included here):

- `cpu-data-py312-v1.sif`: data processing, ingestion and the benchmark driver. Build it from the official Python image with the
  packages of `requirements.txt`, in a sandbox on node-local disk (`$WORK` is any scratch directory):

  ```
  singularity build --sandbox "$WORK/sb" docker://python:3.12-slim
  singularity exec --writable --no-home "$WORK/sb" python -m pip install --no-cache-dir -r requirements.txt
  singularity build "${PROJECT_ROOT}/envs/cpu-data-py312-v1.sif" "$WORK/sb"
  ```

  (Unprivileged builds may first need `find "$WORK/sb" -type d ! -perm -u+w -exec chmod u+w {} +`.)
- `clickhouse-26.8.15.10.sif`: the database server, pulled from the official image:

  ```
  singularity pull "${PROJECT_ROOT}/envs/clickhouse-26.8.15.10.sif" docker://clickhouse/clickhouse-server:26.8.15.10
  ```
- `agent-fw-py312-v1.sif`: only for `fcx/capture.py`; same recipe with `requirements-capture.txt`. Jobs run without network, so
  cache the tiktoken encodings `cl100k_base` and `o200k_base` inside the image (set `TIKTOKEN_CACHE_DIR` to a directory in the
  image while fetching and while running), and point `SWE_AGENT_CONFIG_DIR`, `SWE_AGENT_TOOLS_DIR` and
  `SWE_AGENT_TRAJECTORY_DIR` at a scratch directory when running capture (`GIT_PYTHON_REFRESH=quiet` silences GitPython).

## Data

Agent trajectories: four public Hugging Face datasets, stored as `${PROJECT_ROOT}/data/trajectories/<org>__<name>/`:

| corpus in the paper | dataset | files read | license tag recorded at capture time |
|---|---|---|---|
| SWE-agent | `nebius/SWE-agent-trajectories` | `data/train-*.parquet` | cc-by-4.0 |
| SWE-smith (tool split) | `SWE-bench/SWE-smith-trajectories` | `data/tool-*.parquet` | mit |
| SWE-rebench (OpenHands runs) | `nebius/SWE-rebench-openhands-trajectories` | `trajectories.parquet` | cc-by-4.0 |
| SWE-Gym (OpenHands runs) | `SWE-Gym/OpenHands-Sampled-Trajectories` | `data/*.parquet` | none given |

Download, e.g., with `huggingface_hub.snapshot_download(repo_id=<dataset>, repo_type="dataset", local_dir=<that directory>)`, and
check the current terms of each dataset.

Simple English Wikipedia, dump 20261101 (E-WIKI), from `https://dumps.wikimedia.org/simplewiki/20261101/` or a Wikimedia dump
mirror, verified by SHA-1 (`cluster/wiki_fetch.sbatch` downloads and verifies both into `${PROJECT_ROOT}/data/raw/simplewiki-20261101/`):

- `simplewiki-20261101-stub-meta-history.xml.gz`, SHA-1 `c475f263c78db6f1e538ee43443d2ed43cb01cad`
- `simplewiki-20261101-pages-meta-history.xml.bz2`, SHA-1 `4a6fc322490386d966fb4cd0e4eea09c611e6b4d`

WildChat-1M (E-CHAT; Zhao et al., ICLR 2024), Hugging Face dataset `allenai/WildChat-1M` at commit
`7d6490e462285cf85d91eabea0f9a954fbddcd1f` (license ODC-BY 1.0), 14 Parquet files checked against the SHA-256 values listed by
the Hugging Face tree API for that commit (`cluster/chat_fetch.sbatch` downloads and verifies them into
`${PROJECT_ROOT}/data/raw/wildchat-1m-<commit>/`). `fcx/chat_prep.py` keeps only each message's role and text.

Licenses of the derived results: the JSON files under `results/` hold counts and timings, and the index runs also hold their
query predicates, short substrings (8-32 characters) cut from the source datasets. Those substrings remain under the terms of
their sources: CC BY-SA 4.0 / GFDL (Simple English Wikipedia, Wikimedia Foundation), ODC-BY 1.0 (WildChat-1M, AI2), and the
licenses listed above for the agent trajectories.

## Pipeline

Set the variables of `.env.example`, create `${PROJECT_ROOT}/logs`, and submit from `${PROJECT_ROOT}` with
`sbatch --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" ...` (account and partition are not hard-coded). Each script
documents its own variables in its usage line; job outputs go to `logs/` relative to the submission directory.

1. **Capture and replay** (inputs of every later step). Capture samples, 3,000 trajectories per corpus and seed (1, 2, 3),
   requests of the framework context managers plus append only:
   `python fcx/capture.py --src "<parquet file or glob>" --format <sweagent|smith|openhands|swegym> --out "${PROJECT_ROOT}/data/cap_<swea|smith|oh|gym>_s<seed>" --managers <m1:m2:...> --sample 3000 --seed <seed>`
   (managers per corpus: `cluster/tasks/s1.txt`). Full corpora under the four replayed policies:
   `python fcx/convert.py --src "<parquet file or glob>" --out "${PROJECT_ROOT}/data/<name>" --policies prefix,window32,amortized32,elide5`
   (see `python fcx/convert.py --help` for the trajectory-column options). Wikipedia: `cluster/wiki_fetch.sbatch`, then
   `cluster/wiki_prep.sbatch` (runs `fcx/wiki_convert.py` into `data/wiki_s{1,2,3}`).
2. **Ingest**: `python fcx/ingest_run.py --data <capture dir> --managers <m1:m2:...> --work <node-local dir> --delta 1 --interleave 64 --seed 1`
   writes the dictionary, transition log, checkpoints and arrays into the capture directory; `fcx/ingest_cmp.py` checks that delta
   and full parsing write the same store. Wikipedia: `cluster/wiki_ingest.sbatch`. WildChat: `cluster/chat_fetch.sbatch`,
   `cluster/chat_prep.sbatch` (three disjoint samples), `cluster/chat_capture.sbatch` (capture of the four managers and
   ingestion, array 0-2).
3. **Ingestion baselines and ablations (E-ING)**: `cluster/ing_preflight.sbatch` (unit tests plus `cluster/tasks/sanity.txt`;
   writes a marker with the md5 of the code), then `cluster/ing_ablate.sbatch` as an array over `cluster/tasks/s1.txt` (three
   repetitions: `results/ing_s1_v1`, `ing_s1_rep2`, `ing_s1_rep3`), `cluster/tasks/s23.txt` (`results/ing_s23_v1`) and
   `cluster/tasks/wiki.txt` with `DATA_ROOT=$PROJECT_ROOT/data` (`results/ing_wiki_v1`). Results land in `${PROJECT_ROOT}/runs/<TAG>/`.
4. **Benchmark stages**: `cluster/fcx_bench.sbatch` starts a job-local ClickHouse server and runs `fcx/bench.py` on one ingested
   directory, with the stages and options passed in `EXTRA`, e.g. `--stages base:build:any:point:agg:bytes:life` (checks runs),
   `--stages base:build:bytes:index --seed <s>` (index runs), `--stages base:build:mixed` (running example) or
   `--stages base:build:fallback` (predicates across message boundaries). Each result records its settings, stages and per-stage
   times in its `meta` block; `bench.py` also records the node's host name there (removed from the shipped results).
   WildChat runs use `--workload chat` (`cluster/tasks/chat_checks.txt` and `chat_index.txt` through `cluster/bench_array.sbatch`).
   E-SCALE: `cluster/wikix_prep.sbatch` and `cluster/chatall_prep.sbatch`, then the same checks and index flags with
   `--preds-from` the corresponding 3,000-item index run (the same 42 predicates). E-ONLINE: `cluster/ing_online.sbatch` over `cluster/tasks/online.txt`; E-DUCK: `cluster/duck_bench.sbatch` with the predicate
   files `results/duck/preds_<sample>.json` (the held-out predicates and ClickHouse result counts of the paper's index runs);
   E-DUCK2: `cluster/duck_bench2.sbatch` on the Wikipedia (`MANAGER=wiki`) and WildChat (`MANAGER=prefix`) samples in
   `$PROJECT_ROOT/data`, with `results/duck/preds_<sample>__<manager>.json` (results `results/duck_v2`).
   E-DORD: `cluster/bench_pair.sbatch` runs one index-run command twice in one allocation, with `--dict-order id` and
   `--dict-order first` (order alternating across array elements), over `cluster/tasks/dord_pairs1.txt`, `dord_large.txt` and
   `dord_managers.txt` (`@ROOT@` in the task files stands for the project root and is substituted by the script).
   E-FAST: `cluster/fast_check.sbatch` (`fcx/check_ingest_v2.py` on `cluster/tasks/fast_check.txt`), then the preflight and
   `cluster/ing_ablate_multi.sbatch` over `cluster/tasks/fast_ablate.txt` (all E-ING streams under the v2 tags),
   `cluster/ing_online.sbatch` with `TAG=online_v2`, and `cluster/fast_cmp.sbatch` over `cluster/tasks/fast_cmp.txt`.
5. **Summaries**: `figures/ing_ext_ledger.py` (attempt ledger from a Slurm accounting extract), `figures/ing_ext_summary.py`
   (E-ING), `figures/wiki_summary.py` (E-WIKI), `figures/chat_summary.py` (E-CHAT), `figures/online_duck_summary.py` (E-ONLINE,
   E-DUCK, E-DUCK2), `figures/dord_summary.py` (E-DORD), `figures/fast_summary.py` (E-FAST), `fcx/index_report.py` (index runs).
6. **Figures and tables**: the scripts listed above, after copying the result files into `results/` with the layout used here.

Container paths are bound with `SIF_BIND`, and each job names its work directory with `mktemp` under `$TMPDIR` (set it to
node-local disk if your cluster's default temporary directory is shared).

## What was changed for this package

- Result JSON: keys named `host` were removed (198 occurrences); absolute paths in `data`/`src` fields were cut to the
  package layout (for example `data/cap_swea_s1`; 416 values); sampled predicate texts and query-plan texts that contain a
  path-like token or match an identifier pattern of the release check (all are substrings sampled from the public
  trajectories) were replaced by a placeholder (175 values in 82 files). No script in `figures/` reads
  those texts (they use predicate names). All other values are unchanged.
- `results/ing_ext_sacct.txt` keeps the columns JobID, State, ExitCode, Elapsed, Start and End; `figures/ing_ext_ledger.py` was
  adapted to that layout.
- Job scripts: absolute paths became `${PROJECT_ROOT}` / `${DATA_ROOT}`, account and partition lines were removed, logs go to
  `logs/`, work directory names come from `mktemp`, container bind paths come from `SIF_BIND`, the singularity module is
  loaded only if `SINGULARITY_MODULE` is set, and job names start with `fcx-`. `fcx_bench.sbatch` names the ClickHouse
  temporary directory `$WORK/ch/tmp_files/`.
- `figures/style.py` merges the two plotting styles the scripts were written against (see its docstring); `fig2_workload.py`
  reads the K2 pilot results from `results/k2_trace/`.
