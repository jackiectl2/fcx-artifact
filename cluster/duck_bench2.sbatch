#!/bin/bash
# Package copy: submit from ${PROJECT_ROOT} after `mkdir -p logs`, e.g.
#   sbatch --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" [options below] cluster/duck_bench2.sbatch
# PROJECT_ROOT must be exported (sbatch exports the environment by default); see README.md and .env.example.
# E-DUCK2: the content-query comparison on DuckDB (fcx/duck_bench.py, unchanged) for one non-agent sample of this project
# (wiki_s<k> with manager wiki, cap_chat_s<k> with manager prefix), with the 30 held-out predicates and ClickHouse result counts of
# the paper's index run of that sample (results/duck/preds_<sample>__<manager>.json). Results: runs/duck_v2/<sample>__<manager>.json.
# usage: sbatch --job-name=fcx-duck --time=HH:MM:SS --export=ALL,SAMPLE=<sample>,MANAGER=<wiki|prefix> cluster/duck_bench2.sbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
ROOT=${PROJECT_ROOT:?set PROJECT_ROOT to the package root}
[[ "${SAMPLE:-}" =~ ^(wiki_s[1-3]|cap_chat_s[1-3])$ && "${MANAGER:-}" =~ ^(wiki|prefix)$ ]] || { echo "bad SAMPLE or MANAGER"; exit 2; }
FINAL=$ROOT/runs/duck_v2/${SAMPLE}__${MANAGER}.json
test ! -e "$FINAL" || { echo "refusing to overwrite $FINAL"; exit 2; }
WORK=$(mktemp -d -u --tmpdir "fcx-${SLURM_JOB_ID}.XXXXXX")   # a name under $TMPDIR (node-local disk) or the system default; created below
SIF_BIND=${SIF_BIND:-$ROOT,$WORK}   # paths bound into the containers
mkdir -p "$WORK" "$ROOT/runs/duck_v2"; trap 'rm -rf "$WORK"' EXIT
[ -z "${SINGULARITY_MODULE:-}" ] || module load "$SINGULARITY_MODULE"
cd "$ROOT/fcx"
PREDS="$ROOT/results/duck/preds_${SAMPLE}__${MANAGER}.json"
date -Is; hostname; nproc; lscpu | grep -m1 "Model name"; md5sum duck_bench.py "$PREDS"
singularity exec --bind "$SIF_BIND" --env TMPDIR=$WORK "$ROOT/envs/cpu-data-py312-v1.sif" python duck_bench.py --data "$ROOT/data/$SAMPLE" --manager "$MANAGER" \
    --preds "$PREDS" --work "$WORK" --threads "$SLURM_CPUS_PER_TASK" --mem-gb 52 --out "$WORK/result.json" --cold 2 --warm 3
mv "$WORK/result.json" "$FINAL.part" && mv "$FINAL.part" "$FINAL"
echo DUCK_JOB_OK "$FINAL"; date -Is
