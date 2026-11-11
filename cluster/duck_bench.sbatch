#!/bin/bash
# Package copy: submit from ${PROJECT_ROOT} after `mkdir -p logs`, e.g.
#   sbatch --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" [options below] cluster/duck_bench.sbatch
# PROJECT_ROOT must be exported (sbatch exports the environment by default); see README.md and .env.example.
# E-DUCK: the paper's content-query comparison on DuckDB (fcx/duck_bench.py) for one seed-1 capture sample, append only, with the
# 30 held-out predicates of the paper's ClickHouse run of that sample (results/duck/preds_<sample>.json).
# Stores are read from $DATA_ROOT (read only). Results: runs/duck_v1/<sample>__prefix.json.
# usage: sbatch --job-name=fcx-duck --time=HH:MM:SS --export=ALL,SAMPLE=cap_<corpus>_s1 cluster/duck_bench.sbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
ROOT=${PROJECT_ROOT:?set PROJECT_ROOT to the package root}
OLD=${DATA_ROOT:?set DATA_ROOT to the directory that holds the capture samples}
[[ "${SAMPLE:-}" =~ ^cap_[a-z]+_s1$ ]] || { echo "bad SAMPLE"; exit 2; }
FINAL=$ROOT/runs/duck_v1/${SAMPLE}__prefix.json
test ! -e "$FINAL" || { echo "refusing to overwrite $FINAL"; exit 2; }
WORK=$(mktemp -d -u --tmpdir "fcx-${SLURM_JOB_ID}.XXXXXX")   # a name under $TMPDIR (node-local disk) or the system default; created below
SIF_BIND=${SIF_BIND:-$ROOT,$DATA_ROOT,$WORK}   # paths bound into the containers
mkdir -p "$WORK" "$ROOT/runs/duck_v1"; trap 'rm -rf "$WORK"' EXIT
[ -z "${SINGULARITY_MODULE:-}" ] || module load "$SINGULARITY_MODULE"
cd "$ROOT/fcx"
date -Is; hostname; nproc; lscpu | grep -m1 "Model name"; md5sum duck_bench.py "$ROOT/results/duck/preds_$SAMPLE.json"
singularity exec --bind "$SIF_BIND" --env TMPDIR=$WORK "$ROOT/envs/cpu-data-py312-v1.sif" python duck_bench.py --data "$OLD/$SAMPLE" --manager prefix \
    --preds "$ROOT/results/duck/preds_$SAMPLE.json" --work "$WORK" --threads "$SLURM_CPUS_PER_TASK" --mem-gb 52 --out "$WORK/result.json" --cold 2 --warm 3
mv "$WORK/result.json" "$FINAL.part" && mv "$FINAL.part" "$FINAL"
echo DUCK_JOB_OK "$FINAL"; date -Is
