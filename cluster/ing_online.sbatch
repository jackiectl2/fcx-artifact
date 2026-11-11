#!/bin/bash
# Package copy: submit from ${PROJECT_ROOT} after `mkdir -p logs`, e.g.
#   sbatch --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" [options below] cluster/ing_online.sbatch
# PROJECT_ROOT must be exported (sbatch exports the environment by default); see README.md and .env.example.
# E-ONLINE: per-request derivation latency and active-trace state of FCX's ingestion (fcx/ingest_online.py), one array element
# per line of $TASKS: "<old|new> <capture dir> <manager> <interleave>". old = $DATA_ROOT (the capture samples, read only);
# new = $PROJECT_ROOT/data. Results go to runs/$TAG/<dir>__<manager>__k<interleave>.json.
# usage: sbatch --array=0-K%N --job-name=fcx-online --time=HH:MM:SS --export=ALL,TASKS=<file>,TAG=<tag> cluster/ing_online.sbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=14G
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
ROOT=${PROJECT_ROOT:?set PROJECT_ROOT to the package root}
OLD=${DATA_ROOT:?set DATA_ROOT to the directory that holds the capture samples}
[[ "${TAG:-}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || { echo "bad TAG"; exit 2; }
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASKS")
read -r -a F <<< "$LINE"
[ "${#F[@]}" -eq 4 ] || { echo "task line must have exactly four fields: '$LINE'"; exit 2; }
WHERE=${F[0]}; DATA=${F[1]}; MANAGER=${F[2]}; K=${F[3]}
[[ "$WHERE" =~ ^(old|new)$ && "$DATA" =~ ^(cap|wiki)_[a-z0-9_]+$ && "$MANAGER" =~ ^[a-z0-9_]+$ && "$K" =~ ^[1-9][0-9]*$ ]] || { echo "bad task fields: '$LINE'"; exit 2; }
BASE=$([ "$WHERE" = old ] && echo "$OLD" || echo "$ROOT/data")
OUT_DIR=$ROOT/runs/$TAG; FINAL=$OUT_DIR/${DATA}__${MANAGER}__k$K.json
test ! -e "$FINAL" || { echo "refusing to overwrite $FINAL"; exit 2; }
WORK=$(mktemp -d -u --tmpdir "fcx-${SLURM_JOB_ID}.XXXXXX")   # a name under $TMPDIR (node-local disk) or the system default; created below
SIF_BIND=${SIF_BIND:-$ROOT,$DATA_ROOT,$WORK}   # paths bound into the containers
mkdir -p "$WORK" "$OUT_DIR"; trap 'rm -rf "$WORK"' EXIT
[ -z "${SINGULARITY_MODULE:-}" ] || module load "$SINGULARITY_MODULE"
cd "$ROOT/fcx"
date -Is; hostname; lscpu | grep -m1 "Model name"; cat ingest.py ingest_run.py ingest_online.py | md5sum; echo "task $SLURM_ARRAY_TASK_ID: $LINE"
singularity exec --bind "$SIF_BIND" --env TMPDIR=$WORK "$ROOT/envs/cpu-data-py312-v1.sif" python ingest_online.py \
    --data "$BASE/$DATA" --manager "$MANAGER" --interleave "$K" --work "$WORK" --out "$WORK/result.json"
mv "$WORK/result.json" "$FINAL.part" && mv "$FINAL.part" "$FINAL"
echo ING_ONLINE_JOB_OK "$FINAL"; date -Is
