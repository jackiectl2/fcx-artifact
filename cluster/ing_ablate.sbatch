#!/bin/bash
# Package copy: submit from ${PROJECT_ROOT} after `mkdir -p logs`, e.g.
#   sbatch --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" [options below] cluster/ing_ablate.sbatch
# PROJECT_ROOT must be exported (sbatch exports the environment by default); see README.md and .env.example.
# E-ING: ingestion baselines and ablations (fcx/ingest_ablate.py), one array element per (capture directory, manager).
# Line N+1 of $TASKS (N = SLURM_ARRAY_TASK_ID) is "<capture dir under $DATA_ROOT> <manager>", exactly two fields. DATA_ROOT
# is the directory that holds the capture samples (read in place, read-only); for E-WIKI pass DATA_ROOT=$PROJECT_ROOT/data.
# Results go to $PROJECT_ROOT/runs/$TAG/.
# The unit tests run in the preflight job (cluster/ing_preflight.sbatch); an element refuses to run unless
# runs/$PREFLIGHT/PREFLIGHT_OK holds the md5 of the current code.
# usage: sbatch --array=0-K%N --job-name=fcx-ingabl --time=HH:MM:SS --export=ALL,TASKS=<file>,TAG=<tag>,PREFLIGHT=<preflight tag>,DATA_ROOT=<dir> cluster/ing_ablate.sbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
ROOT=${PROJECT_ROOT:?set PROJECT_ROOT to the package root}
DATA_ROOT=${DATA_ROOT:?set DATA_ROOT to the directory that holds the capture samples}
[[ "${TAG:-}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || { echo "bad TAG"; exit 2; }
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASKS")
read -r -a F <<< "$LINE"
[ "${#F[@]}" -eq 2 ] || { echo "task line must have exactly two fields: '$LINE'"; exit 2; }
DATA=${F[0]}; MANAGER=${F[1]}
[[ "$DATA" =~ ^(cap|wiki)_[a-z0-9_]+$ && "$MANAGER" =~ ^[a-z0-9_]+$ ]] || { echo "bad task fields: '$LINE'"; exit 2; }
OUT_DIR=$ROOT/runs/$TAG; FINAL=$OUT_DIR/${DATA}__${MANAGER}.json
test ! -e "$FINAL" || { echo "refusing to overwrite $FINAL"; exit 2; }
WORK=$(mktemp -d -u --tmpdir "fcx-${SLURM_JOB_ID}.XXXXXX")   # a name under $TMPDIR (node-local disk) or the system default; created below
SIF_BIND=${SIF_BIND:-$ROOT,$DATA_ROOT,$WORK}   # paths bound into the containers
mkdir -p "$WORK" "$OUT_DIR"
trap 'rm -rf "$WORK"' EXIT
export TMPDIR=$WORK
FREE_GB=$(df -BG --output=avail "$WORK" | tail -1 | tr -dc 0-9)
[ "$FREE_GB" -ge 10 ] || { echo "node-local scratch ($WORK) has only ${FREE_GB}G free; need 10G for one dictionary"; exit 3; }
date -Is; hostname; echo "task $SLURM_ARRAY_TASK_ID: $DATA $MANAGER; node-local scratch free ${FREE_GB}G"
cd "$ROOT/fcx"
[[ "${PREFLIGHT:-}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || { echo "bad PREFLIGHT"; exit 2; }
CODE_MD5=$(cat ingest.py ingest_run.py convert.py ingest_ablate.py | md5sum | cut -d' ' -f1)
[ "$(cat "$ROOT/runs/$PREFLIGHT/PREFLIGHT_OK" 2>/dev/null)" = "$CODE_MD5" ] || { echo "no passing preflight for code $CODE_MD5"; exit 4; }
[ -z "${SINGULARITY_MODULE:-}" ] || module load "$SINGULARITY_MODULE"
RUN="singularity exec --bind $SIF_BIND $ROOT/envs/cpu-data-py312-v1.sif python"
$RUN ingest_ablate.py --data "$DATA_ROOT/$DATA" --manager "$MANAGER" --work "$WORK" --out "$WORK/result.json"
mv "$WORK/result.json" "$FINAL.part" && mv "$FINAL.part" "$FINAL"
echo ING_ABLATE_JOB_OK "$FINAL"; date -Is
