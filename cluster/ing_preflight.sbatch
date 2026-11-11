#!/bin/bash
# Package copy: submit from ${PROJECT_ROOT} after `mkdir -p logs`, e.g.
#   sbatch --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" [options below] cluster/ing_preflight.sbatch
# PROJECT_ROOT must be exported (sbatch exports the environment by default); see README.md and .env.example.
# E-ING preflight: unit tests (test_ingest.py, test_ablate.py), then fcx/ingest_ablate.py on every line of $TASKS in turn
# (small sanity captures), to check the code and measure per-task time before the full array is sized and submitted.
# On success it writes runs/$TAG/PREFLIGHT_OK holding the md5 of the code it tested; the array requires that marker.
# usage: sbatch --job-name=fcx-ingpre --time=HH:MM:SS --export=ALL,TASKS=<task file>,TAG=<run tag>,DATA_ROOT=<dir> cluster/ing_preflight.sbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
ROOT=${PROJECT_ROOT:?set PROJECT_ROOT to the package root}
DATA_ROOT=${DATA_ROOT:?set DATA_ROOT to the directory that holds the capture samples}
[[ "${TAG:-}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || { echo "bad TAG"; exit 2; }
FINAL_DIR=$ROOT/runs/$TAG; OUT_DIR=$ROOT/runs/.$TAG.partial-$SLURM_JOB_ID
test ! -e "$FINAL_DIR" || { echo "refusing to overwrite $FINAL_DIR"; exit 2; }
[ "$(grep -c . "$TASKS")" -ge 1 ] || { echo "no sanity tasks in $TASKS"; exit 2; }
WORK=$(mktemp -d -u --tmpdir "fcx-${SLURM_JOB_ID}.XXXXXX")   # a name under $TMPDIR (node-local disk) or the system default; created below
SIF_BIND=${SIF_BIND:-$ROOT,$DATA_ROOT,$WORK}   # paths bound into the containers
mkdir -p "$WORK" "$OUT_DIR"
trap 'rm -rf "$WORK"' EXIT
export TMPDIR=$WORK
date -Is; hostname; df -h "$WORK" | tail -1
[ -z "${SINGULARITY_MODULE:-}" ] || module load "$SINGULARITY_MODULE"
cd "$ROOT/fcx"
CODE_MD5=$(cat ingest.py ingest_run.py convert.py ingest_ablate.py | md5sum | cut -d' ' -f1)
RUN="singularity exec --bind $SIF_BIND $ROOT/envs/cpu-data-py312-v1.sif python"
$RUN test_ingest.py
$RUN test_ablate.py
while read -r DATA MANAGER EXTRA; do
  [ -z "${EXTRA:-}" ] && [[ "$DATA" =~ ^cap_[a-z0-9_]+$ && "$MANAGER" =~ ^[a-z0-9_]+$ ]] || { echo "bad task line"; exit 2; }
  t0=$(date +%s)
  $RUN ingest_ablate.py --data "$DATA_ROOT/$DATA" --manager "$MANAGER" --work "$WORK" --out "$OUT_DIR/${DATA}__${MANAGER}.json" < /dev/null
  echo "TASK_SECONDS $DATA $MANAGER $(( $(date +%s) - t0 ))"
  N_DONE=$(( ${N_DONE:-0} + 1 ))
done < "$TASKS"
[ "${N_DONE:-0}" -ge 1 ] || { echo "no sanity task ran"; exit 2; }
echo "$CODE_MD5" > "$OUT_DIR/PREFLIGHT_OK"
mv "$OUT_DIR" "$FINAL_DIR"
echo ING_PREFLIGHT_OK "$FINAL_DIR" "code $CODE_MD5"; date -Is
