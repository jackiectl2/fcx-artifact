#!/bin/bash
# Package copy: submit from ${PROJECT_ROOT} after `mkdir -p logs`, e.g.
#   sbatch --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" [options below] cluster/fast_cmp.sbatch
# PROJECT_ROOT must be exported (sbatch exports the environment by default); see README.md and .env.example.
# E-FAST: fcx/ingest_cmp.py with the final ingestion code on one seed-1 capture sample ($DATA_ROOT, read only):
# ingest_run.py with full parsing and with delta parsing into node-local scratch; the two stores must be equal file by file.
# Line N+1 of $TASKS: "<capture dir> <managers a:b:c>". Result: runs/$TAG/<capture>__ingest_cmp.json (refuses to overwrite).
# usage: sbatch --array=0-K%N --job-name=fcx-fastcmp --time=HH:MM:SS --export=ALL,TASKS=<file>,TAG=<tag> cluster/fast_cmp.sbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
ROOT=${PROJECT_ROOT:?set PROJECT_ROOT to the package root}
DATA_ROOT=${DATA_ROOT:?set DATA_ROOT to the directory that holds the capture samples}
[[ "${TAG:-}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || { echo "bad TAG"; exit 2; }
read -r DATA MANAGERS <<< "$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${TASKS:?}")"
[[ "$DATA" =~ ^cap_[a-z0-9_]+$ && "$MANAGERS" =~ ^[a-z0-9_:]+$ ]] || { echo "bad task line"; exit 2; }
FINAL=$ROOT/runs/$TAG/${DATA}__ingest_cmp.json
test ! -e "$FINAL" || { echo "refusing to overwrite $FINAL"; exit 2; }
WORK=$(mktemp -d -u --tmpdir "fcx-${SLURM_JOB_ID}.XXXXXX")   # a name under $TMPDIR (node-local disk) or the system default; created below; mkdir -p "$WORK" "$ROOT/runs/$TAG"; trap 'rm -rf "$WORK"' EXIT
SIF_BIND=${SIF_BIND:-$ROOT,$DATA_ROOT,$WORK}   # paths bound into the containers
[ -z "${SINGULARITY_MODULE:-}" ] || module load "$SINGULARITY_MODULE"
cd "$ROOT/fcx"; date -Is; hostname; md5sum ingest.py ingest_run.py ingest_cmp.py convert.py; df -h "$WORK" | tail -1
singularity exec --bind "$SIF_BIND" "$ROOT/envs/cpu-data-py312-v1.sif" python ingest_cmp.py --data "$DATA_ROOT/$DATA" --managers "$MANAGERS" \
    --work "$WORK" --out "$WORK/result.json" --interleave 64
mv "$WORK/result.json" "$FINAL.part" && mv "$FINAL.part" "$FINAL"
echo FAST_CMP_OK "$FINAL"; date -Is
