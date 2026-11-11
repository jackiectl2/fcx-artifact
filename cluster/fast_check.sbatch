#!/bin/bash
# Package copy: submit from ${PROJECT_ROOT} after `mkdir -p logs`, e.g.
#   sbatch --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" [options below] cluster/fast_check.sbatch
# PROJECT_ROOT must be exported (sbatch exports the environment by default); see README.md and .env.example.
# E-FAST: equivalence of the sped-up fcx/ingest.py with the reference copy fcx/ingest_v1.py on one full request stream per array
# element (line N+1 of $TASKS: "<capture dir> <manager>"); prints CHECK_V2_OK or CHECK_V2_FAIL. Reads inputs only; writes nothing.
# usage: sbatch --array=0-K%N --job-name=fcx-fastchk --time=HH:MM:SS --export=ALL,TASKS=<file> cluster/fast_check.sbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=14G
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
ROOT=${PROJECT_ROOT:?set PROJECT_ROOT to the package root}
read -r DATA MANAGER <<< "$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${TASKS:?}")"
DATA=${DATA//@ROOT@/$ROOT}; DATA=${DATA//@DATA_ROOT@/${DATA_ROOT:?set DATA_ROOT to the directory that holds the capture samples}}
WORK=$(mktemp -d -u --tmpdir "fcx-${SLURM_JOB_ID}.XXXXXX")   # a name under $TMPDIR (node-local disk) or the system default; created below; mkdir -p "$WORK"; trap 'rm -rf "$WORK"' EXIT
SIF_BIND=${SIF_BIND:-$ROOT,${DATA_ROOT:-$ROOT},$WORK}   # paths bound into the containers
[ -z "${SINGULARITY_MODULE:-}" ] || module load "$SINGULARITY_MODULE"
cd "$ROOT/fcx"; date -Is; hostname; md5sum ingest.py ingest_v1.py check_ingest_v2.py
singularity exec --bind "$SIF_BIND" "$ROOT/envs/cpu-data-py312-v1.sif" python check_ingest_v2.py --data "$DATA" --manager "$MANAGER" --work "$WORK"
date -Is
