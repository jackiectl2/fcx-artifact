#!/bin/bash
# Package copy: submit from ${PROJECT_ROOT} after `mkdir -p logs`, e.g.
#   sbatch --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" [options below] cluster/chat_slice.sbatch
# PROJECT_ROOT must be exported (sbatch exports the environment by default); see README.md and .env.example.
# E-CHAT sanity input: copy the first 300 traces of cap_chat_s1 (prefix only) into cap_chat_s1_k300 and ingest it with the paper's
# ingestion (delta parsing, 64 interleaved traces, seed 1). Numbers of runs on it are not reported.
# usage: sbatch --job-name=fcx-chatslice --time=HH:MM:SS cluster/chat_slice.sbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
ROOT=${PROJECT_ROOT:?set PROJECT_ROOT to the package root}
WORK=$(mktemp -d -u --tmpdir "fcx-${SLURM_JOB_ID}.XXXXXX")   # a name under $TMPDIR (node-local disk) or the system default; created below
SIF_BIND=${SIF_BIND:-$ROOT,$WORK}   # paths bound into the containers
mkdir -p "$WORK"; trap 'rm -rf "$WORK"' EXIT
[ -z "${SINGULARITY_MODULE:-}" ] || module load "$SINGULARITY_MODULE"
cd "$ROOT/fcx"
RUN="singularity exec --bind $SIF_BIND --env TMPDIR=$WORK $ROOT/envs/cpu-data-py312-v1.sif python"
date -Is; hostname
$RUN slice_sample.py --src "$ROOT/data/cap_chat_s1" --dst "$ROOT/data/cap_chat_s1_k300" --traces 300 --manager prefix
$RUN ingest_run.py --data "$ROOT/data/cap_chat_s1_k300" --managers prefix --work "$WORK" --delta 1 --interleave 64 --seed 1
echo CHAT_SLICE_OK; date -Is
