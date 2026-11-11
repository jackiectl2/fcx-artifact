#!/bin/bash
# Package copy: submit from ${PROJECT_ROOT} after `mkdir -p logs`, e.g.
#   sbatch --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" [options below] cluster/chat_prep.sbatch
# PROJECT_ROOT must be exported (sbatch exports the environment by default); see README.md and .env.example.
# E-CHAT step 2: unit tests of the sample drawer, then three disjoint samples of 3,000 WildChat-1M conversations with at least
# 10 turns into data/chat_pools/ (fcx/chat_prep.py). Refuses to overwrite existing samples.
# usage: sbatch --job-name=fcx-chatprep --time=HH:MM:SS cluster/chat_prep.sbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
ROOT=${PROJECT_ROOT:?set PROJECT_ROOT to the package root}
RAW=$ROOT/data/raw/wildchat-1m-7d6490e462285cf85d91eabea0f9a954fbddcd1f
WORK=$(mktemp -d -u --tmpdir "fcx-${SLURM_JOB_ID}.XXXXXX")   # a name under $TMPDIR (node-local disk) or the system default; created below
SIF_BIND=${SIF_BIND:-$ROOT,$WORK}   # paths bound into the containers
mkdir -p "$WORK"; trap 'rm -rf "$WORK"' EXIT
export TMPDIR=$WORK
[ -z "${SINGULARITY_MODULE:-}" ] || module load "$SINGULARITY_MODULE"
cd "$ROOT/fcx"
RUN="singularity exec --bind $SIF_BIND $ROOT/envs/cpu-data-py312-v1.sif python"
date -Is; hostname; md5sum chat_prep.py test_chat_prep.py
$RUN test_chat_prep.py
$RUN chat_prep.py --src "$RAW/train-*.parquet" --out "$ROOT/data/chat_pools" --min-turns 10 --per-sample 3000 --samples 3 --seed 1
ls -la "$ROOT/data/chat_pools"
echo CHAT_PREP_JOB_OK; date -Is
