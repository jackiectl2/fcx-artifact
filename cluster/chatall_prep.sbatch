#!/bin/bash
# Package copy: submit from ${PROJECT_ROOT} after `mkdir -p logs`, e.g.
#   sbatch --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" [options below] cluster/chatall_prep.sbatch
# PROJECT_ROOT must be exported (sbatch exports the environment by default); see README.md and .env.example.
# E-SCALE: all 24,498 eligible WildChat-1M conversations (turn >= 10) into data/chat_pools_all/chat_pool_s1.parquet
# (fcx/chat_prep.py, unchanged), captured with the prefix manager only (fcx/capture.py, the agent-framework image) into data/cap_chatall_s1, then the paper ingestion (delta, 64 interleaved traces, seed 1).
# usage: sbatch --job-name=fcx-chatall --time=HH:MM:SS cluster/chatall_prep.sbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=14G
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
ROOT=${PROJECT_ROOT:?set PROJECT_ROOT to the package root}
RAW=$ROOT/data/raw/wildchat-1m-7d6490e462285cf85d91eabea0f9a954fbddcd1f
FW=${FW_SIF:-$ROOT/envs/agent-fw-py312-v1.sif}   # the agent-framework image (requirements-capture.txt)
DATA=$ROOT/data/cap_chatall_s1
test ! -e "$DATA" || { echo "refusing to overwrite $DATA"; exit 2; }
WORK=$(mktemp -d -u --tmpdir "fcx-${SLURM_JOB_ID}.XXXXXX")   # a name under $TMPDIR (node-local disk) or the system default; created below
SIF_BIND=${SIF_BIND:-$ROOT,$WORK}   # paths bound into the containers
mkdir -p "$WORK"; trap 'rm -rf "$WORK"' EXIT
[ -z "${SINGULARITY_MODULE:-}" ] || module load "$SINGULARITY_MODULE"
cd "$ROOT/fcx"
date -Is; hostname; md5sum chat_prep.py; cat capture.py ingest.py ingest_run.py convert.py | md5sum
RUN="singularity exec --bind $SIF_BIND --env TMPDIR=$WORK $ROOT/envs/cpu-data-py312-v1.sif python"
FWRUN="singularity exec --bind $SIF_BIND --env GIT_PYTHON_REFRESH=quiet,SWE_AGENT_CONFIG_DIR=/tmp,SWE_AGENT_TOOLS_DIR=/tmp,SWE_AGENT_TRAJECTORY_DIR=/tmp,TIKTOKEN_CACHE_DIR=$TIKTOKEN_CACHE_DIR,TMPDIR=$WORK $FW python"
$RUN chat_prep.py --src "$RAW/train-*.parquet" --out "$ROOT/data/chat_pools_all" --min-turns 10 --per-sample 24498 --samples 1 --seed 1
$FWRUN capture.py --src "$ROOT/data/chat_pools_all/chat_pool_s1.parquet" --format swegym --out "$DATA" --managers prefix --sample 0 --seed 1
$RUN ingest_run.py --data "$DATA" --managers prefix --work "$WORK" --delta 1 --interleave 64 --seed 1
du -sh "$DATA"
echo CHATALL_PREP_OK; date -Is
