#!/bin/bash
# Package copy: submit from ${PROJECT_ROOT} after `mkdir -p logs`, e.g.
#   sbatch --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" [options below] cluster/chat_capture.sbatch
# PROJECT_ROOT must be exported (sbatch exports the environment by default); see README.md and .env.example.
# E-CHAT step 3: for one sample (array element k -> cap_chat_s<k+1>), drive the context managers prefix, autogen_buf32,
# autogen_ht and lc_trim32k (fcx/capture.py, the frameworks' own code in the agent-framework image) over the sampled conversations, then ingest every manager's requests with the paper's ingestion (fcx/ingest_run.py:
# delta parsing, 64 interleaved traces, arrival seed 1). Refuses to overwrite an existing capture directory.
# usage: sbatch --array=0-2%3 --job-name=fcx-chatcap --time=HH:MM:SS cluster/chat_capture.sbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
ROOT=${PROJECT_ROOT:?set PROJECT_ROOT to the package root}
FW=${FW_SIF:-$ROOT/envs/agent-fw-py312-v1.sif}   # the agent-framework image (requirements-capture.txt)
S=$((SLURM_ARRAY_TASK_ID + 1))
DATA=$ROOT/data/cap_chat_s$S
MANAGERS=prefix:autogen_buf32:autogen_ht:lc_trim32k
test ! -e "$DATA" || { echo "refusing to overwrite $DATA"; exit 2; }
WORK=$(mktemp -d -u --tmpdir "fcx-${SLURM_JOB_ID}.XXXXXX")   # a name under $TMPDIR (node-local disk) or the system default; created below
SIF_BIND=${SIF_BIND:-$ROOT,$WORK}   # paths bound into the containers
mkdir -p "$WORK"; trap 'rm -rf "$WORK"' EXIT
[ -z "${SINGULARITY_MODULE:-}" ] || module load "$SINGULARITY_MODULE"
cd "$ROOT/fcx"
date -Is; hostname; cat capture.py ingest.py ingest_run.py convert.py | md5sum; sha256sum "$FW" | cut -c1-16
FWRUN="singularity exec --bind $SIF_BIND --env GIT_PYTHON_REFRESH=quiet,SWE_AGENT_CONFIG_DIR=/tmp,SWE_AGENT_TOOLS_DIR=/tmp,SWE_AGENT_TRAJECTORY_DIR=/tmp,TIKTOKEN_CACHE_DIR=$TIKTOKEN_CACHE_DIR,TMPDIR=$WORK $FW python"
RUN="singularity exec --bind $SIF_BIND --env TMPDIR=$WORK $ROOT/envs/cpu-data-py312-v1.sif python"
$FWRUN capture.py --src "$ROOT/data/chat_pools/chat_pool_s$S.parquet" --format swegym --out "$DATA" --managers "$MANAGERS" --sample 0 --seed "$S"
$RUN ingest_run.py --data "$DATA" --managers "$MANAGERS" --work "$WORK" --delta 1 --interleave 64 --seed 1
python3 -c "import json;d=json.load(open('$DATA/convert_stats.json'));print('spans',d['spans'],'messages',d['distinct_messages']);[print(m,p['transitions_per_span'],p['mean_context_len']) for m,p in d['policies'].items()]"
du -sh "$DATA"
echo CHAT_CAPTURE_OK "$DATA"; date -Is
