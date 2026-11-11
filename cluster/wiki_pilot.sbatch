#!/bin/bash
# Package copy: submit from ${PROJECT_ROOT} after `mkdir -p logs`, e.g.
#   sbatch --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" [options below] cluster/wiki_pilot.sbatch
# PROJECT_ROOT must be exported (sbatch exports the environment by default); see README.md and .env.example.
# E-WIKI sizing pilot (timings only, no data written): decompressed size of the full-history dump, the stub pass, and the
# parse rate of the converter's XML reader on the first 3 GB of the decompressed history.
# usage: sbatch --job-name=fcx-wikipilot --time=HH:MM:SS cluster/wiki_pilot.sbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
ROOT=${PROJECT_ROOT:?set PROJECT_ROOT to the package root}
SIF_BIND=${SIF_BIND:-$ROOT}   # paths bound into the containers
RAW=$ROOT/data/raw/simplewiki-20261101
HIST=$RAW/simplewiki-20261101-pages-meta-history.xml.bz2; STUB=$RAW/simplewiki-20261101-stub-meta-history.xml.gz
[ -z "${SINGULARITY_MODULE:-}" ] || module load "$SINGULARITY_MODULE"
cd "$ROOT/fcx"
RUN="singularity exec --bind $SIF_BIND $ROOT/envs/cpu-data-py312-v1.sif python"
date -Is; hostname
s=$(date +%s); echo "history decompressed bytes: $(lbzip2 -dc -n 6 "$HIST" | wc -c)  seconds: $(( $(date +%s) - s ))"
s=$(date +%s); echo "stub decompressed bytes: $(zcat "$STUB" | wc -c)  seconds: $(( $(date +%s) - s ))"
$RUN wiki_pilot.py "$STUB"
lbzip2 -dc -n 6 "$HIST" | head -c 3000000000 | $RUN wiki_pilot.py - || true
echo WIKI_PILOT_OK; date -Is
