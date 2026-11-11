#!/bin/bash
# Package copy: submit from ${PROJECT_ROOT} after `mkdir -p logs`, e.g.
#   sbatch --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" [options below] cluster/ing_ablate_multi.sbatch
# PROJECT_ROOT must be exported (sbatch exports the environment by default); see README.md and .env.example.
# E-FAST: one array over ingestion-ablation runs of several tags. Line N+1 of $TASKS: "<TAG> <DATA_ROOT or -> <capture dir> <manager>"
# ("-": the DATA_ROOT of the environment, the capture samples). Each element runs the unchanged body of
# cluster/ing_ablate.sbatch (preflight gate $PREFLIGHT, refuses to overwrite runs/<TAG>/<capture>__<manager>.json) on that one line.
# usage: sbatch --array=0-K%N --job-name=fcx-ingabl --time=HH:MM:SS --export=ALL,TASKS=<file>,PREFLIGHT=<tag> cluster/ing_ablate_multi.sbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
ROOT=${PROJECT_ROOT:?set PROJECT_ROOT to the package root}
read -r TAG DR DATA MANAGER <<< "$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${TASKS:?}")"
DR=${DR//@ROOT@/$ROOT}   # task files write the project root as @ROOT@
[ -n "${MANAGER:-}" ] || { echo "bad task line"; exit 2; }
ONE=$(mktemp); trap 'rm -f "$ONE"' EXIT
echo "$DATA $MANAGER" > "$ONE"
echo "task $SLURM_ARRAY_TASK_ID: $TAG $DR $DATA $MANAGER"
if [ "$DR" = "-" ]; then
  SLURM_ARRAY_TASK_ID=0 TASKS="$ONE" TAG="$TAG" bash "$ROOT/cluster/ing_ablate.sbatch"
else
  SLURM_ARRAY_TASK_ID=0 TASKS="$ONE" TAG="$TAG" DATA_ROOT="$DR" bash "$ROOT/cluster/ing_ablate.sbatch"
fi
