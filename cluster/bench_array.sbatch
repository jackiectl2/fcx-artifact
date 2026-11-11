#!/bin/bash
# Package copy: submit from ${PROJECT_ROOT} after `mkdir -p logs`, e.g.
#   sbatch --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" [options below] cluster/bench_array.sbatch
# PROJECT_ROOT must be exported (sbatch exports the environment by default); see README.md and .env.example.
# Array wrapper around cluster/fcx_bench.sbatch: element N runs line N+1 of $TASKS, "<DATA> <OUT> <driver flags...>", with the
# same job body (job-local ClickHouse server, refuses to overwrite runs/<DATA>/<OUT>.json). Resources come from this script.
# usage: sbatch --array=0-K%N --job-name=fcx-bench --time=HH:MM:SS --cpus-per-task=16 --mem=..G --export=ALL,TASKS=<file> cluster/bench_array.sbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
ROOT=${PROJECT_ROOT:?set PROJECT_ROOT to the package root}
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${TASKS:?}")
read -r DATA OUT EXTRA <<< "$LINE"
[ -n "$DATA" ] && [ -n "$OUT" ] && [ -n "$EXTRA" ] || { echo "bad task line: '$LINE'"; exit 2; }
export DATA OUT EXTRA
echo "task $SLURM_ARRAY_TASK_ID: $LINE"
exec bash "$ROOT/cluster/fcx_bench.sbatch"
