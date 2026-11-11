#!/bin/bash
# Package copy: submit from ${PROJECT_ROOT} after `mkdir -p logs`, e.g.
#   sbatch --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" [options below] cluster/bench_pair.sbatch
# PROJECT_ROOT must be exported (sbatch exports the environment by default); see README.md and .env.example.
# E-DORD paired run: the same bench command twice in one allocation (same node), once per dictionary layout, OUT=<out>_<layout>.
# Element N runs line N+1 of $TASKS, "<DATA> <OUT> <driver flags...>" (without --dict-order). Even elements run id then first,
# odd elements first then id, so that the order alternates across samples. Each run is the unchanged fcx_bench.sbatch body
# (job-local ClickHouse server, refuses to overwrite runs/<DATA>/<OUT>_<layout>.json).
# usage: sbatch --array=0-K%N --job-name=fcx-dpair --time=HH:MM:SS --cpus-per-task=16 --mem=80G --export=ALL,TASKS=<file> cluster/bench_pair.sbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
ROOT=${PROJECT_ROOT:?set PROJECT_ROOT to the package root}
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${TASKS:?}")
LINE=${LINE//@ROOT@/$ROOT}   # task files write the project root as @ROOT@
read -r DATA OUT EXTRA <<< "$LINE"
[ -n "$DATA" ] && [ -n "$OUT" ] && [ -n "$EXTRA" ] || { echo "bad task line: '$LINE'"; exit 2; }
case "$EXTRA" in *--dict-order*) echo "task line must not set --dict-order"; exit 2;; esac
if [ $((SLURM_ARRAY_TASK_ID % 2)) -eq 0 ]; then ORDERS="id first"; else ORDERS="first id"; fi
echo "task $SLURM_ARRAY_TASK_ID: $LINE; layouts in order: $ORDERS"
export FCX_PAIR_ID="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}" FCX_PAIR_ORDER="${ORDERS// /,}"   # recorded in each run's meta
for L in $ORDERS; do
  echo "== layout $L"; date -Is
  DATA="$DATA" OUT="${OUT}_$L" EXTRA="$EXTRA --dict-order $L" bash "$ROOT/cluster/fcx_bench.sbatch"
done
echo PAIR_OK; date -Is
