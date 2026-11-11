#!/bin/bash
# Package copy: submit from ${PROJECT_ROOT} after `mkdir -p logs`, e.g.
#   sbatch --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" [options below] cluster/fcx_test.sbatch
# PROJECT_ROOT must be exported (sbatch exports the environment by default); see README.md and .env.example.
# Run unit tests of fcx/ in the project's CPU image (synthetic inputs only; no research data is read or written).
# usage: sbatch --job-name=fcx-test --time=HH:MM:SS --export=ALL,TESTS="test_wiki_convert.py test_ingest.py" cluster/fcx_test.sbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
ROOT=${PROJECT_ROOT:?set PROJECT_ROOT to the package root}
WORK=$(mktemp -d -u --tmpdir "fcx-${SLURM_JOB_ID}.XXXXXX")   # a name under $TMPDIR (node-local disk) or the system default; created below
SIF_BIND=${SIF_BIND:-$ROOT,$WORK}   # paths bound into the containers
mkdir -p "$WORK"
trap 'rm -rf "$WORK"' EXIT
export TMPDIR=$WORK
[ -z "${SINGULARITY_MODULE:-}" ] || module load "$SINGULARITY_MODULE"
cd "$ROOT/fcx"
date -Is; hostname; md5sum wiki_convert.py test_wiki_convert.py bench.py ingest.py ingest_v1.py ingest_run.py ingest_ablate.py test_ingest.py test_ablate.py test_ingest_v2.py
for t in ${TESTS:?}; do
  echo "== $t"; singularity exec --bind "$SIF_BIND" "$ROOT/envs/cpu-data-py312-v1.sif" python "$t" < /dev/null
done
echo FCX_TEST_OK; date -Is
