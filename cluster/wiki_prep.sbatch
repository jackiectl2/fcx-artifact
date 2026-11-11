#!/bin/bash
# Package copy: submit from ${PROJECT_ROOT} after `mkdir -p logs`, e.g.
#   sbatch --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" [options below] cluster/wiki_prep.sbatch
# PROJECT_ROOT must be exported (sbatch exports the environment by default); see README.md and .env.example.
# E-WIKI conversion: run the converter's unit test, then convert the verified Simple English Wikipedia 20261101 dumps (fetched by
# cluster/wiki_fetch.sbatch) into three 3,000-page snapshot-sequence samples data/wiki_s{1,2,3}. lbzip2 decompresses the
# full-history dump with 6 threads and pipes it into the converter (one Python process), leaving CPU for the parser. The
# converter writes to a staging directory and renames the samples into place only after all checks pass.
# usage: sbatch --job-name=fcx-wikiprep --time=HH:MM:SS cluster/wiki_prep.sbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
ROOT=${PROJECT_ROOT:?set PROJECT_ROOT to the package root}
RAW=$ROOT/data/raw/simplewiki-20261101
STUB=$RAW/simplewiki-20261101-stub-meta-history.xml.gz; STUB_SHA1=c475f263c78db6f1e538ee43443d2ed43cb01cad
HIST=$RAW/simplewiki-20261101-pages-meta-history.xml.bz2; HIST_SHA1=4a6fc322490386d966fb4cd0e4eea09c611e6b4d
WORK=$(mktemp -d -u --tmpdir "fcx-${SLURM_JOB_ID}.XXXXXX")   # a name under $TMPDIR (node-local disk) or the system default; created below
SIF_BIND=${SIF_BIND:-$ROOT,$WORK}   # paths bound into the containers
mkdir -p "$WORK"
trap 'rm -rf "$WORK"' EXIT
export TMPDIR=$WORK
date -Is; hostname
echo "$STUB_SHA1  $STUB" | sha1sum -c --quiet && echo "$HIST_SHA1  $HIST" | sha1sum -c --quiet
[ -z "${SINGULARITY_MODULE:-}" ] || module load "$SINGULARITY_MODULE"
cd "$ROOT/fcx"
md5sum wiki_convert.py test_wiki_convert.py
RUN="singularity exec --bind $SIF_BIND $ROOT/envs/cpu-data-py312-v1.sif python"
$RUN test_wiki_convert.py < /dev/null
date -Is
lbzip2 -dc -n 6 "$HIST" | $RUN wiki_convert.py --stub "$STUB" --history - --out "$ROOT/data" \
    --name wiki --seeds 1,2,3 --pages 3000 --min-revs 10 --max-revs 256
ls -la "$ROOT"/data/wiki_s*; du -sh "$ROOT"/data/wiki_s*
echo WIKI_PREP_OK; date -Is
