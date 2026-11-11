#!/bin/bash
# Package copy: submit from ${PROJECT_ROOT} after `mkdir -p logs`, e.g.
#   sbatch --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" [options below] cluster/wiki_ingest.sbatch
# PROJECT_ROOT must be exported (sbatch exports the environment by default); see README.md and .env.example.
# E-WIKI: ingest one converted sample with delta parsing and 64 interleaved traces (fcx/ingest_run.py, the paper's ingestion),
# writing messages.parquet, variants.parquet, wiki/{transitions,checkpoints,arrays}.parquet and convert_stats.json into the
# sample directory, where the benchmark driver reads them. Refuses to run if the directory already holds an ingested store.
# With SLICE=K the job first copies the first K traces of the sample into <sample>_k<K> (sanity runs) and ingests that copy.
# usage: sbatch --array=0-2%3 --job-name=fcx-wikiing --time=HH:MM:SS --export=ALL,SAMPLES="wiki_s1 wiki_s2 wiki_s3"[,SLICE=K] cluster/wiki_ingest.sbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
ROOT=${PROJECT_ROOT:?set PROJECT_ROOT to the package root}
read -r -a S <<< "${SAMPLES:?}"
NAME=${S[$SLURM_ARRAY_TASK_ID]}
[[ "$NAME" =~ ^wiki_s[0-9]+$ ]] || { echo "bad sample $NAME"; exit 2; }
WORK=$(mktemp -d -u --tmpdir "fcx-${SLURM_JOB_ID}.XXXXXX")   # a name under $TMPDIR (node-local disk) or the system default; created below
SIF_BIND=${SIF_BIND:-$ROOT,$WORK}   # paths bound into the containers
mkdir -p "$WORK"
trap 'rm -rf "$WORK"' EXIT
export TMPDIR=$WORK
[ -z "${SINGULARITY_MODULE:-}" ] || module load "$SINGULARITY_MODULE"
cd "$ROOT/fcx"
RUN="singularity exec --bind $SIF_BIND $ROOT/envs/cpu-data-py312-v1.sif python"
date -Is; hostname; cat ingest.py ingest_run.py convert.py | md5sum
D=$ROOT/data/$NAME
if [ -n "${SLICE:-}" ]; then
  [[ "$SLICE" =~ ^[1-9][0-9]*$ ]] || { echo "SLICE must be a positive integer"; exit 2; }
  $RUN slice_sample.py --src "$D" --dst "${D}_k$SLICE" --traces "$SLICE" --manager wiki
  D=${D}_k$SLICE
fi
test ! -e "$D/convert_stats.json" -a ! -e "$D/messages.parquet" || { echo "refusing to overwrite the store in $D"; exit 2; }
$RUN ingest_run.py --data "$D" --managers wiki --work "$WORK" --delta 1 --interleave 64 --seed 1
python3 -c "import json;d=json.load(open('$D/convert_stats.json'));p=d['policies']['wiki'];print('spans',d['spans'],'messages',d['distinct_messages'],'transitions/span',p['transitions_per_span'],'ctx len',p['mean_context_len'],'cpu',p['ingest']['cpu_s'])"
echo WIKI_INGEST_OK "$D"; date -Is
