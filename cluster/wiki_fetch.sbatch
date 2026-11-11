#!/bin/bash
# Package copy: submit from ${PROJECT_ROOT} after `mkdir -p logs`, e.g.
#   sbatch --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" [options below] cluster/wiki_fetch.sbatch
# PROJECT_ROOT must be exported (sbatch exports the environment by default); see README.md and .env.example.
# E-WIKI download: the Simple English Wikipedia 20261101 stub and full-history dumps, checked against the SHA-1 values of the
# official dumpstatus.json. Each file is downloaded to <name>.part (resumed if present); if the checksum fails, the .part file
# is deleted and downloaded once more from scratch; only a verified file is renamed to its final name. Nothing else runs here.
# usage: sbatch --job-name=fcx-wikifetch --time=HH:MM:SS cluster/wiki_fetch.sbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
ROOT=${PROJECT_ROOT:?set PROJECT_ROOT to the package root}
RAW=$ROOT/data/raw/simplewiki-20261101
BASE=https://dumps.wikimedia.org/simplewiki/20261101
mkdir -p "$RAW"
date -Is; hostname
fetch() {  # name sha1
  local f=$RAW/$1 sum=$2
  if [ -e "$f" ] && echo "$sum  $f" | sha1sum -c --quiet; then echo "$1 present and verified"; rm -f "$f.part"; return; fi
  test ! -e "$f" || { echo "$f exists but does not verify; not touching it"; exit 3; }
  wget -q -c -O "$f.part" "$BASE/$1" || true
  if ! echo "$sum  $f.part" | sha1sum -c --quiet; then
    echo "$1: checksum failed after the first download; downloading again from scratch"
    rm -f "$f.part"
    wget -q -O "$f.part" "$BASE/$1"
    echo "$sum  $f.part" | sha1sum -c --quiet
  fi
  mv "$f.part" "$f"
  echo "$1 downloaded and verified"; date -Is
}
fetch simplewiki-20261101-stub-meta-history.xml.gz c475f263c78db6f1e538ee43443d2ed43cb01cad
fetch simplewiki-20261101-pages-meta-history.xml.bz2 4a6fc322490386d966fb4cd0e4eea09c611e6b4d
ls -la "$RAW"; which lbzip2
echo WIKI_FETCH_OK; date -Is
