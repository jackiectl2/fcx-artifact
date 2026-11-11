#!/bin/bash
# Package copy: submit from ${PROJECT_ROOT} after `mkdir -p logs`, e.g.
#   sbatch --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" [options below] cluster/chat_fetch.sbatch
# PROJECT_ROOT must be exported (sbatch exports the environment by default); see README.md and .env.example.
# E-CHAT download: the 14 Parquet files of allenai/WildChat-1M (ODC-BY 1.0, not gated) at the pinned dataset commit, checked
# against the SHA-256 values the Hugging Face tree API lists for that commit. Each file is downloaded to <name>.part (resumed if
# present); a file that fails its checksum is downloaded once more from scratch; only a verified file gets its final name.
# Also saves the dataset card and license of that commit. No conversion or measurement runs here.
# usage: sbatch --job-name=fcx-chatfetch --time=HH:MM:SS cluster/chat_fetch.sbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
ROOT=${PROJECT_ROOT:?set PROJECT_ROOT to the package root}
REV=7d6490e462285cf85d91eabea0f9a954fbddcd1f
RAW=$ROOT/data/raw/wildchat-1m-$REV
BASE=https://huggingface.co/datasets/allenai/WildChat-1M/resolve/$REV
mkdir -p "$RAW"
date -Is; hostname
wget -q -O "$RAW/README.md" "$BASE/README.md"; wget -q -O "$RAW/LICENSE.md" "$BASE/LICENSE.md"
fetch() {  # name sha256
  local f=$RAW/$1 sum=$2
  if [ -e "$f" ] && echo "$sum  $f" | sha256sum -c --quiet; then echo "$1 present and verified"; rm -f "$f.part"; return; fi
  test ! -e "$f" || { echo "$f exists but does not verify; not touching it"; exit 3; }
  wget -q -c -O "$f.part" "$BASE/data/$1" || true
  if ! echo "$sum  $f.part" | sha256sum -c --quiet; then
    echo "$1: checksum failed after the first download; downloading again from scratch"
    rm -f "$f.part"
    wget -q -O "$f.part" "$BASE/data/$1"
    echo "$sum  $f.part" | sha256sum -c --quiet
  fi
  mv "$f.part" "$f"
  echo "$1 downloaded and verified"; date -Is
}
fetch train-00000-of-00014.parquet abec2a13129db8c0e6a2d3a51ff12644873c748205a6fdf6551fbcb34430e51c
fetch train-00001-of-00014.parquet f10fa35bb52703baad5e3177d3382d63b855fce9e9d1b9ab1fe2f0e762464f2a
fetch train-00002-of-00014.parquet 9a4912ec14c1012970f57b6710518dda17dbf6cdfb9b6e694880f5456b0b031f
fetch train-00003-of-00014.parquet 3f2d50ee227e32848f6c4fdb0f8ea543b379b2219bec72be9446e232d3d2a670
fetch train-00004-of-00014.parquet 08c7e0c31edc120606d67cfa82486471d82945a7a414761501ea44b1ab0cd804
fetch train-00005-of-00014.parquet 1caab40d8d4f01d76d8e1fa03e80f31b2f4d8b8b4c47cb3aa0280f4467e04555
fetch train-00006-of-00014.parquet 0bd2fff31c74feb6b44b0397b4e7ab6285dd69ccbe4cc8b5f19c42cdbf691303
fetch train-00007-of-00014.parquet f498ea6771be5bf00d259ec76bd9fba51ceaa99c76147a0bb95fd7fa1769b984
fetch train-00008-of-00014.parquet a69578183dde5de1aa94134da2d7c2d13f00d88a577ef91140a492fbbbf10792
fetch train-00009-of-00014.parquet d2eb8868323274c60c9fd5437eaf2159a08ac278598b1197bc20ac04c633648a
fetch train-00010-of-00014.parquet 328b0d702d91ffd47e51aff2de56d74d0f2b874a31f2530f1948474a582f0467
fetch train-00011-of-00014.parquet a458b915aeab7a4a81bd18770f33c3ebb908c91e67d916db5525819c88dbe778
fetch train-00012-of-00014.parquet 5353f1a829ac86de165c2fb537a30d7d756259dbe37a8a0fa6ec791321a37d6e
fetch train-00013-of-00014.parquet 440be579bf012a6e98d179b8dd0bedd73634e5527f7000c067d19c88b5cefa6c
ls -la "$RAW"
echo CHAT_FETCH_OK; date -Is
