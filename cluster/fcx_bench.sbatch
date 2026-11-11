#!/bin/bash
# Package copy: submit from ${PROJECT_ROOT} after `mkdir -p logs`, e.g.
#   sbatch --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" [options below] cluster/fcx_bench.sbatch
# PROJECT_ROOT must be exported (sbatch exports the environment by default); see README.md and .env.example.
# FCX benchmark: start a job-local ClickHouse server on node-local disk, load the ingested log, build the access paths, check and time queries.
# usage: sbatch --job-name=fcx-bench --time=HH:MM:SS [--cpus-per-task=N --mem=M] --export=ALL,DATA=<dir under data/>,OUT=<result name>,EXTRA="<driver flags>" cluster/fcx_bench.sbatch
# Refuses to overwrite runs/$DATA/$OUT.json. DATA_ROOT (default $ROOT/data) is the directory holding <DATA>; it is only read
# (e.g. the old project's data/ for the agent samples, read-only). PYSCRIPT (default fcx/bench.py) selects another driver with the same flags, e.g. a server test.

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
ROOT=${PROJECT_ROOT:?set PROJECT_ROOT to the package root}
DATA_ROOT=${DATA_ROOT:-$ROOT/data}
WORK=$(mktemp -d -u --tmpdir "fcx-${SLURM_JOB_ID}.XXXXXX")   # a name under $TMPDIR (node-local disk) or the system default; created below
SIF_BIND=${SIF_BIND:-$ROOT,$WORK}   # paths bound into the containers
PORT=$((20000 + RANDOM % 20000))
[ -z "${SINGULARITY_MODULE:-}" ] || module load "$SINGULARITY_MODULE"
[[ "$DATA" =~ ^[A-Za-z0-9_]+$ && "${OUT:-bench}" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "bad DATA or OUT"; exit 2; }
test ! -e "$ROOT/runs/$DATA/${OUT:-bench}.json" || { echo "refusing to overwrite runs/$DATA/${OUT:-bench}.json"; exit 2; }
mkdir -p "$WORK/ch" "$WORK/log" "$ROOT/runs/$DATA"
date -Is; hostname; nproc; df -hT "$WORK" | tail -1; lsblk -d -o NAME,ROTA,SIZE,MODEL 2>/dev/null | head -5

# The server listens on loopback only; its password exists only in this job's environment.
# It cannot read the job's cgroup from inside the container, so its memory limit is set to 80% of the allocation here.
export FCX_CH_PW=$(head -c 16 /dev/urandom | od -An -tx1 | tr -d ' \n')
cat > "$WORK/config.xml" <<XML
<clickhouse>
    <logger><level>warning</level><log>$WORK/log/server.log</log><errorlog>$WORK/log/server.err.log</errorlog><size>100M</size><count>2</count></logger>
    <listen_host>127.0.0.1</listen_host>
    <http_port>$PORT</http_port>
    <tcp_port>$((PORT + 1))</tcp_port>
    <path>$WORK/ch/</path>
    <tmp_path>$WORK/ch/tmp_files/</tmp_path>
    <user_files_path>$DATA_ROOT/$DATA/</user_files_path>
    <max_server_memory_usage>$((SLURM_MEM_PER_NODE * 1024 * 820))</max_server_memory_usage>
    <users><default><password from_env="FCX_CH_PW"/><networks><ip>127.0.0.1</ip></networks><profile>default</profile><quota>default</quota><access_management>1</access_management></default></users>
    <profiles><default/></profiles>
    <quotas><default/></quotas>
</clickhouse>
XML
singularity exec --bind "$SIF_BIND" "$ROOT/envs/clickhouse-26.8.15.10.sif" clickhouse server --config-file="$WORK/config.xml" > "$WORK/log/stdout.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null; tail -n 30 "$WORK"/log/server.err.log 2>/dev/null; rm -rf "$WORK"' EXIT

singularity exec --bind "$SIF_BIND" "$ROOT/envs/cpu-data-py312-v1.sif" python "$ROOT/${PYSCRIPT:-fcx/bench.py}" --port "$PORT" --store "$WORK/ch" \
    --data "$DATA_ROOT/$DATA" --out "$ROOT/runs/$DATA/${OUT:-bench}.json" --threads "$SLURM_CPUS_PER_TASK" ${EXTRA:-}
du -sh "$WORK/ch"
echo BENCH_OK; date -Is
