#!/bin/bash
# Generic Slurm launcher for gen_rruff_conditioned.py.
# Edit the SBATCH lines and module loads for the target cluster.
#
# Example:
#   sbatch --array=0-1 \
#     --export=ALL,REPO_DIR=/path/to/project,WORKROOT=/path/to/workroot,VENV_DIR=/path/to/venv,FULLPROF_DIR=/path/to/fullprof,OUT_DIR=/path/to/output/rruff_conditioned_100k,TOTAL_SAMPLES=100000,NUM_SHARDS=16,RUN_TAG=rruff_cond_100k,EG_SAMPLING_MODE=rruff,PRIOR_MODE=robust,RRUFF_PRIOR_CSV=/path/to/option1_metadata_qc_full.csv,ICSD_CHEMISTRY_CSV=/path/to/ICSD_With_Bravais.csv,EXT_MAP_CSV=/path/to/FINAL_SPG_ExtG_CrysS_Table.csv,SHARDS_PER_JOB=8,NUM_PROCESSES_OVERRIDE=1,RESUME_GENERATION=1 \
#     /path/to/project/Code/Generation/run_gen_rruff_conditioned_array_generic.sh

#SBATCH -J gen-rruff-cond
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 24:00:00

set -euo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
WORKROOT="${WORKROOT:-$HOME/pycrysfml_work}"
VENV_DIR="${VENV_DIR:-$WORKROOT/venv}"
FULLPROF_DIR="${FULLPROF_DIR:-$WORKROOT/fullprof}"
OUT_DIR="${OUT_DIR:-$PWD/rruff_conditioned}"

TOTAL_SAMPLES="${TOTAL_SAMPLES:-100000}"
NUM_SHARDS="${NUM_SHARDS:-16}"
SHARD_ID="${SHARD_ID:-${SLURM_ARRAY_TASK_ID:-0}}"
RUN_TAG="${RUN_TAG:-rruff_cond}"
BATCH_ID="${BATCH_ID:-$RUN_TAG}"
BASE_OUTPUT_HDF5="${BASE_OUTPUT_HDF5:-$OUT_DIR/rruff_conditioned_shard}"
EXT_MAP_CSV="${EXT_MAP_CSV:-$REPO_DIR/Code/Post_Processing/FINAL_SPG_ExtG_CrysS_Table.csv}"
RRUFF_PRIOR_CSV="${RRUFF_PRIOR_CSV:-$REPO_DIR/artifacts/option1_metadata_qc_full.csv}"
ICSD_CHEMISTRY_CSV="${ICSD_CHEMISTRY_CSV:-$REPO_DIR/artifacts/ICSD_With_Bravais.csv}"
PRIOR_MODE="${PRIOR_MODE:-robust}"
EG_SAMPLING_MODE="${EG_SAMPLING_MODE:-rruff}"
NUM_PROCESSES_OVERRIDE="${NUM_PROCESSES_OVERRIDE:-1}"
SHARDS_PER_JOB="${SHARDS_PER_JOB:-8}"
ARRAY_TASK_ID="${ARRAY_TASK_ID_OVERRIDE:-${SLURM_ARRAY_TASK_ID:-0}}"
ARRAY_TASK_COUNT="${ARRAY_TASK_COUNT_OVERRIDE:-${SLURM_ARRAY_TASK_COUNT:-1}}"
LANE_MAX_RETRIES="${LANE_MAX_RETRIES:-3}"
LANE_RETRY_SLEEP="${LANE_RETRY_SLEEP:-15}"

if [ -f /etc/profile.d/modules.sh ]; then
  # Replace these loads with the target cluster's compiler/python stack if needed.
  source /etc/profile.d/modules.sh || true
  module purge || true
  module load gcc || true
  module load python || true
fi

source "$VENV_DIR/bin/activate"
mkdir -p "$OUT_DIR"

export FULLPROF="$FULLPROF_DIR"
export EG_SAMPLING_MODE
export PRIOR_MODE
export TOTAL_SAMPLES
export GENMAP_NUM_SHARDS="$NUM_SHARDS"
export GENMAP_SHARD_ID="$SHARD_ID"
export BATCH_ID
export BASE_OUTPUT_HDF5
export EXT_MAP_CSV
export RRUFF_PRIOR_CSV
export ICSD_CHEMISTRY_CSV
export RESUME_GENERATION="${RESUME_GENERATION:-1}"

cd "$REPO_DIR/Code/Generation"

run_lane() {
  local lane="$1"
  local shard_id=$((ARRAY_TASK_ID * SHARDS_PER_JOB + lane))
  local shard_stride=$((ARRAY_TASK_COUNT * SHARDS_PER_JOB))
  local attempts

  while [ "$shard_id" -lt "$NUM_SHARDS" ]; do
    attempts=0
    while true; do
      echo "[INFO] lane=$lane shard_id=$shard_id stride=$shard_stride attempt=$((attempts + 1)) run_tag=$RUN_TAG eg_mode=$EG_SAMPLING_MODE prior_mode=$PRIOR_MODE"
      if GENMAP_SHARD_ID="$shard_id" \
        NUM_PROCESSES_OVERRIDE="$NUM_PROCESSES_OVERRIDE" \
        python gen_rruff_conditioned.py; then
        echo "[INFO] lane=$lane shard_id=$shard_id completed"
        break
      fi

      attempts=$((attempts + 1))
      echo "[WARN] lane=$lane shard_id=$shard_id failed attempt=$attempts"
      if [ "$attempts" -ge "$LANE_MAX_RETRIES" ]; then
        echo "[WARN] lane=$lane shard_id=$shard_id exceeded retry budget; leaving shard for later resume and continuing"
        break
      fi
      sleep "$LANE_RETRY_SLEEP"
    done
    shard_id=$((shard_id + shard_stride))
  done
}

pids=()
for lane in $(seq 0 $((SHARDS_PER_JOB - 1))); do
  initial_shard=$((ARRAY_TASK_ID * SHARDS_PER_JOB + lane))
  if [ "$initial_shard" -ge "$NUM_SHARDS" ]; then
    continue
  fi
  run_lane "$lane" &
  pids+=("$!")
done

rc=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    rc=1
  fi
done

exit "$rc"
