#!/bin/bash
#SBATCH -J gen-rruff-cond
#SBATCH -p spr
#SBATCH -A CDA24014
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 24:00:00
#SBATCH -o /scratch/ALLOC3/UserE/gen-rruff-cond.%A_%a.out
#SBATCH -e /scratch/ALLOC3/UserE/gen-rruff-cond.%A_%a.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$SCRATCH/project}"
WORKROOT="${WORKROOT:-$SCRATCH/pycrysfml_spr}"
VENV_DIR="${VENV_DIR:-$WORKROOT/venv}"
FULLPROF_DIR="${FULLPROF_DIR:-$WORKROOT/fullprof}"
OUT_DIR="${OUT_DIR:-$SCRATCH/ai_diffraction_generated/rruff_conditioned}"

TOTAL_SAMPLES="${TOTAL_SAMPLES:-1000000}"
NUM_SHARDS="${NUM_SHARDS:-128}"
SHARD_ID="${SHARD_ID:-${SLURM_ARRAY_TASK_ID:-0}}"
RUN_TAG="${RUN_TAG:-rruff_cond}"
BATCH_ID="${BATCH_ID:-$RUN_TAG}"
BASE_OUTPUT_HDF5="${BASE_OUTPUT_HDF5:-$OUT_DIR/rruff_conditioned_shard}"
EXT_MAP_CSV="${EXT_MAP_CSV:-$REPO_DIR/Code/Post_Processing/FINAL_SPG_ExtG_CrysS_Table.csv}"
RRUFF_PRIOR_CSV="${RRUFF_PRIOR_CSV:-/work2/ALLOC3/UserE/rruff-benchmark/option1_metadata_qc_full.csv}"
PRIOR_MODE="${PRIOR_MODE:-robust}"
EG_SAMPLING_MODE="${EG_SAMPLING_MODE:-rruff}"
ENABLE_PREFERRED_ORIENTATION="${ENABLE_PREFERRED_ORIENTATION:-0}"
NUM_PROCESSES_OVERRIDE="${NUM_PROCESSES_OVERRIDE:-1}"
SHARDS_PER_JOB="${SHARDS_PER_JOB:-8}"
ARRAY_TASK_ID="${ARRAY_TASK_ID_OVERRIDE:-${SLURM_ARRAY_TASK_ID:-0}}"
ARRAY_TASK_COUNT="${ARRAY_TASK_COUNT_OVERRIDE:-${SLURM_ARRAY_TASK_COUNT:-1}}"
LANE_MAX_RETRIES="${LANE_MAX_RETRIES:-3}"
LANE_RETRY_SLEEP="${LANE_RETRY_SLEEP:-15}"

module purge
module load gcc/13.2.0
module load python/3.12.11

source "$VENV_DIR/bin/activate"
mkdir -p "$OUT_DIR"

if ! python - <<'PY'
try:
    import pycrysfml  # noqa: F401
    ok = True
except Exception:
    try:
        import CFML_api  # noqa: F401
        ok = True
    except Exception:
        ok = False
raise SystemExit(0 if ok else 1)
PY
then
  python -m pip install "$WORKROOT/crysfml/cfml-0.0.3-py3-none-linux_x86_64.whl"
fi

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
export RESUME_GENERATION="${RESUME_GENERATION:-1}"
export ENABLE_PREFERRED_ORIENTATION
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"

cd "$REPO_DIR/Code/Generation"

run_lane() {
  local lane="$1"
  local shard_id=$((ARRAY_TASK_ID * SHARDS_PER_JOB + lane))
  local shard_stride=$((ARRAY_TASK_COUNT * SHARDS_PER_JOB))
  local attempts

  while [ "$shard_id" -lt "$NUM_SHARDS" ]; do
    attempts=0
    while true; do
      echo "[INFO] lane=$lane shard_id=$shard_id stride=$shard_stride attempt=$((attempts + 1)) run_tag=$RUN_TAG eg_mode=$EG_SAMPLING_MODE prior_mode=$PRIOR_MODE preferred_orientation=$ENABLE_PREFERRED_ORIENTATION"
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
