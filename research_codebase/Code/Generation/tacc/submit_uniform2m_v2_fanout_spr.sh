#!/bin/bash
# Submit the next 2M-sample uniform generation tranche using the
# previously stable 4-job fanout pattern on Stampede3 `spr`.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="$SCRIPT_DIR/run_gen_like_paper_uniform_array_spr.sh"

TOTAL_SAMPLES="${TOTAL_SAMPLES:-2000000}"
NUM_SHARDS="${NUM_SHARDS:-100}"
RUN_TAG="${RUN_TAG:-uniform2m_v2}"
BATCH_ID="${BATCH_ID:-$RUN_TAG}"
OUT_DIR="${OUT_DIR:-$SCRATCH/ai_diffraction_generated/uniform_2m_v2}"
BASE_OUTPUT_HDF5="${BASE_OUTPUT_HDF5:-$OUT_DIR/uniform_eg_impurity_shard}"
SHARDS_PER_JOB="${SHARDS_PER_JOB:-8}"
NUM_PROCESSES_OVERRIDE="${NUM_PROCESSES_OVERRIDE:-1}"
ARRAY_TASK_COUNT_OVERRIDE="${ARRAY_TASK_COUNT_OVERRIDE:-4}"

mkdir -p "$OUT_DIR"

for task_id in 0 1 2 3; do
  env \
    TOTAL_SAMPLES="$TOTAL_SAMPLES" \
    NUM_SHARDS="$NUM_SHARDS" \
    RUN_TAG="$RUN_TAG" \
    BATCH_ID="$BATCH_ID" \
    OUT_DIR="$OUT_DIR" \
    BASE_OUTPUT_HDF5="$BASE_OUTPUT_HDF5" \
    SHARDS_PER_JOB="$SHARDS_PER_JOB" \
    NUM_PROCESSES_OVERRIDE="$NUM_PROCESSES_OVERRIDE" \
    ARRAY_TASK_ID_OVERRIDE="$task_id" \
    ARRAY_TASK_COUNT_OVERRIDE="$ARRAY_TASK_COUNT_OVERRIDE" \
    sbatch "$BASE_SCRIPT"
done
