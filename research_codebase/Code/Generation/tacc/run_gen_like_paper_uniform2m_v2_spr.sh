#!/bin/bash
# Launch the next 2M-sample uniform generation batch on Stampede3 `spr`
# with collision-safe defaults.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="$SCRIPT_DIR/run_gen_like_paper_uniform_array_spr.sh"

export TOTAL_SAMPLES="${TOTAL_SAMPLES:-2000000}"
export NUM_SHARDS="${NUM_SHARDS:-100}"
export RUN_TAG="${RUN_TAG:-uniform2m_v2}"
export BATCH_ID="${BATCH_ID:-$RUN_TAG}"
export OUT_DIR="${OUT_DIR:-$SCRATCH/ai_diffraction_generated/uniform_2m_v2}"
export BASE_OUTPUT_HDF5="${BASE_OUTPUT_HDF5:-$OUT_DIR/uniform_eg_impurity_shard}"
export SHARDS_PER_JOB="${SHARDS_PER_JOB:-8}"
export NUM_PROCESSES_OVERRIDE="${NUM_PROCESSES_OVERRIDE:-1}"
export RESUME_GENERATION="${RESUME_GENERATION:-1}"

mkdir -p "$OUT_DIR"

exec sbatch \
  --array="${ARRAY_SPEC:-0-12%4}" \
  "$BASE_SCRIPT"
