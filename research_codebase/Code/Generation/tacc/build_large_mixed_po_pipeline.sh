#!/bin/bash
# Usage:
#   sbatch Code/Generation/tacc/build_large_mixed_po_pipeline.sh
#
# This script submits a chained Stampede pipeline:
# 1. generate 300k additional PO-conditioned shards on spr
# 2. validate shards on skx-dev
# 3. repack shards to trainready on skx-dev
# 4. merge new 300k PO trainready with existing 200k PO trainready
# 5. build final 2.0M standard + 500k PO mixed trainready

set -euo pipefail

SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/ALLOC3/UserE}"
REPO_DIR="${REPO_DIR:-$SCRATCH_ROOT/project}"
GEN_SCRIPT="${GEN_SCRIPT:-$REPO_DIR/Code/Generation/tacc/run_gen_rruff_conditioned_array_spr.sh}"
VALIDATE_SCRIPT="${VALIDATE_SCRIPT:-$REPO_DIR/Code/Generation/tacc/validate_generated_shards_generic.sh}"
REPACK_SCRIPT_WRAPPER="${REPACK_SCRIPT_WRAPPER:-$REPO_DIR/Code/Generation/tacc/repack_shards_trainready_generic.sh}"
REPACK_SCRIPT="${REPACK_SCRIPT:-$REPO_DIR/Code/Generation/tacc/repack_shards_trainready_with_label.py}"
MERGE_SCRIPT="${MERGE_SCRIPT:-$REPO_DIR/Code/Generation/tacc/merge_trainready_hdf5.py}"
MIX_SCRIPT="${MIX_SCRIPT:-$REPO_DIR/Code/Generation/tacc/build_mixed_trainready_blocks.py}"
VENV_DIR="${VENV_DIR:-$SCRATCH_ROOT/pycrysfml_spr/venv}"

PO300_TAG="${PO300_TAG:-rruff_conditioned_300k_po_v1}"
PO300_DIR="${PO300_DIR:-$SCRATCH_ROOT/ai_diffraction_generated/$PO300_TAG}"
PO300_TRAINREADY="${PO300_TRAINREADY:-$SCRATCH_ROOT/ai_diffraction_generated/${PO300_TAG}_trainready.hdf5}"

PO200_TRAINREADY="${PO200_TRAINREADY:-$SCRATCH_ROOT/ai_diffraction_generated/rruff_conditioned_200k_po_v1_trainready.hdf5}"
PO500_TRAINREADY="${PO500_TRAINREADY:-$SCRATCH_ROOT/ai_diffraction_generated/rruff_conditioned_500k_po_v1_trainready.hdf5}"

STANDARD_TRAINREADY="${STANDARD_TRAINREADY:-$SCRATCH_ROOT/ai_diffraction_generated/rruff_conditioned_2346k_v1_trainready.hdf5}"
FINAL_MIX_TRAINREADY="${FINAL_MIX_TRAINREADY:-$SCRATCH_ROOT/ai_diffraction_generated/rruff_conditioned_mixed_2500k_v1_trainready.hdf5}"

TOTAL_SAMPLES="${TOTAL_SAMPLES:-300000}"
NUM_SHARDS="${NUM_SHARDS:-64}"
SHARDS_PER_JOB="${SHARDS_PER_JOB:-8}"
ARRAY_MAX=$((NUM_SHARDS / SHARDS_PER_JOB - 1))
if [ $((NUM_SHARDS % SHARDS_PER_JOB)) -ne 0 ]; then
  echo "NUM_SHARDS must be divisible by SHARDS_PER_JOB" >&2
  exit 1
fi

mkdir -p "$SCRATCH_ROOT/ai_diffraction_generated"

gen_job_id=$(sbatch --parsable --array=0-"$ARRAY_MAX" \
  --job-name "gen-${PO300_TAG}" \
  --export=ALL,REPO_DIR="$REPO_DIR",OUT_DIR="$PO300_DIR",TOTAL_SAMPLES="$TOTAL_SAMPLES",NUM_SHARDS="$NUM_SHARDS",RUN_TAG="$PO300_TAG",BATCH_ID="$PO300_TAG",EG_SAMPLING_MODE=rruff,PRIOR_MODE=robust,ENABLE_PREFERRED_ORIENTATION=1,HDF5_USE_FILE_LOCKING=FALSE,RESUME_GENERATION=0,SHARDS_PER_JOB="$SHARDS_PER_JOB",NUM_PROCESSES_OVERRIDE=1 \
  "$GEN_SCRIPT")

validate_job_id=$(sbatch --parsable --dependency=afterok:"$gen_job_id" \
  --job-name "validate-${PO300_TAG}" \
  --export=ALL,OUT_DIR="$PO300_DIR",NUM_SHARDS="$NUM_SHARDS",LABEL_KEY=extinction_group,LABEL_MIN=1,LABEL_MAX=99,HDF5_USE_FILE_LOCKING=FALSE \
  "$VALIDATE_SCRIPT")

repack_job_id=$(sbatch --parsable --dependency=afterok:"$validate_job_id" \
  --job-name "repack-${PO300_TAG}" \
  --export=ALL,VENV_DIR="$VENV_DIR",REPACK_SCRIPT="$REPACK_SCRIPT",INPUT_DIR="$PO300_DIR",OUTPUT_H5="$PO300_TRAINREADY",LABEL_KEY=extinction_group,SPLIT_MODE=order,TRAIN_PERCENT=80,VAL_PERCENT=10,HDF5_USE_FILE_LOCKING=FALSE \
  "$REPACK_SCRIPT_WRAPPER")

merge_job_id=$(sbatch --parsable --dependency=afterok:"$repack_job_id" \
  -J "merge-po500k" -A CDA24014 -p skx-dev -N 1 -n 1 -c 8 -t 02:00:00 \
  --wrap="module load python/3.12.11 && source '$VENV_DIR/bin/activate' && export HDF5_USE_FILE_LOCKING=FALSE && python '$MERGE_SCRIPT' --inputs '$PO200_TRAINREADY' '$PO300_TRAINREADY' --output '$PO500_TRAINREADY' --chunk-rows 1024")

mix_job_id=$(sbatch --parsable --dependency=afterok:"$merge_job_id" \
  -J "mix-2500k" -A CDA24014 -p skx-dev -N 1 -n 1 -c 8 -t 04:00:00 \
  --wrap="module load python/3.12.11 && source '$VENV_DIR/bin/activate' && export HDF5_USE_FILE_LOCKING=FALSE && python '$MIX_SCRIPT' --standard '$STANDARD_TRAINREADY' --po '$PO500_TRAINREADY' --output '$FINAL_MIX_TRAINREADY' --std-train 1600000 --po-train 400000 --std-val 200000 --po-val 50000 --std-test 200000 --po-test 50000 --chunk-rows 256 --seed 1337")

echo "gen_job_id=$gen_job_id"
echo "validate_job_id=$validate_job_id"
echo "repack_job_id=$repack_job_id"
echo "merge_job_id=$merge_job_id"
echo "mix_job_id=$mix_job_id"
