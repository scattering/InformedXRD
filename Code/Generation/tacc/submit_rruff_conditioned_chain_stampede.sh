#!/bin/bash
# Submit a full Stampede3 chain:
# 1. generate conditioned shards on /scratch
# 2. validate shard completeness
# 3. repack to a train-ready HDF5 on /scratch
# 4. copy the train-ready HDF5 to /work2
#
# Example:
#   bash Code/Generation/tacc/submit_rruff_conditioned_chain_stampede.sh
#
# Optional overrides:
#   RUN_NAME=rruff_conditioned_250k_v1
#   TOTAL_SAMPLES=250000
#   NUM_SHARDS=16
#   ARRAY_RANGE=0-1
#   SHARDS_PER_JOB=8
#   TRAIN_PERCENT=80
#   VAL_PERCENT=10
#   COPY_DEST=/work2/ALLOC3/UserE/rruff_conditioned_250k_v1_trainready.hdf5

set -euo pipefail

ACCOUNT="${ACCOUNT:-CDA24014}"
REPO_DIR="${REPO_DIR:-/scratch/ALLOC3/UserE/project}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/ALLOC3/UserE}"
WORKROOT="${WORKROOT:-$SCRATCH_ROOT/pycrysfml_spr}"
VENV_DIR="${VENV_DIR:-$WORKROOT/venv}"

RUN_NAME="${RUN_NAME:-rruff_conditioned_250k_v1}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-250000}"
NUM_SHARDS="${NUM_SHARDS:-16}"
ARRAY_RANGE="${ARRAY_RANGE:-0-1}"
SHARDS_PER_JOB="${SHARDS_PER_JOB:-8}"

TRAIN_PERCENT="${TRAIN_PERCENT:-80}"
VAL_PERCENT="${VAL_PERCENT:-10}"

OUT_DIR="${OUT_DIR:-$SCRATCH_ROOT/ai_diffraction_generated/$RUN_NAME}"
SCRATCH_TRAINREADY="${SCRATCH_TRAINREADY:-$SCRATCH_ROOT/ai_diffraction_generated/${RUN_NAME}_trainready.hdf5}"
COPY_DEST="${COPY_DEST:-/work2/ALLOC3/UserE/${RUN_NAME}_trainready.hdf5}"

EXT_MAP_CSV="${EXT_MAP_CSV:-$REPO_DIR/Code/Post_Processing/FINAL_SPG_ExtG_CrysS_Table.csv}"
RRUFF_PRIOR_CSV="${RRUFF_PRIOR_CSV:-/work2/ALLOC3/UserE/rruff-benchmark/option1_metadata_qc_full.csv}"
ICSD_CHEMISTRY_CSV="${ICSD_CHEMISTRY_CSV:-/work2/ALLOC3/UserE/reference_data/ICSD_With_Bravais.csv}"
PRIOR_MODE="${PRIOR_MODE:-robust}"
EG_SAMPLING_MODE="${EG_SAMPLING_MODE:-rruff}"
NUM_PROCESSES_OVERRIDE="${NUM_PROCESSES_OVERRIDE:-1}"
RESUME_GENERATION="${RESUME_GENERATION:-1}"
ENABLE_PREFERRED_ORIENTATION="${ENABLE_PREFERRED_ORIENTATION:-0}"

echo "[INFO] submitting chain for $RUN_NAME"
echo "[INFO] total_samples=$TOTAL_SAMPLES num_shards=$NUM_SHARDS array_range=$ARRAY_RANGE"
echo "[INFO] out_dir=$OUT_DIR"
echo "[INFO] scratch_trainready=$SCRATCH_TRAINREADY"
echo "[INFO] copy_dest=$COPY_DEST"

submit_stampede() {
  local cmd="$1"
  local host
  host="$(hostname 2>/dev/null || true)"
  if [[ "$host" == *stampede3* ]]; then
    bash -lc "$cmd" 2>/dev/null
  else
    ssh stampede3 "$cmd" 2>/dev/null
  fi
}

extract_job_id() {
  local text="$1"
  printf '%s\n' "$text" | grep -E '^[0-9]+([._][A-Za-z0-9-]+)?$|^[0-9]+(;[A-Za-z0-9._-]+)?$' | tail -n1 | cut -d';' -f1
}

gen_job_raw=$(submit_stampede "sbatch --parsable \
  -J gen-${RUN_NAME} \
  -A ${ACCOUNT} \
  -p spr \
  --array=${ARRAY_RANGE} \
  --export=ALL,REPO_DIR=${REPO_DIR},WORKROOT=${WORKROOT},VENV_DIR=${VENV_DIR},OUT_DIR=${OUT_DIR},TOTAL_SAMPLES=${TOTAL_SAMPLES},NUM_SHARDS=${NUM_SHARDS},RUN_TAG=${RUN_NAME},BATCH_ID=${RUN_NAME},EG_SAMPLING_MODE=${EG_SAMPLING_MODE},PRIOR_MODE=${PRIOR_MODE},RRUFF_PRIOR_CSV=${RRUFF_PRIOR_CSV},ICSD_CHEMISTRY_CSV=${ICSD_CHEMISTRY_CSV},EXT_MAP_CSV=${EXT_MAP_CSV},SHARDS_PER_JOB=${SHARDS_PER_JOB},NUM_PROCESSES_OVERRIDE=${NUM_PROCESSES_OVERRIDE},RESUME_GENERATION=${RESUME_GENERATION},ENABLE_PREFERRED_ORIENTATION=${ENABLE_PREFERRED_ORIENTATION} \
  ${REPO_DIR}/Code/Generation/tacc/run_gen_rruff_conditioned_array_spr.sh")
gen_job="$(extract_job_id "$gen_job_raw")"
test -n "$gen_job"

validate_job_raw=$(submit_stampede "sbatch --parsable \
  --dependency=afterok:${gen_job} \
  -J val-${RUN_NAME} \
  -o ${SCRATCH_ROOT}/val-${RUN_NAME}.%j.out \
  -e ${SCRATCH_ROOT}/val-${RUN_NAME}.%j.err \
  --export=ALL,OUT_DIR=${OUT_DIR},NUM_SHARDS=${NUM_SHARDS},LABEL_KEY=extinction_group,LABEL_MIN=1,LABEL_MAX=99,VENV_PYTHON=${VENV_DIR}/bin/python \
  ${REPO_DIR}/Code/Generation/tacc/validate_generated_shards_generic.sh")
validate_job="$(extract_job_id "$validate_job_raw")"
test -n "$validate_job"

repack_job_raw=$(submit_stampede "sbatch --parsable \
  --dependency=afterok:${validate_job} \
  -J repack-${RUN_NAME} \
  -o ${SCRATCH_ROOT}/repack-${RUN_NAME}.%j.out \
  -e ${SCRATCH_ROOT}/repack-${RUN_NAME}.%j.err \
  --export=ALL,VENV_DIR=${VENV_DIR},REPACK_SCRIPT=${REPO_DIR}/Code/Generation/tacc/repack_shards_trainready_with_label.py,INPUT_DIR=${OUT_DIR},OUTPUT_H5=${SCRATCH_TRAINREADY},LABEL_KEY=extinction_group,SPLIT_MODE=order,TRAIN_PERCENT=${TRAIN_PERCENT},VAL_PERCENT=${VAL_PERCENT} \
  ${REPO_DIR}/Code/Generation/tacc/repack_shards_trainready_generic.sh")
repack_job="$(extract_job_id "$repack_job_raw")"
test -n "$repack_job"

copy_job_raw=$(submit_stampede "sbatch --parsable \
  --dependency=afterok:${repack_job} \
  -J copy-${RUN_NAME} \
  -o ${SCRATCH_ROOT}/copy-${RUN_NAME}.%j.out \
  -e ${SCRATCH_ROOT}/copy-${RUN_NAME}.%j.err \
  --export=ALL,SRC_PATH=${SCRATCH_TRAINREADY},DST_PATH=${COPY_DEST} \
  ${REPO_DIR}/Code/Generation/tacc/copy_file_to_work_generic.sh")
copy_job="$(extract_job_id "$copy_job_raw")"
test -n "$copy_job"

echo "generation_job=${gen_job}"
echo "validation_job=${validate_job}"
echo "repack_job=${repack_job}"
echo "copy_job=${copy_job}"
