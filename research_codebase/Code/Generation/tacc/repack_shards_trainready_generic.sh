#!/bin/bash
#SBATCH -J repack-shards
#SBATCH -A CDA24014
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 02:00:00

set -euo pipefail

module load python/3.12.11

VENV_DIR="${VENV_DIR:-/scratch/ALLOC3/UserE/pycrysfml_spr/venv}"
REPACK_SCRIPT="${REPACK_SCRIPT:?REPACK_SCRIPT is required}"
INPUT_DIR="${INPUT_DIR:?INPUT_DIR is required}"
OUTPUT_H5="${OUTPUT_H5:?OUTPUT_H5 is required}"
LABEL_KEY="${LABEL_KEY:-extinction_group}"
SPLIT_MODE="${SPLIT_MODE:-order}"
TRAIN_PERCENT="${TRAIN_PERCENT:-80}"
VAL_PERCENT="${VAL_PERCENT:-10}"
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"

source "$VENV_DIR/bin/activate"

python "$REPACK_SCRIPT" \
  --input-dir "$INPUT_DIR" \
  --output-h5 "$OUTPUT_H5" \
  --label-key "$LABEL_KEY" \
  --split-mode "$SPLIT_MODE" \
  --train-percent "$TRAIN_PERCENT" \
  --val-percent "$VAL_PERCENT"
