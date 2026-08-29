#!/bin/bash
#SBATCH -J repack-u99
#SBATCH -A PHY26002
#SBATCH -p spr
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 04:00:00
#SBATCH -o /scratch/ALLOC3/UserE/repack_uniform_%j.out
#SBATCH -e /scratch/ALLOC3/UserE/repack_uniform_%j.err

set -euo pipefail

module load python/3.12.11
source /scratch/ALLOC3/UserE/pycrysfml_spr/venv/bin/activate

cd /scratch/ALLOC3/UserE/project/Code/Generation/tacc

INPUT_DIR="${INPUT_DIR:-/scratch/ALLOC3/UserE/ai_diffraction_generated/uniform_10m}"
OUTPUT_H5="${OUTPUT_H5:-/scratch/ALLOC3/UserE/ai_diffraction_generated/uniform2m_v1_trainready.hdf5}"
PATTERN="${PATTERN:-*.hdf5}"
TRAIN_PERCENT="${TRAIN_PERCENT:-90}"
VAL_PERCENT="${VAL_PERCENT:-5}"
SPLIT_MODULUS="${SPLIT_MODULUS:-100}"
MAX_SHARDS_ARG=()
if [[ -n "${MAX_SHARDS:-}" ]]; then
  MAX_SHARDS_ARG=(--max-shards "$MAX_SHARDS")
fi

python repack_uniform_shards_trainready.py \
  --input-dir "$INPUT_DIR" \
  --output-h5 "$OUTPUT_H5" \
  --pattern "$PATTERN" \
  --train-percent "$TRAIN_PERCENT" \
  --val-percent "$VAL_PERCENT" \
  --split-modulus "$SPLIT_MODULUS" \
  "${MAX_SHARDS_ARG[@]}" \
  "$@"
