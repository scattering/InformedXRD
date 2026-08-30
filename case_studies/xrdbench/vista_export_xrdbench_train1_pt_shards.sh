#!/bin/bash
#SBATCH -A <ALLOCATION>
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 08:00:00

set -euo pipefail

ROOT=$SCRATCH
VENV_DIR=${VENV_DIR:-$ROOT/xrdbench-venv}
CODE_DIR=${CODE_DIR:-$ROOT/ai-diffraction/Code/XRDBench}
DB_PATH=${DB_PATH:-$ROOT/simxrd_public_xrdbench/train_1/binxrd.db}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/simxrd_train1_pt_shards}
SHARD_SIZE=${SHARD_SIZE:-10000}
LIMIT=${LIMIT:-0}

module load gcc/13.2.0 python3/3.11.8
source "$VENV_DIR/bin/activate"

mkdir -p "$OUTPUT_DIR"
cd "$CODE_DIR"
python export_ase_db_to_pt_shards.py \
  --db-path "$DB_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --shard-size "$SHARD_SIZE" \
  --limit "$LIMIT"
