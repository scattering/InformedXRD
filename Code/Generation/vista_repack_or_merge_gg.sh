#!/bin/bash
#SBATCH -J vista-repack
#SBATCH -A CDA24014
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 02:00:00
#SBATCH -o /scratch/ALLOC3/UserE/vista_repack_merge_%j.out
#SBATCH -e /scratch/ALLOC3/UserE/vista_repack_merge_%j.err

set -euo pipefail

module purge
module load gcc/13.2.0
module load python3/3.11.8

REPO_DIR="${REPO_DIR:-/scratch/ALLOC3/UserE/project}"
VENV_DIR="${VENV_DIR:-/scratch/ALLOC3/UserE/repack_vista_gg_venv}"
MODE="${MODE:?MODE must be repack or merge}"

source "$VENV_DIR/bin/activate"
cd "$REPO_DIR/Code/Generation/tacc"

case "$MODE" in
  repack)
    INPUT_DIR="${INPUT_DIR:?INPUT_DIR is required for repack}"
    OUTPUT_H5="${OUTPUT_H5:?OUTPUT_H5 is required for repack}"
    LABEL_KEY="${LABEL_KEY:-extinction_group}"
    SPLIT_MODE="${SPLIT_MODE:-order}"
    TRAIN_PERCENT="${TRAIN_PERCENT:-80}"
    VAL_PERCENT="${VAL_PERCENT:-10}"
    ALLOW_PARTIAL="${ALLOW_PARTIAL:-0}"
    MIN_WRITTEN="${MIN_WRITTEN:-1}"

    args=(
      --input-dir "$INPUT_DIR"
      --output-h5 "$OUTPUT_H5"
      --label-key "$LABEL_KEY"
      --split-mode "$SPLIT_MODE"
      --train-percent "$TRAIN_PERCENT"
      --val-percent "$VAL_PERCENT"
      --min-written "$MIN_WRITTEN"
    )
    if [ "$ALLOW_PARTIAL" = "1" ]; then
      args+=(--allow-partial)
    fi

    python repack_shards_trainready_with_label.py "${args[@]}"
    ;;

  merge)
    INPUTS="${INPUTS:?INPUTS is required for merge; use colon-separated HDF5 paths}"
    OUTPUT_H5="${OUTPUT_H5:?OUTPUT_H5 is required for merge}"
    CHUNK_ROWS="${CHUNK_ROWS:-32768}"

    IFS=':' read -r -a input_array <<< "$INPUTS"
    python merge_trainready_hdf5.py \
      --inputs "${input_array[@]}" \
      --output "$OUTPUT_H5" \
      --chunk-rows "$CHUNK_ROWS"
    ;;

  *)
    echo "[ERROR] Unsupported MODE=$MODE"
    exit 1
    ;;
esac
