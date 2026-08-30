#!/bin/bash
#SBATCH -A <ALLOCATION>
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00

set -euo pipefail

ROOT=$SCRATCH
REPO_DIR=${REPO_DIR:-$ROOT/XRDBench}
VENV_DIR=${VENV_DIR:-$ROOT/xrdbench-venv}
DATA_DIR=${DATA_DIR:-$ROOT/simxrd_public_xrdbench}
MODEL=${MODEL:?MODEL must be set, e.g. transformer, bigru, cnn11}
TASK=${TASK:-spg}
CODE_SYNC_DIR=${CODE_SYNC_DIR:-$ROOT/ai-diffraction/Code/XRDBench}

module load gcc/13.2.0 python3/3.11.8 cuda/12.4
source "$VENV_DIR/bin/activate"

export WANDB_MODE=offline
export XRDBENCH_TRAIN_DB_NAMES="${XRDBENCH_TRAIN_DB_NAMES:-train_1,train_2}"
export XRDBENCH_CHECKPOINT_EVERY_STEPS="${XRDBENCH_CHECKPOINT_EVERY_STEPS:-250}"

if [[ -n "${XRDBENCH_TRAIN_SHARD_MANIFEST:-}" ]]; then
  mkdir -p "$REPO_DIR/dataset"
  cp "$CODE_SYNC_DIR/overrides/train.py" "$REPO_DIR/train.py"
  cp "$CODE_SYNC_DIR/overrides/dataset/pt_shard_dataset.py" "$REPO_DIR/dataset/pt_shard_dataset.py"
fi

cd "$REPO_DIR"
python main.py \
  --model "$MODEL" \
  --task "$TASK" \
  --epochs "${EPOCHS:-10}" \
  --batch_size "${BATCH_SIZE:-128}" \
  --num_workers "${NUM_WORKERS:-16}" \
  --patience "${PATIENCE:-3}" \
  --seed "${SEED:-100}" \
  --datapath "$DATA_DIR" \
  --gpu 0
