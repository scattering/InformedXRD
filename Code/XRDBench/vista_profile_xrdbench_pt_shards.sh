#!/bin/bash
#SBATCH -A CDA24014
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -J xrdbench-pt-prof

set -euo pipefail

ROOT=/scratch/ALLOC3/UserE
export REPO_DIR=${REPO_DIR:-$ROOT/XRDBench}
export VENV_DIR=${VENV_DIR:-$ROOT/xrdbench-venv}
export DATA_DIR=${DATA_DIR:-$ROOT/simxrd_public_xrdbench}
export CODE_SYNC_DIR=${CODE_SYNC_DIR:-$ROOT/project/Code/XRDBench}
export MODEL=${MODEL:-transformer}
export TASK=${TASK:-spg}
export EPOCHS=${EPOCHS:-1}
export BATCH_SIZE=${BATCH_SIZE:-128}
export NUM_WORKERS=${NUM_WORKERS:-4}
export PATIENCE=${PATIENCE:-1}
export SEED=${SEED:-100}
export XRDBENCH_TRAIN_SHARD_MANIFEST=${XRDBENCH_TRAIN_SHARD_MANIFEST:-$ROOT/simxrd_train1_pt_shards/manifest.json}
export XRDBENCH_VAL_SHARD_MANIFEST=${XRDBENCH_VAL_SHARD_MANIFEST:-$ROOT/simxrd_val_pt_shards/manifest.json}
export XRDBENCH_TEST_SHARD_MANIFEST=${XRDBENCH_TEST_SHARD_MANIFEST:-$ROOT/simxrd_test_pt_shards/manifest.json}
export XRDBENCH_MAX_TRAIN_STEPS=${XRDBENCH_MAX_TRAIN_STEPS:-200}
export XRDBENCH_CHECKPOINT_EVERY_STEPS=${XRDBENCH_CHECKPOINT_EVERY_STEPS:-0}
export XRDBENCH_TRAIN_DB_NAMES=${XRDBENCH_TRAIN_DB_NAMES:-train_1}

"$CODE_SYNC_DIR/vista_train_xrdbench_model_simxrd.sh"
