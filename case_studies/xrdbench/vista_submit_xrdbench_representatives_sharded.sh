#!/bin/bash

set -euo pipefail

ROOT=$SCRATCH
CODE_DIR=${CODE_DIR:-$ROOT/ai-diffraction/Code/XRDBench}

TRAIN_SCRIPT="$CODE_DIR/vista_train_xrdbench_model_simxrd.sh"
EVAL_SCRIPT="$CODE_DIR/vista_eval_xrdbench_model_rruff473.sh"

TRAIN_MANIFEST=${TRAIN_MANIFEST:-$ROOT/simxrd_train1_pt_shards/manifest.json}
VAL_MANIFEST=${VAL_MANIFEST:-$ROOT/simxrd_val_pt_shards/manifest.json}
TEST_MANIFEST=${TEST_MANIFEST:-$ROOT/simxrd_test_pt_shards/manifest.json}

MODELS=${MODELS:-"transformer bigru cnn11"}
TASK=${TASK:-spg}
EPOCHS=${EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-128}
NUM_WORKERS=${NUM_WORKERS:-4}
PATIENCE=${PATIENCE:-3}
SEED=${SEED:-100}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-250}

extract_job_id() {
  awk 'NF {line=$0} END {print line}'
}

for MODEL in $MODELS; do
  train_job=$(sbatch --parsable \
    --job-name="xrdbench-${MODEL}-full" \
    --output="$ROOT/xrdbench_${MODEL}_train_full.%j.out" \
    --error="$ROOT/xrdbench_${MODEL}_train_full.%j.err" \
    --export=ALL,MODEL="$MODEL",TASK="$TASK",EPOCHS="$EPOCHS",BATCH_SIZE="$BATCH_SIZE",NUM_WORKERS="$NUM_WORKERS",PATIENCE="$PATIENCE",SEED="$SEED",XRDBENCH_CHECKPOINT_EVERY_STEPS="$CHECKPOINT_EVERY",XRDBENCH_TRAIN_DB_NAMES="train_1",XRDBENCH_TRAIN_SHARD_MANIFEST="$TRAIN_MANIFEST",XRDBENCH_VAL_SHARD_MANIFEST="$VAL_MANIFEST",XRDBENCH_TEST_SHARD_MANIFEST="$TEST_MANIFEST" \
    "$TRAIN_SCRIPT" | extract_job_id)
  echo "submitted train $MODEL -> $train_job"

  eval_job=$(sbatch --parsable \
    --dependency=afterok:${train_job} \
    --job-name="xrdbench-${MODEL}-r473" \
    --output="$ROOT/xrdbench_${MODEL}_r473_eval.%j.out" \
    --error="$ROOT/xrdbench_${MODEL}_r473_eval.%j.err" \
    --export=ALL,MODEL="$MODEL",TASK="$TASK",OUT="$ROOT/xrdbench_${MODEL}_rruff473_eval.json",FAILURE_JSON="$ROOT/xrdbench_${MODEL}_rruff473_failure.json",TOPOLOGY_JSON="$ROOT/xrdbench_${MODEL}_rruff473_topology.json",TOPOLOGY_BOOTSTRAP_JSON="$ROOT/xrdbench_${MODEL}_rruff473_topology_bootstrap.json" \
    "$EVAL_SCRIPT" | extract_job_id)
  echo "submitted eval  $MODEL -> $eval_job"
done
