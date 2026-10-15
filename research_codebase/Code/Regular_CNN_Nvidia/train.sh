#!/bin/bash

# Set default parameters
CONFIG_PATH="sweep_config_cnn.json"  # Default single-run config
DISTRIBUTED=true  # Enable distributed training

# Multi-node and distributed training parameters
NNODES=${NNODES:-1}  # Default to 1 node, override if needed
NODE_RANK=${NODE_RANK:-0}  # Default to rank 0, override if needed
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}  # Default master address
MASTER_PORT=${MASTER_PORT:-29500}  # Default PyTorch port
NPROC_PER_NODE=4  # !Fixed to use 8 GPUs per node,

# Activate Virtual Environment
source /storage/conda/mambaforge/etc/profile.d/conda.sh
conda activate training
#export OCL_ICD_VENDORS=/etc/OpenCL/vendors
#export CCL_ROOT=${CONDA_PREFIX}



SWEEP_ID=$(wandb sweep --project project-cnn sweep_config_cnn.json 2>&1 | tee /dev/tty | awk '/Creating sweep with ID:/ {print $NF}')
export SWEEP_ID
echo "Initialized Sweep ID: $SWEEP_ID"

# Print configuration
echo "Running training with $NPROC_PER_NODE GPUs per node"
echo "Nodes: $NNODES, Rank: $NODE_RANK, Master: $MASTER_ADDR:$MASTER_PORT"
echo "Distributed Training: $DISTRIBUTED"
echo "Using Sweep: $SWEEP"

# Construct base command
CMD="torchrun \
      --nnodes=$NNODES \
      --node_rank=$NODE_RANK \
      --nproc_per_node=$NPROC_PER_NODE \
      --master_addr=$MASTER_ADDR \
      --master_port=$MASTER_PORT \
      train.py \
      --config $CONFIG_PATH \
      --sweep_id $SWEEP_ID"

# Add flags based on the run type
if [ "$SWEEP" = true ]; then
    CMD="$CMD --sweep"
fi

if [ "$DISTRIBUTED" = true ]; then
    CMD="$CMD --distributed"
fi

# Run the command
echo "Executing: $CMD"
eval $CMD
