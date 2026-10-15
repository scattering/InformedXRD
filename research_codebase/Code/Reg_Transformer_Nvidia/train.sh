#!/bin/bash

# Set default parameters
VENV_PATH="$SCRATCH/nvidia/bin/activate"  
CONFIG_PATH="./single_config.json"  # Default single-run config
PROJECT="08-17"
NUM_SWEEPS=1

# Multi-node and distributed training parameters
NNODES=${SLURM_JOB_NUM_NODES}  # Detects active node count
NODE_RANK=${SLURM_NODEID}
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)  # Default master address
MASTER_PORT=${MASTER_PORT:-29500}  # Default PyTorch port

# Optional: pass an existing sweep ID as first argument
PREV_SWEEP_ID="$1"

# Detect number of NVIDIA GPUs
if command -v nvidia-smi &> /dev/null; then
    NPROC_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    echo "nvidia-smi not found; defaulting to 1 GPU"
    NPROC_PER_NODE=1
fi

echo "Detected $NPROC_PER_NODE NVIDIA GPU(s)"
echo "Running on $NNODES node(s), node rank $NODE_RANK"
echo "Master address: $MASTER_ADDR:$MASTER_PORT"

# Activate Virtual Environment
source $VENV_PATH

# Load the module environment
ml use /scratch/projects/compilers/modulefiles/

# Only rank 0 runs wandb sweep
if [[ $RANK -eq 0 ]]; then
  if [[ -n "$PREV_SWEEP_ID" ]]; then
    echo "Using existing sweep ID: $PREV_SWEEP_ID"
    SWEEP_ID="$PREV_SWEEP_ID"
  else
    echo "Creating a new sweep from rank 0..."
    SWEEP_ID=$(wandb sweep --project "$PROJECT" "$CONFIG_PATH" 2>&1 | tee /dev/tty | awk '/Creating sweep with ID:/ {print $NF}')
    echo "Created new sweep with ID: $SWEEP_ID"
  fi
fi


# Export sweep ID to all processes via environment variable
export SWEEP_ID=${SWEEP_ID}

# Now all ranks have the same sweep ID
echo "Using Sweep ID: $SWEEP_ID"

# Print configuration
echo "Running training with $NPROC_PER_NODE GPUs per node"
echo "Nodes: $NNODES, Rank: $NODE_RANK, Master: $MASTER_ADDR:$MASTER_PORT"

# Diagnostics
echo "=== NCCL using interface: $NCCL_SOCKET_IFNAME ==="
ip addr show "$NCCL_SOCKET_IFNAME"
echo "=== Hostname: $(hostname) ==="
echo "=== SLURM Node List: $SLURM_NODELIST ==="
echo "=== SLURM Job ID: $SLURM_JOB_ID ==="

echo "=== End connectivity check ==="

torchrun \
  --nnodes=$NNODES \
  --nproc_per_node=$NPROC_PER_NODE \
  --node_rank=$NODE_RANK \
  --rdzv_id=123 \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
  train.py \
  --config "$CONFIG_PATH" \
  --sweep_id "$SWEEP_ID" \
  --num_sweeps $NUM_SWEEPS \
  --sweep \
  --distributed
