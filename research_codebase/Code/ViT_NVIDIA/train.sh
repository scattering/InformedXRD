#!/bin/bash

# Set default parameters
VENV_PATH="$SCRATCH/nvidia-scratch/bin/activate"  # Modify this to point to your virtual environment
CONFIG_PATH="./sweep_config_single.json"  # Default single-run config
PROJECT="project"
NUM_SWEEPS=1
RESUME_SWEEP_ID="" # Default empty; set to existing sweep ID to resume


# Multi-node and distributed training parameters
NNODES=${SLURM_JOB_NUM_NODES}  # Detects active node count
NODE_RANK=${SLURM_NODEID}
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)  # Default master address
MASTER_PORT=${MASTER_PORT:-29500}  # Default PyTorch port

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

# --- HARD RESET ---
unset PYTHONPATH
unset CUDA_VISIBLE_DEVICES   # torchrun will set this

# Purge all currently loaded modules
module purge

# Load the modules you need
module load nvidia/24.5
module load python/3.9     # if needed

# Activate Virtual Environment
source $VENV_PATH

# Ensure RESUME_SWEEP_ID is defined (optional safety)
RESUME_SWEEP_ID=${RESUME_SWEEP_ID:-""}

# --- Sweep ID setup: resume if exists ---
if [[ -z "$RESUME_SWEEP_ID" ]]; then
    # Create a new sweep if no ID provided
    if [[ $NODE_RANK -eq 0 ]]; then
        echo "Creating new W&B sweep..."
        SWEEP_ID=$(wandb sweep --project "$PROJECT" "$CONFIG_PATH" 2>&1 | tee /dev/tty | awk '/Creating sweep with ID:/ {print $NF}')
        echo "New SWEEP_ID=$SWEEP_ID"
    fi
else
    SWEEP_ID="$RESUME_SWEEP_ID"
    echo "Resuming existing sweep: $SWEEP_ID"
fi

# Export sweep ID to all processes via environment variable
export SWEEP_ID=${SWEEP_ID}

# Now all ranks have the same sweep ID
echo "Using Sweep ID: $SWEEP_ID"

# Print configuration
echo "Running training with $NPROC_PER_NODE GPUs per node"
echo "Nodes: $NNODES, Rank: $NODE_RANK, Master: $MASTER_ADDR:$MASTER_PORT"

# Select a stable Ethernet interface — try eth0 or eno1 instead of Infiniband (ibp10s0)
export NCCL_SOCKET_IFNAME=ibp92s0
# Disable InfiniBand to avoid crash if it's unstable or unavailable
export NCCL_IB_DISABLE=0

# Optional debugging flags (keep these)
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# (Optional) You can remove NCCL_NET_GDR_LEVEL if not using NVLink+IB
unset NCCL_NET_GDR_LEVEL  # or comment it out

# Diagnostics
echo "=== NCCL using interface: $NCCL_SOCKET_IFNAME ==="
ip addr show "$NCCL_SOCKET_IFNAME"
echo "=== Hostname: $(hostname) ==="
echo "=== SLURM Node List: $SLURM_NODELIST ==="
echo "=== SLURM Job ID: $SLURM_JOB_ID ==="

echo "Checking connectivity to master node..."

# Try pinging the MASTER_ADDR (optional)
ping -c 2 "$MASTER_ADDR"

echo "=== End connectivity check ==="

/scratch/10486/UserB/nvidia-scratch/bin/torchrun \
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
