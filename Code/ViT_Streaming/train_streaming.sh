#!/bin/bash
# =============================================================================
# train_streaming.sh — SLURM launcher for train_streaming.py
#
# Normal production run:
#   sbatch train_streaming.sh
#   DATASET_SIZE=1.5m sbatch train_streaming.sh
#
# Debug run (no GPU, no DDP, tiny dataset, runs in idev):
#   DEBUG=1 bash train_streaming.sh
#   DEBUG=1 DATASET_SIZE=500 VAL_SIZE=100 bash train_streaming.sh
#
# Resume:
#   RESUME_SWEEP_ID=abc123 \
#     RESUME_CHECKPOINT=$SCRATCH/checkpoints/ckpt_epoch_004.pt \
#     sbatch train_streaming.sh
#
# DATASET_SIZE/VAL_SIZE accept: 218k  1.5m  20m  500000  2g  etc.
# =============================================================================

#SBATCH --job-name=xrd_streaming
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:4
#SBATCH --mem=512G
#SBATCH --time=48:00:00
#SBATCH --output=logs/streaming_%j.out
#SBATCH --error=logs/streaming_%j.err

set -euo pipefail

# ---- Debug flag — must be first -----------------------------------------
DEBUG=${DEBUG:-0}

# ---- Paths --------------------------------------------------------------
VENV_NAME="my_pycrysfml_env"
VENV_ROOT=${VENV_ROOT:-"$SCRATCH/$VENV_NAME"}
CONFIG_PATH="./sweep_config_streaming.json"
PROJECT="project"
CHECKPOINT_DIR="${SCRATCH:-$HOME}/checkpoints"
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-""}
RESUME_SWEEP_ID=${RESUME_SWEEP_ID:-""}

# ---- Generation dependencies --------------------------------------------
WORKROOT=${WORKROOT:-"/work/ALLOC3/UserE/shared/pycrysfml_spr"}
export FULLPROF=${FULLPROF:-"$WORKROOT/fullprof"}
export FULLPROF_DIR=${FULLPROF_DIR:-"$WORKROOT/fullprof"}
export EXT_MAP_CSV=${EXT_MAP_CSV:-"/scratch/10486/UserB/project/Code/Post_Processing/FINAL_SPG_ExtG_CrysS_Table.csv"}
export RRUFF_PRIOR_CSV=${RRUFF_PRIOR_CSV:-"/work2/ALLOC3/UserE/rruff-benchmark/option1_metadata_qc_full.csv"}
export ICSD_CHEMISTRY_CSV=${ICSD_CHEMISTRY_CSV:-"/work2/ALLOC3/UserE/reference_data/ICSD_With_Bravais.csv"}

# ---- Mode-dependent defaults --------------------------------------------
if [[ "$DEBUG" == "1" ]]; then
    DATASET_SIZE=${DATASET_SIZE:-"500"}
    VAL_SIZE=${VAL_SIZE:-"100"}
    N_WORKERS=${N_WORKERS:-4}
    NUM_SWEEPS=${NUM_SWEEPS:-1}
    NNODES=1
    NPROC_PER_NODE=1
    NODE_RANK=0
    MASTER_ADDR="localhost"
    MASTER_PORT=${MASTER_PORT:-29500}
else
    DATASET_SIZE=${DATASET_SIZE:-"10m"}
    VAL_SIZE=${VAL_SIZE:-"100k"}
    N_WORKERS=${N_WORKERS:-88}
    NUM_SWEEPS=${NUM_SWEEPS:-5}
    NNODES=${SLURM_JOB_NUM_NODES:-1}
    NODE_RANK=${SLURM_NODEID:-0}
    MASTER_PORT=${MASTER_PORT:-29500}
    if [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
        MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    else
        MASTER_ADDR="localhost"
    fi
    if command -v nvidia-smi &> /dev/null; then
        NPROC_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    else
        NPROC_PER_NODE=1
    fi
fi

POOL_CHUNKSIZE=${POOL_CHUNKSIZE:-16}

echo "============================================================"
echo "  XRD Streaming Training  (in-process Pool)"
echo "  mode          : $([ "$DEBUG" == "1" ] && echo DEBUG || echo PRODUCTION)"
echo "  dataset_size  : $DATASET_SIZE   val_size   : $VAL_SIZE"
echo "  n_workers     : $N_WORKERS      chunksize  : $POOL_CHUNKSIZE"
echo "  num_sweeps    : $NUM_SWEEPS"
echo "  checkpoint_dir: $CHECKPOINT_DIR"
echo "  resume_ckpt   : ${RESUME_CHECKPOINT:-'(none)'}"
echo "  nodes         : $NNODES         GPUs/node  : $NPROC_PER_NODE"
echo "  node_rank     : $NODE_RANK      master     : $MASTER_ADDR:$MASTER_PORT"
echo "============================================================"

# ---- Environment --------------------------------------------------------
unset PYTHONPATH
unset CUDA_VISIBLE_DEVICES

# Prevent OpenMP deadlock when torch + pyxtal are both imported
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

module purge 2>/dev/null || true
# Only load GPU modules in production — guard against failure on login nodes
if [[ "$DEBUG" != "1" ]]; then
    module load nvidia/24.5 || echo "[WARN] module load nvidia/24.5 failed — continuing"
fi

# Activate pure Python venv
if [[ -f "$VENV_ROOT/bin/activate" ]]; then
    source "$VENV_ROOT/bin/activate"
else
    echo "ERROR: venv not found at $VENV_ROOT" && exit 1
fi

# Ensure libpython3.12 is findable after module purge
export LD_LIBRARY_PATH="/opt/apps/python/3.12.11/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "=== Python  : $(which python) ==="
echo "=== Venv    : $VIRTUAL_ENV ==="

# ---- NCCL (production only) ---------------------------------------------
if [[ "$DEBUG" != "1" ]]; then
    export NCCL_SOCKET_IFNAME=ibp92s0
    export NCCL_IB_DISABLE=0
    export NCCL_DEBUG=INFO
    export TORCH_DISTRIBUTED_DEBUG=DETAIL
    unset NCCL_NET_GDR_LEVEL
    echo "=== Interface : $NCCL_SOCKET_IFNAME ==="
    ip addr show "$NCCL_SOCKET_IFNAME" 2>/dev/null || echo "(not found)"
    echo "=== Hostname  : $(hostname) ==="
    echo "=== SLURM Job : ${SLURM_JOB_ID:-N/A} ==="
    ping -c 2 "$MASTER_ADDR" || echo "(ping failed)"
fi

mkdir -p "$CHECKPOINT_DIR" logs

# ---- Sweep ID -----------------------------------------------------------
if [[ -z "$RESUME_SWEEP_ID" ]]; then
    if [[ "$NODE_RANK" -eq 0 ]]; then
        echo "Creating new W&B sweep..."
        SWEEP_ID=$(wandb sweep --project "$PROJECT" "$CONFIG_PATH" 2>&1 \
                   | tee /dev/stderr \
                   | awk '/Creating sweep with ID:/ {print $NF}')
        echo "New SWEEP_ID=$SWEEP_ID"
    fi
else
    SWEEP_ID="$RESUME_SWEEP_ID"
    echo "Resuming sweep: $SWEEP_ID"
fi
export SWEEP_ID
echo "Using Sweep ID: $SWEEP_ID"

# ---- Resume arg ---------------------------------------------------------
RESUME_ARG=""
if [[ -n "$RESUME_CHECKPOINT" ]]; then
    [[ ! -f "$RESUME_CHECKPOINT" ]] && echo "ERROR: checkpoint not found: $RESUME_CHECKPOINT" && exit 1
    RESUME_ARG="--resume_checkpoint $RESUME_CHECKPOINT"
fi

# ---- Launch -------------------------------------------------------------
if [[ "$DEBUG" == "1" ]]; then
    echo "=== DEBUG: plain python, no DDP ==="
    python train_streaming.py               \
        --config         "$CONFIG_PATH"     \
        --sweep_id       "$SWEEP_ID"        \
        --num_sweeps     "$NUM_SWEEPS"      \
        --dataset_size   "$DATASET_SIZE"    \
        --val_size       "$VAL_SIZE"        \
        --n_workers      "$N_WORKERS"       \
        --pool_chunksize "$POOL_CHUNKSIZE"  \
        --checkpoint_dir "$CHECKPOINT_DIR"  \
        $RESUME_ARG
else
    torchrun \
        --nnodes="$NNODES"                          \
        --nproc_per_node="$NPROC_PER_NODE"          \
        --node_rank="$NODE_RANK"                    \
        --rdzv_id=456                               \
        --rdzv_backend=c10d                         \
        --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
        train_streaming.py                          \
        --config         "$CONFIG_PATH"             \
        --sweep_id       "$SWEEP_ID"                \
        --num_sweeps     "$NUM_SWEEPS"              \
        --distributed                               \
        --dataset_size   "$DATASET_SIZE"            \
        --val_size       "$VAL_SIZE"                \
        --n_workers      "$N_WORKERS"               \
        --pool_chunksize "$POOL_CHUNKSIZE"          \
        --checkpoint_dir "$CHECKPOINT_DIR"          \
        $RESUME_ARG
fi
