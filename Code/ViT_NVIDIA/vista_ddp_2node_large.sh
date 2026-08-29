#!/bin/bash
#SBATCH -J vista-ddp-large
#SBATCH -A CDA24014
#SBATCH -p gh
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -t 12:00:00
#SBATCH -o /scratch/ALLOC3/UserE/vista_ddp_large_%j.out
#SBATCH -e /scratch/ALLOC3/UserE/vista_ddp_large_%j.err

set -euo pipefail

module load gcc/13.2.0 cuda/12.5 python3/3.11.8
source /scratch/ALLOC3/UserE/project-venv/bin/activate
cd /scratch/ALLOC3/UserE/project/Code/ViT_NVIDIA

export WANDB_API_KEY
WANDB_API_KEY="$(cat /scratch/ALLOC3/UserE/project/.wandb_api_key)"
export MASTER_ADDR
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT=29531
export OMP_NUM_THREADS=4

srun bash -lc '
  echo "HOST=$(hostname) RANK=$SLURM_PROCID LOCALID=$SLURM_LOCALID MASTER=$MASTER_ADDR:$MASTER_PORT"
  nvidia-smi -L || true
  torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=1 \
    --node_rank=$SLURM_PROCID \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train.py \
    --config config_rruff_vista_large.json \
    --distributed
'
