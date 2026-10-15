#!/bin/bash
#SBATCH -J eval-r325-temp
#SBATCH -A CDA24014
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH -t 00:45:00
#SBATCH -o /scratch/ALLOC3/UserE/eval_r325_temp_sweep_%j.out
#SBATCH -e /scratch/ALLOC3/UserE/eval_r325_temp_sweep_%j.err

set -euo pipefail

if [ $# -ne 2 ]; then
  echo "usage: $0 CHECKPOINT CONFIG" >&2
  exit 2
fi

CHECKPOINT="$1"
CONFIG="$2"

module load gcc/13.2.0 cuda/12.5 python3/3.11.8
source /scratch/ALLOC3/UserE/project-venv/bin/activate
cd /scratch/ALLOC3/UserE/project/Code/ViT_NVIDIA

python -u evaluate_cross_model_fusion.py \
  --split-checkpoint "$CHECKPOINT" \
  --split-config "$CONFIG" \
  --aux-checkpoint "$CHECKPOINT" \
  --aux-config "$CONFIG" \
  --eval-data-path /work2/ALLOC3/UserE/rruff-benchmark/RRUFF_usable_plus_recoverable_325_with_labels_maxnorm.hdf5 \
  --prior-data-path /scratch/ALLOC3/UserE/ai_diffraction_generated/rruff_conditioned_2346k_v1_trainready.hdf5 \
  --output-json /scratch/ALLOC3/UserE/eval_xgy_stage2_r325_temp_sweep.json \
  > /scratch/ALLOC3/UserE/eval_xgy_stage2_r325_temp_sweep.txt
