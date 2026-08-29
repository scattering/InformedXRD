#!/bin/bash
#SBATCH -J vista-sg2eg-ref
#SBATCH -A CDA24014
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 01:00:00
#SBATCH -o /scratch/ALLOC3/UserE/vista_sg2eg_ref_%j.out
#SBATCH -e /scratch/ALLOC3/UserE/vista_sg2eg_ref_%j.err

set -euo pipefail

: "${SG_RUN_ID:?Set SG_RUN_ID to the rerun W&B / checkpoint run id before submitting.}"

module load gcc/13.2.0 cuda/12.5 python3/3.11.8
source /scratch/ALLOC3/UserE/project-venv/bin/activate

SG_MODEL_DIR=${SG_MODEL_DIR:-/scratch/ALLOC3/UserE/ai_diffraction_models}
SG_CONFIG=${SG_CONFIG:-/scratch/ALLOC3/UserE/project/Code/ViT_NVIDIA/config_uniform_2m_blackbird_sg230_physpe_coord_referee_rerun.json}
SG_CHECKPOINT=${SG_CHECKPOINT:-${SG_MODEL_DIR}/xrd_model_${SG_RUN_ID}_best.pth}
EG_CHECKPOINT=${EG_CHECKPOINT:-/scratch/ALLOC3/UserE/ai_diffraction_models/xrd_model_xgy3h6uw_best.pth}
OUT_STEM=${OUT_STEM:-/scratch/ALLOC3/UserE/sg_to_eg_control_referee_${SG_RUN_ID}}

cd /scratch/ALLOC3/UserE/project/Code/ViT_NVIDIA

# Evaluate the rerun SG model in SG space, collapse probabilities to EG at
# scoring time, and compare against the existing direct-EG control checkpoint.
python evaluate_sg_to_eg_control.py \
  --sg-checkpoint "${SG_CHECKPOINT}" \
  --sg-config "${SG_CONFIG}" \
  --sg-eval-data-path /scratch/ALLOC3/UserE/ai_diffraction_generated/uniform_2m_blackbird_20260401_sg_trainready.hdf5 \
  --eg-checkpoint "${EG_CHECKPOINT}" \
  --eg-config /scratch/ALLOC3/UserE/project/Code/ViT_NVIDIA/config_uniform_2m_blackbird_from_ic6gfmvm_physpe_coord.json \
  --eg-truth-data-path /scratch/ALLOC3/UserE/ai_diffraction_generated/uniform_2m_blackbird_20260401_trainready.hdf5 \
  --canonical-table-path /scratch/ALLOC3/UserE/project/Code/Post_Processing/canonical_extinction_to_space_group.csv \
  --output-json "${OUT_STEM}.json" \
  --output-csv "${OUT_STEM}_topk.csv"
