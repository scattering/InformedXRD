#!/bin/bash
#SBATCH -J vista-sg2eg-ctl
#SBATCH -A <ALLOCATION>
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 01:00:00
#SBATCH -o $SCRATCH/vista_sg2eg_control_%j.out
#SBATCH -e $SCRATCH/vista_sg2eg_control_%j.err

set -euo pipefail

module load gcc/13.2.0 cuda/12.5 python3/3.11.8
source $SCRATCH/ai-diffraction-venv/bin/activate

cd $SCRATCH/ai-diffraction/Code/ViT_NVIDIA

python evaluate_sg_to_eg_control.py \
  --sg-checkpoint $SCRATCH/ai_diffraction_models/xrd_model_x10oanpe_latest.pth \
  --sg-config $SCRATCH/ai-diffraction/Code/ViT_NVIDIA/config_uniform_2m_blackbird_sg230_physpe_coord_1epoch.json \
  --sg-eval-data-path $SCRATCH/ai_diffraction_generated/uniform_2m_blackbird_20260401_sg_trainready.hdf5 \
  --eg-checkpoint $SCRATCH/ai_diffraction_models/xrd_model_xgy3h6uw_best.pth \
  --eg-config $SCRATCH/ai-diffraction/Code/ViT_NVIDIA/config_uniform_2m_blackbird_from_ic6gfmvm_physpe_coord.json \
  --eg-truth-data-path $SCRATCH/ai_diffraction_generated/uniform_2m_blackbird_20260401_trainready.hdf5 \
  --canonical-table-path $SCRATCH/ai-diffraction/Code/Post_Processing/canonical_extinction_to_space_group.csv \
  --output-json $SCRATCH/sg_to_eg_control_2m_uniform.json \
  --output-csv $SCRATCH/sg_to_eg_control_2m_uniform_topk.csv
