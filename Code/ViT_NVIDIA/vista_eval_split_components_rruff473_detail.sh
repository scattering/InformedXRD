#!/bin/bash
#SBATCH -J vista-split473
#SBATCH -A CDA24014
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH -t 00:45:00
#SBATCH -o /scratch/ALLOC3/UserE/vista_split473_%j.out
#SBATCH -e /scratch/ALLOC3/UserE/vista_split473_%j.err

set -euo pipefail

module load gcc/13.2.0 cuda/12.5 python3/3.11.8
source /scratch/ALLOC3/UserE/project-venv/bin/activate
cd /scratch/ALLOC3/UserE/project/Code/ViT_NVIDIA

BENCH473=/work2/ALLOC3/UserE/rruff-benchmark/RRUFF_option1_473_maxnorm.hdf5
OUTDIR=/scratch/ALLOC3/UserE/split_component_eval_20260601
mkdir -p "$OUTDIR"

run_eval () {
  local label="$1"
  local checkpoint="$2"
  local config="$3"
  python -u evaluate_split_head_components_h5.py \
    --checkpoint "$checkpoint" \
    --config "$config" \
    --data-path "$BENCH473" \
    --split test \
    --batch-size 256 \
    --output-json "$OUTDIR/${label}.json" \
    --save-logits "$OUTDIR/${label}.npz"
}

run_eval stage2c_rruff473_detail \
  /scratch/ALLOC3/UserE/ai_diffraction_models/xrd_model_9rwv1qly_best.pth \
  /scratch/ALLOC3/UserE/project/Code/ViT_NVIDIA/config_rruff_conditioned_2346k_from_ic6gfmvm_physpe_coord.json

run_eval mixed2500k_rruff473_detail \
  /scratch/ALLOC3/UserE/ai_diffraction_models/xrd_model_82ept35h_best.pth \
  /scratch/ALLOC3/UserE/project/Code/ViT_NVIDIA/config_rruff_conditioned_dualsource_2346k_500kpo_from_ic6gfmvm_physpe_coord.json
