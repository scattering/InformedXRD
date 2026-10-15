#!/bin/bash
#SBATCH -J vista-splitcmp
#SBATCH -A CDA24014
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH -t 00:45:00
#SBATCH -o /scratch/ALLOC3/UserE/vista_split_components_%j.out
#SBATCH -e /scratch/ALLOC3/UserE/vista_split_components_%j.err

set -euo pipefail

module load gcc/13.2.0 cuda/12.5 python3/3.11.8
source /scratch/ALLOC3/UserE/project-venv/bin/activate
cd /scratch/ALLOC3/UserE/project/Code/ViT_NVIDIA

BENCH325=/work2/ALLOC3/UserE/rruff-benchmark/RRUFF_usable_plus_recoverable_325_with_labels_maxnorm.hdf5
BENCH473=/work2/ALLOC3/UserE/rruff-benchmark/RRUFF_option1_473_maxnorm.hdf5
OUTDIR=/scratch/ALLOC3/UserE/split_component_eval_20260401
mkdir -p "$OUTDIR"

run_eval () {
  local label="$1"
  local checkpoint="$2"
  local config="$3"
  local data="$4"
  python -u evaluate_split_head_components_h5.py \
    --checkpoint "$checkpoint" \
    --config "$config" \
    --data-path "$data" \
    --split test \
    --batch-size 256 \
    > "$OUTDIR/${label}.txt"
}

run_eval stage1_rruff325 \
  /scratch/ALLOC3/UserE/ai_diffraction_models/xrd_model_ic6gfmvm_best.pth \
  /scratch/ALLOC3/UserE/project/Code/ViT_NVIDIA/config_uniform_premaint_1381k_from_ic6gfmvm_physpe_coord.json \
  "$BENCH325"

run_eval stage1_rruff473 \
  /scratch/ALLOC3/UserE/ai_diffraction_models/xrd_model_ic6gfmvm_best.pth \
  /scratch/ALLOC3/UserE/project/Code/ViT_NVIDIA/config_uniform_premaint_1381k_from_ic6gfmvm_physpe_coord.json \
  "$BENCH473"

run_eval stage2c_rruff325 \
  /scratch/ALLOC3/UserE/ai_diffraction_models/xrd_model_9rwv1qly_best.pth \
  /scratch/ALLOC3/UserE/project/Code/ViT_NVIDIA/config_rruff_conditioned_2346k_from_ic6gfmvm_physpe_coord.json \
  "$BENCH325"

run_eval stage2c_rruff473 \
  /scratch/ALLOC3/UserE/ai_diffraction_models/xrd_model_9rwv1qly_best.pth \
  /scratch/ALLOC3/UserE/project/Code/ViT_NVIDIA/config_rruff_conditioned_2346k_from_ic6gfmvm_physpe_coord.json \
  "$BENCH473"

run_eval stage2c_aux025_rruff325 \
  /scratch/ALLOC3/UserE/ai_diffraction_models/xrd_model_0879pvmm_best.pth \
  /scratch/ALLOC3/UserE/project/Code/ViT_NVIDIA/config_rruff_conditioned_2346k_from_ic6gfmvm_aux025_physpe_coord.json \
  "$BENCH325"

run_eval stage2c_aux025_rruff473 \
  /scratch/ALLOC3/UserE/ai_diffraction_models/xrd_model_0879pvmm_best.pth \
  /scratch/ALLOC3/UserE/project/Code/ViT_NVIDIA/config_rruff_conditioned_2346k_from_ic6gfmvm_aux025_physpe_coord.json \
  "$BENCH473"

