#!/bin/bash
#SBATCH -J rruff-thresh-sense
#SBATCH -A CDA24014
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00
#SBATCH -o /scratch/ALLOC3/UserE/rruff_thresh_sense_%j.out
#SBATCH -e /scratch/ALLOC3/UserE/rruff_thresh_sense_%j.err

set -euo pipefail

module load gcc/13.2.0 cuda/12.5 python3/3.11.8
source /scratch/ALLOC3/UserE/project-venv/bin/activate
cd /scratch/ALLOC3/UserE/project/Code/ViT_NVIDIA

CKPT=/scratch/ALLOC3/UserE/ai_diffraction_models/xrd_model_82ept35h_best.pth
CFG=/scratch/ALLOC3/UserE/project/Code/ViT_NVIDIA/config_rruff_conditioned_dualsource_2346k_500kpo_from_ic6gfmvm_physpe_coord.json
PRIOR=/scratch/ALLOC3/UserE/ai_diffraction_generated/rruff_conditioned_500k_po_v1_trainready.hdf5
SOURCE473=/work2/ALLOC3/UserE/rruff-benchmark/RRUFF_option1_473_with_buckets_maxnorm.hdf5
OUTDIR=/scratch/ALLOC3/UserE/rruff_threshold_sensitivity_20260422
mkdir -p "$OUTDIR"

make_subset () {
  local tag="$1"
  local usable="$2"
  local recoverable="$3"
  local poor="$4"
  python build_rruff_nuisance_threshold_subset.py \
    --input-h5 "$SOURCE473" \
    --output-h5 "$OUTDIR/${tag}.hdf5" \
    --output-json "$OUTDIR/${tag}_subset.json" \
    --retain-severity usable_or_better recoverable \
    --usable-max-rwp "$usable" \
    --recoverable-max-rwp "$recoverable" \
    --poor-max-rwp "$poor"
}

run_eval () {
  local tag="$1"
  python evaluate_calibration_metrics.py \
    --checkpoint "$CKPT" \
    --config "$CFG" \
    --eval-data-path "$OUTDIR/${tag}.hdf5" \
    --prior-data-path "$PRIOR" \
    --aux-temperature 5.0 \
    --bootstrap 500 \
    --output-json "$OUTDIR/${tag}_metrics.json"
}

make_subset baseline_u012_r022_p050 0.12 0.22 0.50
run_eval baseline_u012_r022_p050

make_subset stricter_u012_r018_p050 0.12 0.18 0.50
run_eval stricter_u012_r018_p050

make_subset looser_u012_r026_p050 0.12 0.26 0.50
run_eval looser_u012_r026_p050

make_subset stricter_u010_r020_p050 0.10 0.20 0.50
run_eval stricter_u010_r020_p050

make_subset looser_u014_r024_p050 0.14 0.24 0.50
run_eval looser_u014_r024_p050
