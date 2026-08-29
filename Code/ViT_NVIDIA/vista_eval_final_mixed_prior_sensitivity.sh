#!/bin/bash
#SBATCH -J prior-sense-655279
#SBATCH -A CDA24014
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00
#SBATCH -o /scratch/ALLOC3/UserE/prior_sensitivity_%j.out
#SBATCH -e /scratch/ALLOC3/UserE/prior_sensitivity_%j.err

set -euo pipefail

module load gcc/13.2.0 cuda/12.5 python3/3.11.8
source /scratch/ALLOC3/UserE/project-venv/bin/activate
cd /scratch/ALLOC3/UserE/project/Code/ViT_NVIDIA

CKPT=/scratch/ALLOC3/UserE/ai_diffraction_models/xrd_model_82ept35h_best.pth
CFG=/scratch/ALLOC3/UserE/project/Code/ViT_NVIDIA/config_rruff_conditioned_dualsource_2346k_500kpo_from_ic6gfmvm_physpe_coord.json
BENCH325=/work2/ALLOC3/UserE/rruff-benchmark/RRUFF_usable_plus_recoverable_325_with_labels_maxnorm.hdf5
BENCH473=/work2/ALLOC3/UserE/rruff-benchmark/RRUFF_option1_473_with_buckets_maxnorm.hdf5
OUTDIR=/scratch/ALLOC3/UserE/prior_sensitivity_20260422
PRIOR325=/scratch/ALLOC3/UserE/rruff325_eval_prior_20260422.hdf5
PRIOR473=/scratch/ALLOC3/UserE/rruff473_eval_prior_20260422.hdf5
TARGET=${TARGET:-both}
mkdir -p "$OUTDIR"

python make_prior_hdf5_from_eval_labels.py --eval-data-path "$BENCH325" --output-path "$PRIOR325"
python make_prior_hdf5_from_eval_labels.py --eval-data-path "$BENCH473" --output-path "$PRIOR473"

run_eval () {
  local tag="$1"
  local eval_path="$2"
  local prior_path="$3"
  python evaluate_calibration_metrics.py \
    --checkpoint "$CKPT" \
    --config "$CFG" \
    --eval-data-path "$eval_path" \
    --prior-data-path "$prior_path" \
    --aux-temperature 5.0 \
    --bootstrap 500 \
    --output-json "$OUTDIR/${tag}.json"
}

if [[ "$TARGET" == "325" || "$TARGET" == "both" ]]; then
  run_eval r325_prior_po500k "$BENCH325" /scratch/ALLOC3/UserE/ai_diffraction_generated/rruff_conditioned_500k_po_v1_trainready.hdf5
  run_eval r325_prior_stage2 "$BENCH325" /scratch/ALLOC3/UserE/ai_diffraction_generated/rruff_conditioned_2346k_v1_trainready.hdf5
  run_eval r325_prior_uniform2m "$BENCH325" /scratch/ALLOC3/UserE/ai_diffraction_generated/uniform2m_v1_partial_trainready_ordered_v2.hdf5
  run_eval r325_prior_eval325 "$BENCH325" "$PRIOR325"
  run_eval r325_prior_eval473 "$BENCH325" "$PRIOR473"
fi

if [[ "$TARGET" == "473" || "$TARGET" == "both" ]]; then
  run_eval r473_prior_po500k "$BENCH473" /scratch/ALLOC3/UserE/ai_diffraction_generated/rruff_conditioned_500k_po_v1_trainready.hdf5
  run_eval r473_prior_stage2 "$BENCH473" /scratch/ALLOC3/UserE/ai_diffraction_generated/rruff_conditioned_2346k_v1_trainready.hdf5
  run_eval r473_prior_uniform2m "$BENCH473" /scratch/ALLOC3/UserE/ai_diffraction_generated/uniform2m_v1_partial_trainready_ordered_v2.hdf5
  run_eval r473_prior_eval473 "$BENCH473" "$PRIOR473"
  run_eval r473_prior_eval325 "$BENCH473" "$PRIOR325"
fi
