#!/bin/bash
#SBATCH -J rank-lift-325
#SBATCH -A <ALLOCATION>
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 00:10:00
#SBATCH -o $SCRATCH/rank_lift_325.%j.out
#SBATCH -e $SCRATCH/rank_lift_325.%j.err

set -euo pipefail

code_root=${NEURIPS_REBUTTAL_CODE_ROOT:-$SCRATCH/neurips_rebuttal_20260726/repo}
output_root=${NEURIPS_REBUTTAL_OUTPUT_ROOT:-$SCRATCH/neurips_rebuttal_20260726}
posterior="${output_root}/mixed2500k_rruff325_posteriors.npz"

module purge
module load gcc/13.2.0 cuda/12.5 python3/3.11.8
source $SCRATCH/ai-diffraction-venv/bin/activate

python "${code_root}/Information Theory/dump_rruff325_calibrated_posteriors.py" \
  --checkpoint $SCRATCH/ai_diffraction_models/xrd_model_82ept35h_best.pth \
  --config "${code_root}/Code/ViT_NVIDIA/config_rruff_conditioned_dualsource_2346k_500kpo_from_ic6gfmvm_physpe_coord.json" \
  --eval-data-path $WORK/rruff-benchmark/RRUFF_usable_plus_recoverable_325_with_labels_maxnorm.hdf5 \
  --prior-data-path $SCRATCH/ai_diffraction_generated/rruff_conditioned_500k_po_v1_trainready.hdf5 \
  --aux-temperature 5.0 \
  --num-workers 2 \
  --output-npz "${posterior}" \
  --output-json "${output_root}/mixed2500k_rruff325_posteriors.json"

python "${code_root}/Information Theory/rank_lift_null.py" \
  --posterior-npz "${posterior}" \
  --n-perm 10000 \
  --tag rruff325_final_mixed_t5 \
  --out "${output_root}/rruff325_final_mixed_t5_rank_lift.json"
