#!/bin/bash
#SBATCH -J rank-lift-325
#SBATCH -A PHY26002
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 00:30:00
#SBATCH -o /scratch/ALLOC3/UserE/rank_lift_325.%j.out
#SBATCH -e /scratch/ALLOC3/UserE/rank_lift_325.%j.err

set -euo pipefail

code_root=${NEURIPS_REBUTTAL_CODE_ROOT:-/scratch/ALLOC3/UserE/neurips_rebuttal_20260726/repo}
output_root=${NEURIPS_REBUTTAL_OUTPUT_ROOT:-/scratch/ALLOC3/UserE/neurips_rebuttal_20260726}
posterior="${output_root}/mixed2500k_rruff325_posteriors.npz"

mkdir -p "${output_root}"

python3 "${code_root}/Information Theory/dump_rruff325_calibrated_posteriors.py" \
  --checkpoint "${output_root}/xrd_model_82ept35h_best.pth" \
  --config "${code_root}/Code/ViT_NVIDIA/config_rruff_conditioned_dualsource_2346k_500kpo_from_ic6gfmvm_physpe_coord.json" \
  --eval-data-path "${output_root}/RRUFF_usable_plus_recoverable_325_with_labels_maxnorm.hdf5" \
  --prior-data-path "${output_root}/rruff_conditioned_500k_po_labels_only.hdf5" \
  --aux-temperature 5.0 \
  --num-workers 0 \
  --device cpu \
  --output-npz "${posterior}" \
  --output-json "${output_root}/mixed2500k_rruff325_posteriors.json"

python3 "${code_root}/Information Theory/rank_lift_null.py" \
  --posterior-npz "${posterior}" \
  --n-perm 10000 \
  --tag rruff325_final_mixed_t5 \
  --out "${output_root}/rruff325_final_mixed_t5_rank_lift.json"
