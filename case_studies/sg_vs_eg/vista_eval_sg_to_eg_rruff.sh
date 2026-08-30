#!/bin/bash
#SBATCH -J sg-eg-rruff
#SBATCH -A <ALLOCATION>
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 00:20:00
#SBATCH -o $SCRATCH/sg_eg_rruff.%j.out
#SBATCH -e $SCRATCH/sg_eg_rruff.%j.err

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Refusing to run outside a Slurm compute allocation. Submit this file with sbatch." >&2
  exit 2
fi

code_root=${NEURIPS_REBUTTAL_CODE_ROOT:-$SCRATCH/neurips_rebuttal_20260726/repo}
output_root=${NEURIPS_REBUTTAL_OUTPUT_ROOT:-$SCRATCH/neurips_rebuttal_20260726}

module purge
module load gcc/13.2.0 cuda/12.5 python3/3.11.8
source $SCRATCH/ai-diffraction-venv/bin/activate

python "${code_root}/Code/ViT_NVIDIA/evaluate_sg_to_eg_rruff.py" \
  --sg-checkpoint $SCRATCH/ai_diffraction_models/xrd_model_lw1dbmsm_best.pth \
  --sg-config "${code_root}/Code/ViT_NVIDIA/config_uniform_2m_blackbird_sg230_physpe_coord_referee_rerun.json" \
  --eg-checkpoint $SCRATCH/ai_diffraction_models/xrd_model_xgy3h6uw_best.pth \
  --eg-config "${code_root}/Code/ViT_NVIDIA/config_uniform_2m_blackbird_from_ic6gfmvm_physpe_coord.json" \
  --canonical-table-path "${code_root}/Code/Post_Processing/canonical_extinction_to_space_group.csv" \
  --benchmark rruff473=$WORK/rruff-benchmark/RRUFF_option1_473_with_buckets_maxnorm.hdf5 \
  --benchmark rruff325=$WORK/rruff-benchmark/RRUFF_usable_plus_recoverable_325_with_labels_maxnorm.hdf5 \
  --num-workers 4 \
  --output-json "${output_root}/sg_to_eg_real_rruff.json"
