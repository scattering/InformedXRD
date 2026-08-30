#!/bin/bash
#SBATCH -J rank-lift-473
#SBATCH -A <ALLOCATION>
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -t 00:05:00
#SBATCH -o $SCRATCH/rank_lift_473.%j.out
#SBATCH -e $SCRATCH/rank_lift_473.%j.err

set -euo pipefail

code_root=${NEURIPS_REBUTTAL_CODE_ROOT:-$SCRATCH/neurips_rebuttal_20260726/repo}
output_root=${NEURIPS_REBUTTAL_OUTPUT_ROOT:-$SCRATCH/neurips_rebuttal_20260726}
posterior=${RRUFF473_POSTERIOR:-$SCRATCH/corrected_flat_information_theory_20260623/corrected_flat37_stage2c_rruff473_t5_posteriors.npz}

module purge
module load gcc/13.2.0 cuda/12.5 python3/3.11.8
source $SCRATCH/ai-diffraction-venv/bin/activate

python "${code_root}/Information Theory/rank_lift_null.py" \
  --posterior-npz "${posterior}" \
  --n-perm 10000 \
  --tag rruff473_stage2c_t5 \
  --out "${output_root}/rruff473_stage2c_t5_rank_lift.json"
