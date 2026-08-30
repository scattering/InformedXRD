#!/bin/bash
#SBATCH -J cnn-rruff-converge
#SBATCH -A <ALLOCATION>
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 01:30:00
#SBATCH -o $SCRATCH/cnn_rruff_converge.%j.out
#SBATCH -e $SCRATCH/cnn_rruff_converge.%j.err

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Refusing to run outside a Slurm compute allocation. Submit this file with sbatch." >&2
  exit 2
fi

target_epoch=${1:?Usage: vista_resnet_rruff_converge.sh TARGET_EPOCH RESUME_CHECKPOINT}
resume_checkpoint=${2:?Usage: vista_resnet_rruff_converge.sh TARGET_EPOCH RESUME_CHECKPOINT}
code_root=${CNN_CONVERGENCE_CODE_ROOT:-$SCRATCH/neurips_revision_cnn_20260726}
run_name=${CNN_CONVERGENCE_RUN_NAME:-submitted_resnet18_convergence}
max_epoch=${CNN_CONVERGENCE_MAX_EPOCH:-50}

module purge
module load gcc/13.2.0 cuda/12.5 python3/3.11.8
source $SCRATCH/ai-diffraction-venv/bin/activate
cd "${code_root}/Code/Regular_CNN_Nvidia"

python resnet_rruff_train.py \
  --train-h5 $SCRATCH/ai_diffraction_generated/rruff_conditioned_2346k_v1_trainready.hdf5 \
  --eval-h5-473 $WORK/rruff-benchmark/RRUFF_option1_473_with_buckets_maxnorm.hdf5 \
  --eval-h5-325 $WORK/rruff-benchmark/RRUFF_usable_plus_recoverable_325_with_labels_maxnorm.hdf5 \
  --output-dir $SCRATCH/resnet_rruff_runs \
  --run-name "${run_name}" \
  --epochs "${target_epoch}" \
  --batch-size 128 \
  --num-workers 2 \
  --prefetch-factor 4 \
  --learning-rate 1e-3 \
  --weight-decay 1e-4 \
  --lr-factor 0.3 \
  --lr-patience 2 \
  --min-lr 1e-6 \
  --early-stopping-patience 8 \
  --early-stopping-min-delta 1e-3 \
  --minimum-epochs 18 \
  --log-interval 100 \
  --save-every-epoch \
  --io-mode contiguous_h5 \
  --resume-checkpoint "${resume_checkpoint}"

summary=$SCRATCH/resnet_rruff_runs/${run_name}/summary.json
python - "${summary}" "${max_epoch}" <<'PY'
import json
import sys

summary_path, max_epoch = sys.argv[1], int(sys.argv[2])
summary = json.load(open(summary_path))
print(
    f"converged={str(summary['converged']).lower()} "
    f"selected_epoch={summary['selected_epoch']} "
    f"max_epoch={max_epoch}"
)
PY
