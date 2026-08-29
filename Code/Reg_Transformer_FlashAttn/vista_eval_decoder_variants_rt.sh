#!/bin/bash
#SBATCH -J vista-rt-eval
#SBATCH -A CDA24014
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00
#SBATCH -o /scratch/ALLOC3/UserE/vista_rt_eval_%j.out
#SBATCH -e /scratch/ALLOC3/UserE/vista_rt_eval_%j.err

set -euo pipefail

: "${RT_RUN_ID:?Set RT_RUN_ID to the checkpoint run id before submitting.}"
: "${RT_CONFIG:?Set RT_CONFIG to the config path used for the checkpoint.}"
: "${RT_EVAL_DATA_PATH:?Set RT_EVAL_DATA_PATH to the benchmark HDF5 path.}"
: "${RT_PRIOR_DATA_PATH:?Set RT_PRIOR_DATA_PATH to the HDF5 path providing y_train priors.}"

module load gcc/13.2.0 cuda/12.5 python3/3.11.8
source /scratch/ALLOC3/UserE/project-venv/bin/activate

cd /scratch/ALLOC3/UserE/project/Code/Reg_Transformer_FlashAttn

python evaluate_decoder_variants_multilabel.py \
  --checkpoint "/scratch/ALLOC3/UserE/ai_diffraction_models/xrd_model_${RT_RUN_ID}_best.pth" \
  --config "${RT_CONFIG}" \
  --eval-data-path "${RT_EVAL_DATA_PATH}" \
  --prior-data-path "${RT_PRIOR_DATA_PATH}" \
  --aux-temperature 1.0 \
  --group-keys severity
