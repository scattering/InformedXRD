#!/bin/bash
#SBATCH -J ml4px-fail
#SBATCH -A PHY26002
#SBATCH -p gg-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -o /scratch/ALLOC3/UserE/ml4pxrds_failure_modes.%j.out
#SBATCH -e /scratch/ALLOC3/UserE/ml4pxrds_failure_modes.%j.err

set -euo pipefail

module load gcc/13.2.0 python3/3.11.8

VENV=${VENV:-/scratch/ALLOC3/UserE/ml4pxrds-eval-venv}
REPO=${REPO:-/scratch/ALLOC3/UserE/project}
REL=${REL:-/scratch/ALLOC3/UserE/ml4pxrds_release}
MODEL_NAME=${MODEL_NAME:-resnet50_experimental}
SQRT_MODE=${SQRT_MODE:-off}
EVAL_H5=${EVAL_H5:-/work2/ALLOC3/UserE/rruff-benchmark/RRUFF_option1_473_with_buckets_maxnorm.hdf5}
MAP_CSV=${MAP_CSV:-${REPO}/Code/Post_Processing/canonical_extinction_to_space_group.csv}
OUT_DIR=${OUT_DIR:-/scratch/ALLOC3/UserE/ml4pxrds_audit}

case "${MODEL_NAME}" in
  resnet50_experimental)
    MODEL_ZIP=${REL}/resnet_50_experimental_weights.zip
    MODEL_DIR=${REL}/resnet_50_experimental/final
    ;;
  resnet101_synthetic)
    MODEL_ZIP=${REL}/resnet_101_weights.zip
    MODEL_DIR=${REL}/resnet_101/final
    ;;
  *)
    echo "Unknown MODEL_NAME=${MODEL_NAME}" >&2
    exit 2
    ;;
esac

source "${VENV}/bin/activate"

mkdir -p "${REL}" "${OUT_DIR}"
if [ ! -f "${MODEL_ZIP}" ]; then
  case "${MODEL_NAME}" in
    resnet50_experimental)
      URL="https://github.com/aimat-lab/ML4pXRDs/releases/download/v1.0/resnet_50_experimental_weights.zip"
      ;;
    resnet101_synthetic)
      URL="https://github.com/aimat-lab/ML4pXRDs/releases/download/v1.0/resnet_101_weights.zip"
      ;;
  esac
  curl -L -o "${MODEL_ZIP}" "${URL}"
fi

if [ ! -f "${MODEL_DIR}/saved_model.pb" ]; then
  rm -rf "${MODEL_DIR%/final}"
  unzip -q "${MODEL_ZIP}" -d "${MODEL_DIR%/final}"
fi

OUT_JSON="${OUT_DIR}/${MODEL_NAME}_${SQRT_MODE}_failure_modes.json"

python "${REPO}/Code/Regular_CNN_Nvidia/eval_ml4pxrds_failure_modes.py" \
  --model-dir "${MODEL_DIR}" \
  --eval-h5 "${EVAL_H5}" \
  --mapping-csv "${MAP_CSV}" \
  --sqrt "${SQRT_MODE}" \
  --model-name "${MODEL_NAME}_${SQRT_MODE}" \
  --output-json "${OUT_JSON}"
