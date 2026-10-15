#!/bin/bash
#SBATCH -J alphadiff-473
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -o /scratch/ALLOC3/UserE/vista_alphadiff_473_%j.out
#SBATCH -e /scratch/ALLOC3/UserE/vista_alphadiff_473_%j.err

set -euo pipefail

module load gcc/13.2.0 python3/3.11.8
source /scratch/ALLOC3/UserE/project-venv/bin/activate

REPO=/scratch/ALLOC3/UserE/project
MODEL_DIR=/scratch/ALLOC3/UserE/OpenAlphaDiffract
EVAL_H5=/work2/ALLOC3/UserE/rruff-benchmark/RRUFF_option1_473_with_buckets_maxnorm.hdf5
MAPPING_CSV=$REPO/Code/Post_Processing/canonical_extinction_to_space_group.csv
OUT_JSON=/scratch/ALLOC3/UserE/alphadiffract_rruff473${SUBSET_TAG:-}.json

mkdir -p "$MODEL_DIR"
cd "$MODEL_DIR"
for f in model.py config.json maxsub.json example_inference.py model.safetensors; do
  if [ ! -f "$f" ]; then
    curl -L -O "https://huggingface.co/linked-liszt/OpenAlphaDiffract/resolve/main/$f"
  fi
done

python - <<'PY'
mods=['torch','numpy','h5py','safetensors']
for m in mods:
    __import__(m)
print('alpha-eval-env-ok')
PY

EXTRA_ARGS=()
if [ -n "${SUBSET_JSON:-}" ]; then
  EXTRA_ARGS+=(--subset-json "$SUBSET_JSON")
fi

python "$REPO/Code/Regular_CNN_Nvidia/eval_openalphadiffract_rruff473.py" \
  --model-dir "$MODEL_DIR" \
  --eval-h5 "$EVAL_H5" \
  --mapping-csv "$MAPPING_CSV" \
  --output-json "$OUT_JSON" \
  "${EXTRA_ARGS[@]}"
