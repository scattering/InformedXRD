#!/bin/bash
#SBATCH -J alpha-209-no-mp
#SBATCH -A CDA24014
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 01:00:00
#SBATCH -o /scratch/ALLOC3/UserE/alpha_209_no_mp_%j.out
#SBATCH -e /scratch/ALLOC3/UserE/alpha_209_no_mp_%j.err

set -euo pipefail

ROOT=/scratch/ALLOC3/UserE
REPO=$ROOT/project
IT_DIR="$REPO/Information Theory"
MAP_CSV=$REPO/Code/Post_Processing/canonical_extinction_to_space_group.csv
MODEL_DIR="$ROOT/OpenAlphaDiffract"
EVAL_H5=${EVAL_H5:-/work2/ALLOC3/UserE/stage2c_stampede/RRUFF_option1_473_xy_processed_overlap_209_maxnorm.hdf5}
SUBSET_JSON=${SUBSET_JSON:-$ROOT/rruff209_no_mp_evidence_92.json}
OUT_DIR=${OUT_DIR:-/scratch/ALLOC3/UserE/alphadiffract_no_mp_evidence_209}
TAG=${TAG:-alphadiffract_rruff209_no_mp_evidence_92}

module purge
module load gcc/13.2.0 python3/3.11.8
source "$ROOT/project-venv/bin/activate"

mkdir -p "$OUT_DIR"

python "$REPO/Code/Regular_CNN_Nvidia/eval_openalphadiffract_rruff473.py" \
  --model-dir "$MODEL_DIR" \
  --eval-h5 "$EVAL_H5" \
  --mapping-csv "$MAP_CSV" \
  --subset-json "$SUBSET_JSON" \
  --output-json "$OUT_DIR/${TAG}_eval.json" \
  --output-npz "$OUT_DIR/${TAG}_posteriors.npz" \
  --device cuda

python "$IT_DIR/summarize_posterior_npz.py" \
  --posterior-npz "$OUT_DIR/${TAG}_posteriors.npz" \
  --label "$TAG" \
  --output-json "$OUT_DIR/${TAG}_summary.json"
