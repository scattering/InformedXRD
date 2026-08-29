#!/bin/bash
#SBATCH -J g2-decoder-vit
#SBATCH -A CDA24014
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 01:00:00
#SBATCH -o /scratch/ALLOC3/UserE/g2_decoder_vit_%j.out
#SBATCH -e /scratch/ALLOC3/UserE/g2_decoder_vit_%j.err

set -euo pipefail

ROOT=/scratch/ALLOC3/UserE
REPO=$ROOT/project
IT_DIR="$REPO/Information Theory"
OUT_DIR=${OUT_DIR:-$ROOT/gate2_decoder_control}
BENCH_H5=${BENCH_H5:?Set BENCH_H5}
BENCH_NPZ=${BENCH_NPZ:?Set BENCH_NPZ}
TAG=${TAG:?Set TAG}
CHECKPOINT=${CHECKPOINT:-$ROOT/ai_diffraction_models/xrd_model_9rwv1qly_best.pth}
CONFIG=${CONFIG:-$REPO/Code/ViT_NVIDIA/config_uniform_2m_blackbird_from_ic6gfmvm_physpe_coord.json}
SYMBOLS_KEY=${SYMBOLS_KEY:-symbols_background_noise_like}
LABEL_INDEXING=${LABEL_INDEXING:-auto}

mkdir -p "$OUT_DIR"
module purge
module load gcc/13.2.0 python3/3.11.8
source "$ROOT/project-venv/bin/activate"

python "$IT_DIR/minimal_vit_synthetic_eval.py" \
  --checkpoint "$CHECKPOINT" \
  --config "$CONFIG" \
  --eval-h5 "$BENCH_H5" \
  --output-json "$OUT_DIR/${TAG}_stage2c_summary.json" \
  --aux-temperature 5.0 \
  --label-indexing "$LABEL_INDEXING" \
  --device cuda

python "$IT_DIR/analyze_gate2_decoder_control.py" \
  --benchmark-npz "$BENCH_NPZ" \
  --symbols-key "$SYMBOLS_KEY" \
  --decoder-summary "$OUT_DIR/${TAG}_stage2c_summary.json" \
  --output-json "$OUT_DIR/${TAG}_comparison.json" \
  --output-md "$OUT_DIR/${TAG}_comparison.md"
