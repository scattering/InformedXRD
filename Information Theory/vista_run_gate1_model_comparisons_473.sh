#!/bin/bash
#SBATCH -J it-gate1-473
#SBATCH -A CDA24014
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 02:00:00
#SBATCH -o /scratch/ALLOC3/UserE/it_gate1_473.%j.out
#SBATCH -e /scratch/ALLOC3/UserE/it_gate1_473.%j.err

set -euo pipefail

ROOT=/scratch/ALLOC3/UserE
REPO=$ROOT/project
IT_DIR="$REPO/Information Theory"
OUT_DIR=$ROOT/information_theory_model_comparison_473
BENCH473=/work2/ALLOC3/UserE/rruff-benchmark/RRUFF_option1_473_with_buckets_maxnorm.hdf5
MAP_CSV=$REPO/Code/Post_Processing/canonical_extinction_to_space_group.csv

mkdir -p "$OUT_DIR"

module purge
module load gcc/13.2.0 python3/3.11.8

run_gate1 () {
  local label="$1"
  local npz="$2"
  source "$ROOT/project-venv/bin/activate"
  python "$IT_DIR/analyze_gate1_suite.py" \
    --posterior-npz "$npz" \
    --label "$label" \
    --bootstrap 1000 \
    --permutation-rounds 1000 \
    --output-json "$OUT_DIR/${label}_gate1.json"
}

source "$ROOT/project-venv/bin/activate"

python "$IT_DIR/dump_rruff325_calibrated_posteriors.py" \
  --checkpoint "$ROOT/ai_diffraction_models/xrd_model_9rwv1qly_best.pth" \
  --config "$REPO/Code/ViT_NVIDIA/config_rruff_conditioned_2346k_from_ic6gfmvm_physpe_coord.json" \
  --eval-data-path "$BENCH473" \
  --prior-data-path "$ROOT/ai_diffraction_generated/rruff_conditioned_2346k_v1_trainready.hdf5" \
  --aux-temperature 5.0 \
  --output-npz "$OUT_DIR/stage2c_rruff473_posteriors.npz" \
  --output-json "$OUT_DIR/stage2c_rruff473_posteriors.json"
run_gate1 stage2c_rruff473 "$OUT_DIR/stage2c_rruff473_posteriors.npz"

python "$IT_DIR/dump_rruff325_calibrated_posteriors.py" \
  --checkpoint "$ROOT/ai_diffraction_models/xrd_model_82ept35h_best.pth" \
  --config "$REPO/Code/ViT_NVIDIA/config_rruff_conditioned_dualsource_2346k_500kpo_from_ic6gfmvm_physpe_coord.json" \
  --eval-data-path "$BENCH473" \
  --prior-data-path "$ROOT/ai_diffraction_generated/rruff_conditioned_500k_po_v1_trainready.hdf5" \
  --aux-temperature 5.0 \
  --output-npz "$OUT_DIR/mixed2500k_rruff473_posteriors.npz" \
  --output-json "$OUT_DIR/mixed2500k_rruff473_posteriors.json"
run_gate1 mixed2500k_rruff473 "$OUT_DIR/mixed2500k_rruff473_posteriors.npz"

python "$REPO/Code/Regular_CNN_Nvidia/eval_openalphadiffract_rruff473.py" \
  --model-dir "$ROOT/OpenAlphaDiffract" \
  --eval-h5 "$BENCH473" \
  --mapping-csv "$MAP_CSV" \
  --output-json "$OUT_DIR/alphadiffract_rruff473_eval.json" \
  --output-npz "$OUT_DIR/alphadiffract_rruff473_posteriors.npz"
run_gate1 alphadiffract_rruff473 "$OUT_DIR/alphadiffract_rruff473_posteriors.npz"

deactivate

for model in bigru cnn11 transformer; do
  source "$ROOT/xrdbench-venv/bin/activate"
  export PYTHONPATH="$ROOT/XRDBench${PYTHONPATH:+:$PYTHONPATH}"
  checkpoint=$(ls -td "$ROOT/XRDBench/checkpoints/spg-${model}_"* 2>/dev/null | head -n 1)/checkpoint.pth
  python "$REPO/Code/XRDBench/eval_xrdbench_sg_model_rruff473.py" \
    --checkpoint "$checkpoint" \
    --eval-h5 "$BENCH473" \
    --mapping-csv "$MAP_CSV" \
    --output-json "$OUT_DIR/xrdbench_${model}_rruff473_eval.json" \
    --output-npz "$OUT_DIR/xrdbench_${model}_rruff473_posteriors.npz"
  deactivate
  run_gate1 "xrdbench_${model}_rruff473" "$OUT_DIR/xrdbench_${model}_rruff473_posteriors.npz"
done

python - <<'PY'
import json, glob, os
rows = []
for path in sorted(glob.glob("/scratch/ALLOC3/UserE/information_theory_model_comparison_473/*_gate1.json")):
    d = json.load(open(path))
    c = d["compact_summary"]
    rows.append({
        "label": d["label"],
        "bits_over_prior_bits": c["bits_over_prior_bits"],
        "bits_null_mean": c["bits_over_prior_null_mean"],
        "top1": c["top1"],
        "top1_null_mean": c["top1_null_mean"],
        "top5": c["top5"],
        "top5_null_mean": c["top5_null_mean"],
        "top5_lift": c["top5_pattern_conditional_lift"],
        "mean_true_rank": c["mean_true_rank"],
        "mean_true_rank_null_mean": c["mean_true_rank_null_mean"],
    })
out = "/scratch/ALLOC3/UserE/information_theory_model_comparison_473/summary.json"
with open(out, "w") as f:
    json.dump(rows, f, indent=2)
print(json.dumps(rows, indent=2))
PY
