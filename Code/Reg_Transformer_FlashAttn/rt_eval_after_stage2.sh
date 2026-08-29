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

# Matched RT-physics eval on RRUFF-325/473, run as a dependency after Stage-2.
# Resolves the Stage-2 W&B run id from the Stage-2 job log (STAGE2_JOB).
set -euo pipefail

: "${STAGE2_JOB:?Set STAGE2_JOB to the Stage-2 slurm job id}"

module load gcc/13.2.0 cuda/12.5 python3/3.11.8
source /scratch/ALLOC3/UserE/project-venv/bin/activate
export WANDB_API_KEY
WANDB_API_KEY="$(cat /scratch/ALLOC3/UserE/project/.wandb_api_key)"

cd /scratch/ALLOC3/UserE/project/Code/Reg_Transformer_FlashAttn

RT_RUN_ID="$(STAGE2_JOB="$STAGE2_JOB" python3 - <<'PY'
import os, re
from pathlib import Path
job = os.environ["STAGE2_JOB"]
text = ""
for ext in ("err", "out"):
    p = Path(f"/scratch/ALLOC3/UserE/vista_rt_rruff2346_{job}.{ext}")
    if p.exists():
        text += p.read_text(errors="ignore") + "\n"
m = re.search(r"/runs/([a-z0-9]+)", text) or re.search(r"run-[0-9_]+-([a-z0-9]+)", text)
if not m:
    raise SystemExit("could not resolve Stage-2 run id")
print(m.group(1))
PY
)"
echo "Resolved Stage-2 RT_RUN_ID=$RT_RUN_ID"

CKPT="/scratch/ALLOC3/UserE/ai_diffraction_models/xrd_model_${RT_RUN_ID}_best.pth"
CFG="config_rruff_conditioned_2346k_from_rt_uniform_physpe_coord.json"
PRIOR="/scratch/ALLOC3/UserE/ai_diffraction_generated/rruff_conditioned_2346k_v1_trainready.hdf5"
BENCH_DIR="/work2/ALLOC3/UserE/rruff-benchmark"
OUT="/scratch/ALLOC3/UserE/rt_physics_matched_eval"
GRAPH_JSON="/scratch/ALLOC3/UserE/project/artifacts/topology_graphs/extinction_group_adjacency.json"
mkdir -p "$OUT"

declare -A BM=(
  [325]="RRUFF_usable_plus_recoverable_325_with_labels_maxnorm.hdf5"
  [473]="RRUFF_option1_473_with_buckets_maxnorm.hdf5"
)

for tag in 325 473; do
  for T in 1 5; do
    echo "=== RRUFF-$tag  T=$T ==="
    extra_args=()
    if [[ "$T" == "5" ]]; then
      extra_args+=(--failure-json "${OUT}/rt_rruff${tag}_failure.json" --failure-model-name rt_aux_bayes_t5)
    fi
    python evaluate_decoder_variants_multilabel.py \
      --checkpoint "$CKPT" \
      --config "$CFG" \
      --eval-data-path "${BENCH_DIR}/${BM[$tag]}" \
      --prior-data-path "$PRIOR" \
      --aux-temperature "$T" \
      --group-keys severity \
      "${extra_args[@]}" \
      > "${OUT}/rt_physics_rruff${tag}_T${T}.txt" 2>&1 || echo "  (eval failed for ${tag} T=${T}; see log)"
    tail -n 25 "${OUT}/rt_physics_rruff${tag}_T${T}.txt" || true
    if [[ "$T" == "5" && -f "${OUT}/rt_rruff${tag}_failure.json" ]]; then
      python /scratch/ALLOC3/UserE/project/Code/ViT_NVIDIA/analyze_topological_error_distance.py \
        --graph-json "$GRAPH_JSON" \
        --failure-json "${OUT}/rt_rruff${tag}_failure.json" \
        --output-json "${OUT}/rt_rruff${tag}_topology.json" \
        >> "${OUT}/rt_physics_rruff${tag}_T${T}.txt" 2>&1 || echo "  (topology failed for ${tag}; see log)"
    fi
  done
done

echo "DONE: results under $OUT"
