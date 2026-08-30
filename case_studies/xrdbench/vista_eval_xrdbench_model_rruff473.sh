#!/bin/bash
#SBATCH -A <ALLOCATION>
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00

set -euo pipefail

ROOT=$SCRATCH
REPO_DIR=${REPO_DIR:-$ROOT/XRDBench}
SCRIPT=${SCRIPT:-$ROOT/ai-diffraction/Code/XRDBench/eval_xrdbench_sg_model_rruff473.py}
FAILURE_ADAPTER=${FAILURE_ADAPTER:-$ROOT/ai-diffraction/Code/XRDBench/xrdbench_eval_to_failure_json.py}
TOPOLOGY_SCRIPT=${TOPOLOGY_SCRIPT:-$ROOT/ai-diffraction/Code/ViT_NVIDIA/analyze_topological_error_distance.py}
TOPOLOGY_BOOTSTRAP=${TOPOLOGY_BOOTSTRAP:-$ROOT/ai-diffraction/Code/ViT_NVIDIA/bootstrap_topological_error_distance.py}
CHECKPOINT=${CHECKPOINT:-}
EVAL_H5=${EVAL_H5:-$WORK/rruff-benchmark/RRUFF_option1_473_with_buckets_maxnorm.hdf5}
MAPPING=${MAPPING:-$ROOT/ai-diffraction/Code/Post_Processing/canonical_extinction_to_space_group.csv}
GRAPH_JSON=${GRAPH_JSON:-$ROOT/ai-diffraction/artifacts/topology_graphs/extinction_group_adjacency.json}
MODEL=${MODEL:?MODEL must be set, e.g. transformer, bigru, cnn11}
TASK=${TASK:-spg}
OUT=${OUT:-$ROOT/xrdbench_${MODEL}_rruff473_eval.json}
FAILURE_JSON=${FAILURE_JSON:-$ROOT/xrdbench_${MODEL}_rruff473_failure.json}
TOPOLOGY_JSON=${TOPOLOGY_JSON:-$ROOT/xrdbench_${MODEL}_rruff473_topology.json}
TOPOLOGY_BOOTSTRAP_JSON=${TOPOLOGY_BOOTSTRAP_JSON:-$ROOT/xrdbench_${MODEL}_rruff473_topology_bootstrap.json}
TOPOLOGY_BOOTSTRAP_ROUNDS=${TOPOLOGY_BOOTSTRAP_ROUNDS:-2000}

module load gcc/13.2.0 python3/3.11.8 cuda/12.4
source "$ROOT/xrdbench-venv/bin/activate"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

if ! python - <<'PY'
import h5py  # noqa: F401
PY
then
  python -m pip install --quiet 'h5py>=3.10,<4'
fi

cd "$REPO_DIR"
if [[ -z "$CHECKPOINT" ]]; then
  latest_dir=$(ls -td "checkpoints/${TASK}-${MODEL}_"* 2>/dev/null | head -n 1)
  CHECKPOINT="$latest_dir/checkpoint.pth"
fi

python "$SCRIPT" \
  --checkpoint "$CHECKPOINT" \
  --eval-h5 "$EVAL_H5" \
  --mapping-csv "$MAPPING" \
  --output-json "$OUT"

python "$FAILURE_ADAPTER" \
  --eval-json "$OUT" \
  --model-name "$MODEL" \
  --output-json "$FAILURE_JSON"

python "$TOPOLOGY_SCRIPT" \
  --graph-json "$GRAPH_JSON" \
  --failure-json "$FAILURE_JSON" \
  --output-json "$TOPOLOGY_JSON"

python "$TOPOLOGY_BOOTSTRAP" \
  --graph-json "$GRAPH_JSON" \
  --failure-json "$FAILURE_JSON" \
  --bootstrap "$TOPOLOGY_BOOTSTRAP_ROUNDS" \
  --output-json "$TOPOLOGY_BOOTSTRAP_JSON"
