#!/bin/bash
#SBATCH -J xrdbench-prxscale
#SBATCH -A <ALLOCATION>
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 00:20:00
#SBATCH -o $SCRATCH/xrdbench_normalization_sensitivity_20260726/slurm.%j.out
#SBATCH -e $SCRATCH/xrdbench_normalization_sensitivity_20260726/slurm.%j.err

set -euo pipefail

source_job=${SOURCE_JOB:-867777}
input_target_max=${INPUT_TARGET_MAX:-1000}
normalization_mode=${NORMALIZATION_MODE:-target-max}
base=${XRDBENCH_CLEAN_BASE:-$SCRATCH/xrdbench_clean_second_set_20260726}
source_run=${base}/runs/job_${source_job}
tools=${base}/tools
eval_script=${EVAL_SCRIPT:-${tools}/eval_xrdbench_sg_model_rruff_common_grid.py}
venv=${base}/venv
result_parent=${RESULT_PARENT:-$SCRATCH/xrdbench_normalization_sensitivity_20260726}
run_root=${result_parent}/job_${SLURM_JOB_ID}_${normalization_mode}_targetmax_${input_target_max}
mapping=${tools}/canonical_extinction_to_space_group.csv
rruff473=/work2/09870/anon/rruff-benchmark/RRUFF_option1_473_with_buckets_maxnorm.hdf5
rruff325=/work2/09870/anon/rruff-benchmark/RRUFF_usable_plus_recoverable_325_with_labels_maxnorm.hdf5

mkdir "${run_root}"

module purge
module load gcc/13.2.0 cuda/12.8 python3/3.11.8
source "${venv}/bin/activate"

python - <<'PY'
import json
import torch

assert torch.cuda.is_available(), "CUDA is not available"
print(json.dumps({
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
}, indent=2))
PY

evaluate_checkpoint() {
  local model_name=$1
  local checkpoint=$2
  local repo=$3
  local model_root=${run_root}/${model_name}

  mkdir "${model_root}"
  for benchmark in rruff473 rruff325; do
    local eval_h5
    if [[ "${benchmark}" == "rruff473" ]]; then
      eval_h5=${rruff473}
    else
      eval_h5=${rruff325}
    fi

    local benchmark_dir=${model_root}/${benchmark}
    mkdir "${benchmark_dir}"
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH="${repo}" python \
      "${eval_script}" \
      --checkpoint "${checkpoint}" \
      --eval-h5 "${eval_h5}" \
      --mapping-csv "${mapping}" \
      --output-json "${benchmark_dir}/eval.json" \
      --output-npz "${benchmark_dir}/posterior.npz" \
      --device cuda:0 \
      --batch-size 32 \
      --target-label-base 1 \
      --input-target-max "${input_target_max}" \
      --normalization-mode "${normalization_mode}"
  done
}

read_checkpoint() {
  local model_dir=$1
  local pointer=${source_run}/${model_dir}/selected_checkpoint.txt
  [[ -s "${pointer}" ]] || {
    printf 'Missing selected-checkpoint pointer: %s\n' "${pointer}" >&2
    exit 1
  }
  head -n 1 "${pointer}"
}

evaluate_checkpoint \
  cnn10 \
  "$(read_checkpoint cnn10)" \
  "${source_run}/cnn10/repo"
evaluate_checkpoint \
  PatchTST \
  "$(read_checkpoint PatchTST)" \
  "${source_run}/PatchTST/repo"

source_repo=${base}/source
evaluate_checkpoint \
  cnn11 \
  "$SCRATCH/XRDBench/checkpoints/spg-cnn11_lr0.00025_bs128_2026-05-31 16:55:45_100/checkpoint.pth" \
  "${source_repo}"
evaluate_checkpoint \
  bigru \
  "$SCRATCH/XRDBench/checkpoints/spg-bigru_lr0.00025_bs128_2026-05-31 16:55:44_100/checkpoint.pth" \
  "${source_repo}"
evaluate_checkpoint \
  transformer \
  "$SCRATCH/XRDBench/checkpoints/spg-transformer_lr0.00025_bs128_2026-05-31 16:55:44_100/checkpoint.pth" \
  "${source_repo}"

python - "${run_root}" "${source_job}" "${input_target_max}" "${normalization_mode}" <<'PY'
import json
import pathlib
import sys

run_root = pathlib.Path(sys.argv[1])
source_job = int(sys.argv[2])
input_target_max = float(sys.argv[3])
normalization_mode = sys.argv[4]
models = {}
for eval_path in sorted(run_root.glob("*/*/eval.json")):
    result = json.loads(eval_path.read_text())
    model_name = eval_path.parents[1].name
    benchmark = eval_path.parent.name
    models.setdefault(model_name, {})[benchmark] = {
        "n": result["n"],
        "top1_eg": result["top1_eg"],
        "top3_eg": result["top3_eg"],
        "top5_eg": result["top5_eg"],
        "normalization_mode": result["preprocessing"]["normalization_mode"],
        "input_target_max": result["preprocessing"]["input_target_max"],
    }

summary = {
    "source_training_job": source_job,
    "normalization_mode": normalization_mode,
    "input_target_max": input_target_max,
    "normalization": {
        "target-max": (
            "Each aligned 10-80 degree RRUFF pattern is independently scaled "
            "to the requested input maximum."
        ),
        "preserve": (
            "The stored PRX RRUFF intensity scale is retained after exact "
            "10-80 degree crop and integer decimation."
        ),
    }[normalization_mode],
    "models": models,
}
(run_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
