#!/bin/bash
#SBATCH -J bilstm-norm
#SBATCH -A <ALLOCATION>
#SBATCH -p gb
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 00:10:00
#SBATCH -o $SCRATCH/xrdbench_normalization_sensitivity_20260726/slurm.%j.out
#SBATCH -e $SCRATCH/xrdbench_normalization_sensitivity_20260726/slurm.%j.err

set -euo pipefail

source_job=${SOURCE_JOB:-867777}
base=${XRDBENCH_CLEAN_BASE:-$SCRATCH/xrdbench_clean_second_set_20260726}
source_run=${base}/runs/job_${source_job}
tools=${base}/tools
eval_script=${EVAL_SCRIPT:-${tools}/eval_xrdbench_sg_model_rruff_common_grid.py}
venv=${base}/venv
result_parent=${RESULT_PARENT:-$SCRATCH/xrdbench_normalization_sensitivity_20260726}
run_root=${result_parent}/job_${SLURM_JOB_ID}_bilstm
mapping=${tools}/canonical_extinction_to_space_group.csv
rruff473=$WORK/rruff-benchmark/RRUFF_option1_473_with_buckets_maxnorm.hdf5
rruff325=$WORK/rruff-benchmark/RRUFF_usable_plus_recoverable_325_with_labels_maxnorm.hdf5
checkpoint_pointer=${source_run}/bilstm/selected_checkpoint.txt

[[ -s "${checkpoint_pointer}" ]] || {
  printf 'Missing BiLSTM checkpoint pointer: %s\n' "${checkpoint_pointer}" >&2
  exit 1
}
checkpoint=$(head -n 1 "${checkpoint_pointer}")
repo=${source_run}/bilstm/repo

mkdir "${run_root}"

module purge
module load gcc/13.2.0 cuda/12.8 python3/3.11.8
source "${venv}/bin/activate"

for normalization_mode in target-max preserve; do
  mode_root=${run_root}/${normalization_mode}
  mkdir "${mode_root}"
  for benchmark in rruff473 rruff325; do
    if [[ "${benchmark}" == "rruff473" ]]; then
      eval_h5=${rruff473}
    else
      eval_h5=${rruff325}
    fi
    benchmark_dir=${mode_root}/${benchmark}
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
      --input-target-max 1000 \
      --normalization-mode "${normalization_mode}"
  done
done

python - "${run_root}" "${source_job}" <<'PY'
import json
import pathlib
import sys

run_root = pathlib.Path(sys.argv[1])
source_job = int(sys.argv[2])
results = {}
for eval_path in sorted(run_root.glob("*/*/eval.json")):
    result = json.loads(eval_path.read_text())
    mode = eval_path.parents[1].name
    benchmark = eval_path.parent.name
    results.setdefault(mode, {})[benchmark] = {
        "n": result["n"],
        "top1_eg": result["top1_eg"],
        "top3_eg": result["top3_eg"],
        "top5_eg": result["top5_eg"],
        "normalization_mode": result["preprocessing"]["normalization_mode"],
        "input_target_max": result["preprocessing"]["input_target_max"],
    }

summary = {
    "source_training_job": source_job,
    "model": "bilstm",
    "results": results,
}
(run_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
