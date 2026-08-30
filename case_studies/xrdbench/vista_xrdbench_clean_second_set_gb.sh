#!/bin/bash
#SBATCH -J xrdbench-clean2
#SBATCH -A <ALLOCATION>
#SBATCH -p gb
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 48
#SBATCH -t 03:00:00
#SBATCH -o $SCRATCH/xrdbench_clean_second_set_20260726/slurm.%j.out
#SBATCH -e $SCRATCH/xrdbench_clean_second_set_20260726/slurm.%j.err

set -euo pipefail

base=${XRDBENCH_CLEAN_BASE:-$SCRATCH/xrdbench_clean_second_set_20260726}
source_repo=${base}/source
tools=${base}/tools
venv=${base}/venv
run_group=${base}/runs/job_${SLURM_JOB_ID}
mapping=${tools}/canonical_extinction_to_space_group.csv
graph=${tools}/extinction_group_adjacency.json
train_manifest=$SCRATCH/simxrd_train1_pt_shards/manifest.json
val_manifest=$SCRATCH/simxrd_val_pt_shards/manifest.json
test_manifest=$SCRATCH/simxrd_test_pt_shards/manifest.json
rruff473=/work2/09870/anon/rruff-benchmark/RRUFF_option1_473_with_buckets_maxnorm.hdf5
rruff325=/work2/09870/anon/rruff-benchmark/RRUFF_usable_plus_recoverable_325_with_labels_maxnorm.hdf5

module purge
module load gcc/13.2.0 cuda/12.8 python3/3.11.8
source "${venv}/bin/activate"
export WANDB_MODE=offline
export OMP_NUM_THREADS=4

mkdir "${run_group}"
nvidia-smi -L
python - <<'PY'
import json
import torch

assert torch.cuda.is_available(), "CUDA is not available"
assert torch.cuda.device_count() >= 3, "At least three GPUs are required"
print(json.dumps({
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "gpu_count": torch.cuda.device_count(),
    "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
}, indent=2))
torch.ones(8, device="cuda").square().sum().cpu()
PY

evaluate_checkpoint() {
  local model_name=$1
  local checkpoint=$2
  local repo=$3
  local output_root=$4
  local gpu=$5

  for benchmark in rruff473 rruff325; do
    local eval_h5
    if [[ "${benchmark}" == "rruff473" ]]; then
      eval_h5=${rruff473}
    else
      eval_h5=${rruff325}
    fi
    local benchmark_dir=${output_root}/${benchmark}
    mkdir -p "${benchmark_dir}"

    CUDA_VISIBLE_DEVICES=${gpu} PYTHONPATH="${repo}" python \
      "${tools}/eval_xrdbench_sg_model_rruff_common_grid.py" \
      --checkpoint "${checkpoint}" \
      --eval-h5 "${eval_h5}" \
      --mapping-csv "${mapping}" \
      --output-json "${benchmark_dir}/eval.json" \
      --output-npz "${benchmark_dir}/posterior.npz" \
      --device cuda:0 \
      --batch-size 32 \
      --target-label-base 1 \
      --input-target-max 100

    python "${tools}/xrdbench_eval_to_failure_json.py" \
      --eval-json "${benchmark_dir}/eval.json" \
      --model-name "${model_name}" \
      --output-json "${benchmark_dir}/failure.json"

    python "${tools}/analyze_topological_error_distance.py" \
      --graph-json "${graph}" \
      --failure-json "${benchmark_dir}/failure.json" \
      --output-json "${benchmark_dir}/topology.json"

    python "${tools}/bootstrap_topological_error_distance.py" \
      --graph-json "${graph}" \
      --failure-json "${benchmark_dir}/failure.json" \
      --bootstrap 2000 \
      --output-json "${benchmark_dir}/topology_bootstrap.json"
  done
}

run_model() {
  local model=$1
  local gpu=$2
  local model_root=${run_group}/${model}
  local repo=${model_root}/repo

  mkdir "${model_root}"
  cp -a "${source_repo}" "${repo}"
  mkdir -p "${repo}/checkpoints"

  (
    set -euo pipefail
    cd "${repo}"
    export XRDBENCH_TRAIN_SHARD_MANIFEST="${train_manifest}"
    export XRDBENCH_VAL_SHARD_MANIFEST="${val_manifest}"
    export XRDBENCH_TEST_SHARD_MANIFEST="${test_manifest}"
    export XRDBENCH_CHECKPOINT_EVERY_STEPS=0
    export CUDA_VISIBLE_DEVICES=${gpu}

    python main.py \
      --model "${model}" \
      --task spg \
      --epochs 10 \
      --batch_size 128 \
      --num_workers 4 \
      --patience 3 \
      --seed 100 \
      --datapath $SCRATCH/simxrd_public_xrdbench \
      --gpu 0

    mapfile -t checkpoints < <(find checkpoints -mindepth 2 -maxdepth 2 -type f -name checkpoint.pth)
    if [[ ${#checkpoints[@]} -ne 1 ]]; then
      printf 'Expected one selected checkpoint for %s, found %s\n' "${model}" "${#checkpoints[@]}" >&2
      exit 1
    fi
    checkpoint=$(realpath "${checkpoints[0]}")
    printf '%s\n' "${checkpoint}" > "${model_root}/selected_checkpoint.txt"
    evaluate_checkpoint \
      "xrdbench_${model}_common_grid" \
      "${checkpoint}" \
      "${repo}" \
      "${model_root}/evaluation" \
      "${gpu}"
  ) >"${model_root}/run.log" 2>"${model_root}/run.err"
}

run_model cnn10 0 &
pid_cnn=$!
run_model bilstm 1 &
pid_rnn=$!
run_model PatchTST 2 &
pid_attention=$!

status=0
for pid in "${pid_cnn}" "${pid_rnn}" "${pid_attention}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [[ ${status} -ne 0 ]]; then
  printf 'At least one new-model run failed; preserving all partial outputs in %s\n' "${run_group}" >&2
  exit ${status}
fi

original_root=${run_group}/original_models_common_grid
mkdir "${original_root}"
original_repo=${original_root}/repo
cp -a "${source_repo}" "${original_repo}"

evaluate_checkpoint \
  xrdbench_cnn11_common_grid \
  "$SCRATCH/XRDBench/checkpoints/spg-cnn11_lr0.00025_bs128_2026-05-31 16:55:45_100/checkpoint.pth" \
  "${original_repo}" \
  "${original_root}/cnn11" \
  0
evaluate_checkpoint \
  xrdbench_bigru_common_grid \
  "$SCRATCH/XRDBench/checkpoints/spg-bigru_lr0.00025_bs128_2026-05-31 16:55:44_100/checkpoint.pth" \
  "${original_repo}" \
  "${original_root}/bigru" \
  0
evaluate_checkpoint \
  xrdbench_transformer_common_grid \
  "$SCRATCH/XRDBench/checkpoints/spg-transformer_lr0.00025_bs128_2026-05-31 16:55:44_100/checkpoint.pth" \
  "${original_repo}" \
  "${original_root}/transformer" \
  0

python - "${run_group}" <<'PY'
import hashlib
import json
import pathlib
import sys

run_group = pathlib.Path(sys.argv[1])
files = sorted(path for path in run_group.rglob("*") if path.is_file())
manifest = {
    "run_group": str(run_group),
    "files": [
        {
            "path": str(path.relative_to(run_group)),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in files
    ],
}
(run_group / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
PY
