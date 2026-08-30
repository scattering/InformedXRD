#!/bin/bash
#SBATCH -J xrdbench-opxrd
#SBATCH -A <ALLOCATION>
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 00:15:00
#SBATCH -o $SCRATCH/xrdbench_opxrd_community_20260727/slurm.%j.out
#SBATCH -e $SCRATCH/xrdbench_opxrd_community_20260727/slurm.%j.err

set -euo pipefail

root=$SCRATCH
base=${XRDBENCH_OPXRD_BASE:-${root}/xrdbench_opxrd_community_20260727}
input=${base}/input/opxrd_cukalpha_10_80_maxnorm.npz
tools=${base}/tools
venv=$SCRATCH/xrdbench_clean_second_set_20260726/venv
source_repo=$SCRATCH/xrdbench_clean_second_set_20260726/source
clean_run=$SCRATCH/xrdbench_clean_second_set_20260726/runs/job_867777
run_group=${base}/runs/job_${SLURM_JOB_ID}
mapping=${tools}/canonical_extinction_to_space_group.csv
graph=${tools}/extinction_group_adjacency.json
train_manifest=${root}/simxrd_train1_pt_shards/manifest.json
val_manifest=${root}/simxrd_val_pt_shards/manifest.json
test_manifest=${root}/simxrd_test_pt_shards/manifest.json

module purge
module load gcc/13.2.0 cuda/12.8 python3/3.11.8
source "${venv}/bin/activate"
export OMP_NUM_THREADS=4

mkdir "${run_group}"
nvidia-smi -L
python - <<'PY'
import json
import torch

assert torch.cuda.is_available(), "CUDA is not available"
print(json.dumps({
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "gpu_count": torch.cuda.device_count(),
    "gpu": torch.cuda.get_device_name(0),
}, indent=2))
torch.ones(8, device="cuda").square().sum().cpu()
PY

python - \
  "${run_group}/provenance.json" \
  "${input}" \
  "${mapping}" \
  "${train_manifest}" \
  "${val_manifest}" \
  "${test_manifest}" <<'PY'
import hashlib
import json
import pathlib
import sys

output, *raw_paths = sys.argv[1:]
paths = [pathlib.Path(path) for path in raw_paths]
for path in paths:
    if not path.is_file():
        raise FileNotFoundError(path)
payload = {
    "training_contract": {
        "source": "SimXRD public train_1 split",
        "target": "230 space groups",
        "two_theta_min": 10.0,
        "two_theta_max": 80.0,
        "two_theta_step": 0.02,
        "sequence_length": 3501,
        "nominal_intensity_max": 100.0,
        "wavelength_angstrom": 1.54184,
    },
    "files": {
        str(path): {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in paths
    },
}
pathlib.Path(output).write_text(json.dumps(payload, indent=2) + "\n")
PY

evaluate_model() {
  local model=$1
  local checkpoint=$2
  local repo=$3
  local output_dir=${run_group}/${model}

  mkdir "${output_dir}"
  printf '%s\n' "${checkpoint}" > "${output_dir}/checkpoint_path.txt"

  PYTHONPATH="${repo}:${tools}" python \
    "${tools}/eval_xrdbench_sg_model_opxrd.py" \
    --checkpoint "${checkpoint}" \
    --eval-npz "${input}" \
    --mapping-csv "${mapping}" \
    --output-json "${output_dir}/eval.json" \
    --output-npz "${output_dir}/posterior.npz" \
    --device cuda:0 \
    --batch-size 32 \
    --grid-min 10 \
    --grid-max 80 \
    --grid-step 0.02 \
    --input-target-max 100

  python "${tools}/xrdbench_eval_to_failure_json.py" \
    --eval-json "${output_dir}/eval.json" \
    --model-name "xrdbench_${model}_opxrd10_80" \
    --output-json "${output_dir}/failure.json"

  PYTHONPATH="${tools}" python \
    "${tools}/analyze_topological_error_distance.py" \
    --graph-json "${graph}" \
    --failure-json "${output_dir}/failure.json" \
    --output-json "${output_dir}/topology.json"

  PYTHONPATH="${tools}" python \
    "${tools}/bootstrap_topological_error_distance.py" \
    --graph-json "${graph}" \
    --failure-json "${output_dir}/failure.json" \
    --bootstrap 2000 \
    --output-json "${output_dir}/topology_bootstrap.json"

  PYTHONPATH="${tools}" python \
    "${tools}/analyze_topological_null.py" \
    --graph-json "${graph}" \
    --failure-json "${output_dir}/failure.json" \
    --condition-on-model-errors \
    --output-json "${output_dir}/topology_error_conditioned_null.json"
}

evaluate_model \
  cnn10 \
  "${clean_run}/cnn10/repo/checkpoints/spg-cnn10_lr0.00025_bs128_2026-07-26 14:08:41_100/checkpoint.pth" \
  "${clean_run}/cnn10/repo"
evaluate_model \
  cnn11 \
  "${root}/XRDBench/checkpoints/spg-cnn11_lr0.00025_bs128_2026-05-31 16:55:45_100/checkpoint.pth" \
  "${source_repo}"
evaluate_model \
  bilstm \
  "${clean_run}/bilstm/repo/checkpoints/spg-bilstm_lr0.00025_bs128_2026-07-26 14:08:41_100/checkpoint.pth" \
  "${clean_run}/bilstm/repo"
evaluate_model \
  bigru \
  "${root}/XRDBench/checkpoints/spg-bigru_lr0.00025_bs128_2026-05-31 16:55:44_100/checkpoint.pth" \
  "${source_repo}"
evaluate_model \
  transformer \
  "${root}/XRDBench/checkpoints/spg-transformer_lr0.00025_bs128_2026-05-31 16:55:44_100/checkpoint.pth" \
  "${source_repo}"
evaluate_model \
  PatchTST \
  "${clean_run}/PatchTST/repo/checkpoints/spg-PatchTST_lr0.00025_bs128_2026-07-26 14:08:41_100/checkpoint.pth" \
  "${clean_run}/PatchTST/repo"

PYTHONPATH="${tools}" python \
  "${tools}/analyze_topological_null.py" \
  --graph-json "${graph}" \
  --failure-json "${run_group}/cnn10/failure.json" \
  --output-json "${run_group}/opxrd10_80_uniform_wrong_null.json"

python - "${run_group}" <<'PY'
import hashlib
import json
import pathlib
import sys

run_group = pathlib.Path(sys.argv[1])
rows = []
for path in sorted(run_group.glob("*/eval.json")):
    data = json.loads(path.read_text())
    topology = json.loads((path.parent / "topology.json").read_text())
    model_key = next(iter(topology["models"]))
    top = topology["models"][model_key]
    rows.append({
        "model": path.parent.name,
        "n": data["n"],
        "top1_sg": data["top1_sg"],
        "top3_sg": data["top3_sg"],
        "top5_sg": data["top5_sg"],
        "top1_eg": data["top1_eg"],
        "top3_eg": data["top3_eg"],
        "top5_eg": data["top5_eg"],
        "wrong_pct_distance_le2": top["pct_wrong_le2"],
        "mean_wrong_distance": top["mean_wrong_dist"],
        "checkpoint_sha256": data["checkpoint_sha256"],
        "eval_npz_sha256": data["eval_npz_sha256"],
    })
(run_group / "summary.json").write_text(
    json.dumps({"models": rows}, indent=2) + "\n"
)

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
print(json.dumps({"models": rows}, indent=2))
PY
