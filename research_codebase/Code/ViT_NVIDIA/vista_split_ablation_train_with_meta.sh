#!/bin/bash
#SBATCH -J split-ablate
#SBATCH -A CDA24014
#SBATCH -p gh
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH -t 08:00:00
#SBATCH -o /scratch/ALLOC3/UserE/split_ablation_%x_%j.out
#SBATCH -e /scratch/ALLOC3/UserE/split_ablation_%x_%j.err

set -euo pipefail

: "${BASE_CONFIG:?BASE_CONFIG must be set}"
: "${OUTPUT_CONFIG:?OUTPUT_CONFIG must be set}"
: "${META_OUT:?META_OUT must be set}"
: "${JOB_TAG:?JOB_TAG must be set}"
SEED="${SEED:-1337}"
RESUME_META="${RESUME_META:-}"
MODEL_DIR=/scratch/ALLOC3/UserE/ai_diffraction_models

module load gcc/13.2.0 cuda/12.5 python3/3.11.8
source /scratch/ALLOC3/UserE/project-venv/bin/activate

export WANDB_API_KEY
WANDB_API_KEY="$(cat /scratch/ALLOC3/UserE/project/.wandb_api_key)"

cd /scratch/ALLOC3/UserE/project/Code/ViT_NVIDIA

MARKER_FILE="/scratch/ALLOC3/UserE/${JOB_TAG}_${SLURM_JOB_ID}.marker"
rm -f "$MARKER_FILE"
touch "$MARKER_FILE"

export BASE_CONFIG OUTPUT_CONFIG META_OUT JOB_TAG SEED RESUME_META MARKER_FILE MODEL_DIR
python - <<'PY'
import json
import os
from pathlib import Path

base_config = Path(os.environ["BASE_CONFIG"])
output_config = Path(os.environ["OUTPUT_CONFIG"])
seed = int(os.environ["SEED"])
resume_meta = os.environ.get("RESUME_META", "")

with open(base_config, "r") as handle:
    cfg = json.load(handle)

cfg["seed"] = seed
cfg["run_tag"] = os.environ["JOB_TAG"]
if resume_meta:
    with open(resume_meta, "r") as handle:
        meta = json.load(handle)
    cfg["resume_checkpoint"] = meta["checkpoint"]
    cfg["resume_weights_only"] = True
else:
    cfg.pop("resume_checkpoint", None)
    cfg.pop("resume_weights_only", None)

output_config.parent.mkdir(parents=True, exist_ok=True)
with open(output_config, "w") as handle:
    json.dump(cfg, handle, indent=2)
print(output_config)
PY

python - <<'PY'
import os
import wandb
wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)
PY

python train.py --config "$OUTPUT_CONFIG"

LATEST_CHECKPOINT="$MODEL_DIR/xrd_model_${JOB_TAG}_best.pth"
if [ ! -s "$LATEST_CHECKPOINT" ]; then
  echo "Expected checkpoint was not written: $LATEST_CHECKPOINT" >&2
  exit 1
fi
export LATEST_CHECKPOINT

python - <<'PY'
import json
import os

with open(os.environ["OUTPUT_CONFIG"], "r") as handle:
    cfg = json.load(handle)

payload = {
    "train_job_id": os.environ["SLURM_JOB_ID"],
    "job_tag": os.environ["JOB_TAG"],
    "checkpoint": os.environ["LATEST_CHECKPOINT"],
    "config": os.environ["OUTPUT_CONFIG"],
    "seed": int(os.environ["SEED"]),
    "use_split_head": bool(cfg.get("use_split_head", False)),
    "prior_data_path": "/scratch/ALLOC3/UserE/ai_diffraction_generated/rruff_conditioned_2346k_v1_trainready.hdf5",
}
with open(os.environ["META_OUT"], "w") as handle:
    json.dump(payload, handle, indent=2)
print(json.dumps(payload, indent=2))
PY
