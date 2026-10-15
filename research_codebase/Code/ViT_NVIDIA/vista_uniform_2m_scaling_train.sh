#!/bin/bash
#SBATCH -J vista-u2m-scale
#SBATCH -A CDA24014
#SBATCH -p gh
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH -t 06:00:00
#SBATCH -o /scratch/ALLOC3/UserE/vista_uniform_2m_scaling_%j.out
#SBATCH -e /scratch/ALLOC3/UserE/vista_uniform_2m_scaling_%j.err

set -euo pipefail

: "${SCALE_MODE:?Set SCALE_MODE to sg or eg.}"
: "${SCALE_FRACTION:?Set SCALE_FRACTION to one of 1, 1_2, 1_4, 1_8.}"

module load gcc/13.2.0 cuda/12.5 python3/3.11.8
source /scratch/ALLOC3/UserE/project-venv/bin/activate

export WANDB_API_KEY
WANDB_API_KEY="$(cat /scratch/ALLOC3/UserE/project/.wandb_api_key)"

cd /scratch/ALLOC3/UserE/project/Code/ViT_NVIDIA

case "$SCALE_MODE" in
  sg) TEMPLATE="config_uniform_2m_blackbird_scaling_sg_template.json" ;;
  eg) TEMPLATE="config_uniform_2m_blackbird_scaling_eg_template.json" ;;
  *) echo "Unsupported SCALE_MODE=$SCALE_MODE" >&2; exit 1 ;;
esac

case "$SCALE_FRACTION" in
  1) TRAIN_SAMPLES=1600000 ;;
  1_2) TRAIN_SAMPLES=800000 ;;
  1_4) TRAIN_SAMPLES=400000 ;;
  1_8) TRAIN_SAMPLES=200000 ;;
  *) echo "Unsupported SCALE_FRACTION=$SCALE_FRACTION" >&2; exit 1 ;;
esac

export TEMPLATE
export TRAIN_SAMPLES
export SCALE_MODE
export SCALE_FRACTION

CFG_PATH=$(
python - <<'PY'
import json
import os
from pathlib import Path

template = Path(os.environ["TEMPLATE"])
cfg = json.loads(template.read_text())
cfg["max_samples_train"] = int(os.environ["TRAIN_SAMPLES"])
cfg["scaling_mode"] = os.environ["SCALE_MODE"]
cfg["scaling_fraction"] = os.environ["SCALE_FRACTION"]
tmp = Path("/tmp") / f"config_uniform_2m_{os.environ['SCALE_MODE']}_{os.environ['SCALE_FRACTION']}.json"
tmp.write_text(json.dumps(cfg, indent=2))
print(tmp)
PY
)

python - <<'PY'
import os
import wandb
wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)
PY

python train.py --config "$CFG_PATH"
