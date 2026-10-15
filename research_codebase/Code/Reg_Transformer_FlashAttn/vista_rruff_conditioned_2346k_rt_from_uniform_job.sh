#!/bin/bash
#SBATCH -J vista-rt-r2346
#SBATCH -A CDA24014
#SBATCH -p gh
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH -t 06:00:00
#SBATCH -o /scratch/ALLOC3/UserE/vista_rt_rruff2346_%j.out
#SBATCH -e /scratch/ALLOC3/UserE/vista_rt_rruff2346_%j.err

set -euo pipefail

: "${UNIFORM_JOB_ID:?Set UNIFORM_JOB_ID to the uniform-pretrain Slurm job id before submitting.}"

module load gcc/13.2.0 cuda/12.5 python3/3.11.8
source /scratch/ALLOC3/UserE/project-venv/bin/activate

export WANDB_API_KEY
WANDB_API_KEY="$(cat /scratch/ALLOC3/UserE/project/.wandb_api_key)"

cd /scratch/ALLOC3/UserE/project/Code/Reg_Transformer_FlashAttn

RT_UNIFORM_RUN_ID=$(
python - <<'PY'
import re
from pathlib import Path
import os

jobid = os.environ["UNIFORM_JOB_ID"]
paths = [
    Path(f"/scratch/ALLOC3/UserE/vista_rt_uniform_2m_{jobid}.err"),
    Path(f"/scratch/ALLOC3/UserE/vista_rt_uniform_2m_{jobid}.out"),
]
text = ""
for path in paths:
    if path.exists():
        text += path.read_text(errors="ignore") + "\n"

match = re.search(r"/runs/([a-z0-9]+)", text)
if match is None:
    match = re.search(r"run-[0-9_]+-([a-z0-9]+)", text)
if match is None:
    raise SystemExit("Could not determine RT uniform run id from logs")
print(match.group(1))
PY
)

export RT_UNIFORM_RUN_ID

CFG_PATH=$(
python - <<'PY'
import json
import os
from pathlib import Path

src = Path("config_rruff_conditioned_2346k_from_rt_uniform_physpe_coord.json")
cfg = json.loads(src.read_text())
cfg["resume_checkpoint"] = f"/scratch/ALLOC3/UserE/ai_diffraction_models/xrd_model_{os.environ['RT_UNIFORM_RUN_ID']}_best.pth"
tmp = Path("/tmp") / f"config_rruff_conditioned_2346k_from_rt_uniform_{os.environ['RT_UNIFORM_RUN_ID']}.json"
tmp.write_text(json.dumps(cfg, indent=2))
print(tmp)
PY
)

python - <<'PY'
import os
import wandb
wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)
PY

python train_multilabel.py --config "$CFG_PATH"
