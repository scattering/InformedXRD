#!/bin/bash
#SBATCH -J vista-u2m-scale-e
#SBATCH -A CDA24014
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -t 01:00:00
#SBATCH -o /scratch/ALLOC3/UserE/vista_uniform_2m_scaling_eval_%j.out
#SBATCH -e /scratch/ALLOC3/UserE/vista_uniform_2m_scaling_eval_%j.err

set -euo pipefail

: "${UNIFORM_SG_JOB_ID:?Set UNIFORM_SG_JOB_ID to the SG scaling training job id.}"
: "${UNIFORM_EG_JOB_ID:?Set UNIFORM_EG_JOB_ID to the EG scaling training job id.}"
: "${SCALE_FRACTION:?Set SCALE_FRACTION to the same fraction label used in training.}"

module load gcc/13.2.0 cuda/12.5 python3/3.11.8
source /scratch/ALLOC3/UserE/project-venv/bin/activate

cd /scratch/ALLOC3/UserE/project/Code/ViT_NVIDIA

case "$SCALE_FRACTION" in
  1) TRAIN_SAMPLES=1600000 ;;
  1_2) TRAIN_SAMPLES=800000 ;;
  1_4) TRAIN_SAMPLES=400000 ;;
  1_8) TRAIN_SAMPLES=200000 ;;
  *) echo "Unsupported SCALE_FRACTION=$SCALE_FRACTION" >&2; exit 1 ;;
esac

export TRAIN_SAMPLES
export SCALE_FRACTION

SG_CONFIG=$(
python - <<'PY'
import json
import os
from pathlib import Path

template = Path("config_uniform_2m_blackbird_scaling_sg_template.json")
cfg = json.loads(template.read_text())
cfg["max_samples_train"] = int(os.environ["TRAIN_SAMPLES"])
cfg["scaling_mode"] = "sg"
cfg["scaling_fraction"] = os.environ["SCALE_FRACTION"]
tmp = Path("/tmp") / f"config_uniform_2m_sg_{os.environ['SCALE_FRACTION']}.json"
tmp.write_text(json.dumps(cfg, indent=2))
print(tmp)
PY
)

EG_CONFIG=$(
python - <<'PY'
import json
import os
from pathlib import Path

template = Path("config_uniform_2m_blackbird_scaling_eg_template.json")
cfg = json.loads(template.read_text())
cfg["max_samples_train"] = int(os.environ["TRAIN_SAMPLES"])
cfg["scaling_mode"] = "eg"
cfg["scaling_fraction"] = os.environ["SCALE_FRACTION"]
tmp = Path("/tmp") / f"config_uniform_2m_eg_{os.environ['SCALE_FRACTION']}.json"
tmp.write_text(json.dumps(cfg, indent=2))
print(tmp)
PY
)

SG_RUN_ID=$(
TARGET_JOB_ID="${UNIFORM_SG_JOB_ID}" python - <<'PY'
import re
from pathlib import Path
import os

jobid = os.environ["TARGET_JOB_ID"]
paths = [
    Path(f"/scratch/ALLOC3/UserE/vista_uniform_2m_scaling_{jobid}.err"),
    Path(f"/scratch/ALLOC3/UserE/vista_uniform_2m_scaling_{jobid}.out"),
]
text = ""
for path in paths:
    if path.exists():
        text += path.read_text(errors="ignore") + "\n"
match = re.search(r"/runs/([a-z0-9]+)", text)
if match is None:
    match = re.search(r"run-[0-9_]+-([a-z0-9]+)", text)
if match is None:
raise SystemExit("Could not determine SG scaling run id from logs")
print(match.group(1))
PY
)

EG_RUN_ID=$(
TARGET_JOB_ID="${UNIFORM_EG_JOB_ID}" python - <<'PY'
import re
from pathlib import Path
import os

jobid = os.environ["TARGET_JOB_ID"]
paths = [
    Path(f"/scratch/ALLOC3/UserE/vista_uniform_2m_scaling_{jobid}.err"),
    Path(f"/scratch/ALLOC3/UserE/vista_uniform_2m_scaling_{jobid}.out"),
]
text = ""
for path in paths:
    if path.exists():
        text += path.read_text(errors="ignore") + "\n"
match = re.search(r"/runs/([a-z0-9]+)", text)
if match is None:
    match = re.search(r"run-[0-9_]+-([a-z0-9]+)", text)
if match is None:
    raise SystemExit("Could not determine EG scaling run id from logs")
print(match.group(1))
PY
)

python evaluate_sg_to_eg_control.py \
  --sg-checkpoint "/scratch/ALLOC3/UserE/ai_diffraction_models/xrd_model_${SG_RUN_ID}_best.pth" \
  --sg-config "${SG_CONFIG}" \
  --sg-eval-data-path /scratch/ALLOC3/UserE/ai_diffraction_generated/uniform_2m_blackbird_20260401_sg_trainready.hdf5 \
  --eg-checkpoint "/scratch/ALLOC3/UserE/ai_diffraction_models/xrd_model_${EG_RUN_ID}_best.pth" \
  --eg-config "${EG_CONFIG}" \
  --eg-truth-data-path /scratch/ALLOC3/UserE/ai_diffraction_generated/uniform_2m_blackbird_20260401_trainready.hdf5 \
  --canonical-table-path /scratch/ALLOC3/UserE/project/Code/Post_Processing/canonical_extinction_to_space_group.csv \
  --output-json "/scratch/ALLOC3/UserE/sg_to_eg_control_scaling_${SCALE_FRACTION}_sg_${SG_RUN_ID}_eg_${EG_RUN_ID}.json" \
  --output-csv "/scratch/ALLOC3/UserE/sg_to_eg_control_scaling_${SCALE_FRACTION}_sg_${SG_RUN_ID}_eg_${EG_RUN_ID}_topk.csv"
