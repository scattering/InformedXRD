#!/bin/bash
#SBATCH -J split-ablate-eval
#SBATCH -A CDA24014
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00
#SBATCH -o /scratch/ALLOC3/UserE/split_ablation_eval_%j.out
#SBATCH -e /scratch/ALLOC3/UserE/split_ablation_eval_%j.err

set -euo pipefail

: "${META_IN:?META_IN must be set}"
TEMPS="${TEMPS:-5.0 7.5}"

module load gcc/13.2.0 cuda/12.5 python3/3.11.8
source /scratch/ALLOC3/UserE/project-venv/bin/activate
cd /scratch/ALLOC3/UserE/project/Code/ViT_NVIDIA

export META_IN TEMPS
ckpt="$(python - <<'PY'
import json, os
with open(os.environ["META_IN"], "r") as handle:
    print(json.load(handle)["checkpoint"])
PY
)"
cfg="$(python - <<'PY'
import json, os
with open(os.environ["META_IN"], "r") as handle:
    print(json.load(handle)["config"])
PY
)"
prior="$(python - <<'PY'
import json, os
with open(os.environ["META_IN"], "r") as handle:
    print(json.load(handle)["prior_data_path"])
PY
)"
tag="$(python - <<'PY'
import json, os
with open(os.environ["META_IN"], "r") as handle:
    print(json.load(handle)["job_tag"])
PY
)"
split_enabled="$(python - <<'PY'
import json, os
with open(os.environ["META_IN"], "r") as handle:
    print("1" if json.load(handle)["use_split_head"] else "0")
PY
)"

for temp in $TEMPS; do
  temp_tag="${temp//./p}"
  python evaluate_calibration_metrics.py \
    --checkpoint "$ckpt" \
    --config "$cfg" \
    --eval-data-path /work2/ALLOC3/UserE/rruff-benchmark/RRUFF_usable_plus_recoverable_325_with_labels_maxnorm.hdf5 \
    --prior-data-path "$prior" \
    --aux-temperature "$temp" \
    --bootstrap 1000 \
    --output-json "/scratch/ALLOC3/UserE/${tag}_r325_t${temp_tag}.json"
done

python evaluate_calibration_metrics.py \
  --checkpoint "$ckpt" \
  --config "$cfg" \
  --eval-data-path /work2/ALLOC3/UserE/rruff-benchmark/RRUFF_option1_473_with_buckets_maxnorm.hdf5 \
  --prior-data-path "$prior" \
  --aux-temperature 5.0 \
  --bootstrap 1000 \
  --output-json "/scratch/ALLOC3/UserE/${tag}_r473_t5p0.json"

if [ "$split_enabled" = "1" ]; then
  python evaluate_split_head_validity.py \
    --checkpoint "$ckpt" \
    --config "$cfg" \
    --eval-data-path /work2/ALLOC3/UserE/rruff-benchmark/RRUFF_usable_plus_recoverable_325_with_labels_maxnorm.hdf5 \
    --output-json "/scratch/ALLOC3/UserE/${tag}_split_validity_325.json"

  python evaluate_split_head_validity.py \
    --checkpoint "$ckpt" \
    --config "$cfg" \
    --eval-data-path /work2/ALLOC3/UserE/rruff-benchmark/RRUFF_option1_473_with_buckets_maxnorm.hdf5 \
    --output-json "/scratch/ALLOC3/UserE/${tag}_split_validity_473.json"
fi
