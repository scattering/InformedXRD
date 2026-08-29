#!/bin/bash
#SBATCH -J repack-u2m-sg230
#SBATCH -A CDA24014
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -t 02:00:00
#SBATCH -o /scratch/ALLOC3/UserE/repack_uniform_2m_sg230_%j.out
#SBATCH -e /scratch/ALLOC3/UserE/repack_uniform_2m_sg230_%j.err

set -euo pipefail

module load gcc/13.2.0 python3/3.11.8
source /scratch/ALLOC3/UserE/repack_vista_gg_venv/bin/activate

input_dir=/work2/ALLOC3/UserE/uniform_2m_blackbird_20260401
output_h5=/scratch/ALLOC3/UserE/ai_diffraction_generated/uniform_2m_blackbird_20260401_sg_trainready.hdf5

mkdir -p "$(dirname "$output_h5")"
rm -f "$output_h5"

cd /scratch/ALLOC3/UserE/project/Code/Generation/tacc
python repack_shards_trainready_with_label.py \
  --input-dir "$input_dir" \
  --output-h5 "$output_h5" \
  --label-key space_group \
  --train-percent 80 \
  --val-percent 10 \
  --split-mode order

python - <<'PY'
import h5py

p = "/scratch/ALLOC3/UserE/ai_diffraction_generated/uniform_2m_blackbird_20260401_sg_trainready.hdf5"
with h5py.File(p, "r") as f:
    print("label_key", f.attrs.get("label_key"))
    for key in ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test"):
        if key not in f:
            raise SystemExit(f"missing dataset: {key}")
    print(
        "validated",
        int(f["X_train"].shape[0]),
        int(f["X_val"].shape[0]),
        int(f["X_test"].shape[0]),
        int(f["y_train"][:1000].min()),
        int(f["y_train"][:1000].max()),
    )
PY
