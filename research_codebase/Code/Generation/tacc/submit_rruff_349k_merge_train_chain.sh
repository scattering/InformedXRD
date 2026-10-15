#!/bin/bash
set -euo pipefail

STAMPede_HOST="${STAMPede_HOST:-stampede3}"
VISTA_HOST="${VISTA_HOST:-vista}"

REMOTE_REPO_STAMPede="${REMOTE_REPO_STAMPede:-/scratch/ALLOC3/UserE/project}"
REMOTE_REPO_VISTA="${REMOTE_REPO_VISTA:-/scratch/ALLOC3/UserE/project}"
STAMPede_VENV="${STAMPede_VENV:-/scratch/ALLOC3/UserE/pycrysfml_spr/venv}"

TRANSFER_DIR="${TRANSFER_DIR:-/work2/ALLOC3/UserE/rruff_cond_250k_20260330}"
TRANSFER_TRAINREADY_SCRATCH="${TRANSFER_TRAINREADY_SCRATCH:-/scratch/ALLOC3/UserE/ai_diffraction_generated/rruff_conditioned_249k6_external_trainready.hdf5}"
LOCAL_100K_TRAINREADY="${LOCAL_100K_TRAINREADY:-/scratch/ALLOC3/UserE/ai_diffraction_generated/rruff_conditioned_100k_v3_trainready_801010.hdf5}"
MERGED_TRAINREADY_SCRATCH="${MERGED_TRAINREADY_SCRATCH:-/scratch/ALLOC3/UserE/ai_diffraction_generated/rruff_conditioned_349k_v1_trainready.hdf5}"
MERGED_TRAINREADY_WORK="${MERGED_TRAINREADY_WORK:-/work2/ALLOC3/UserE/rruff_conditioned_349k_v1_trainready.hdf5}"
MERGED_TRAINREADY_VISTA_SCRATCH="${MERGED_TRAINREADY_VISTA_SCRATCH:-/scratch/ALLOC3/UserE/ai_diffraction_generated/rruff_conditioned_349k_v1_trainready.hdf5}"

extract_jobid() {
  awk '/^[0-9]+([_.-][0-9]+)?$/ {print $1}' | tail -n1
}

echo "[INFO] Syncing updated scripts/configs to Stampede and Vista"
scp /Users/UserE/project/Code/Generation/tacc/repack_shards_trainready_with_label.py \
  "${STAMPede_HOST}:${REMOTE_REPO_STAMPede}/Code/Generation/tacc/"
scp /Users/UserE/project/Code/Generation/tacc/merge_trainready_hdf5.py \
  "${STAMPede_HOST}:${REMOTE_REPO_STAMPede}/Code/Generation/tacc/"
scp /Users/UserE/project/Code/Generation/tacc/submit_rruff_349k_merge_train_chain.sh \
  "${STAMPede_HOST}:${REMOTE_REPO_STAMPede}/Code/Generation/tacc/" >/dev/null 2>&1 || true
scp /Users/UserE/project/Code/ViT_NVIDIA/train.py \
  /Users/UserE/project/Code/ViT_NVIDIA/config_rruff_conditioned_349k_from_wwuvp1kj_physpe_coord.json \
  /Users/UserE/project/Code/ViT_NVIDIA/vista_rruff_conditioned_349k_from_wwuvp1kj_physpe_coord.sh \
  "${VISTA_HOST}:${REMOTE_REPO_VISTA}/Code/ViT_NVIDIA/"

echo "[INFO] Submitting Stampede repack job"
REPACK_JOB="$(
ssh -q "${STAMPede_HOST}" "cat <<'EOS' > /tmp/repack_rruff_249k6_external.sh
#!/bin/bash
#SBATCH -J repack-r249
#SBATCH -A CDA24014
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 02:00:00
#SBATCH -o /scratch/ALLOC3/UserE/repack_r249_%j.out
#SBATCH -e /scratch/ALLOC3/UserE/repack_r249_%j.err
set -euo pipefail
module load python/3.12.11
source '${STAMPede_VENV}/bin/activate'
python '${REMOTE_REPO_STAMPede}/Code/Generation/tacc/repack_shards_trainready_with_label.py' \
  --input-dir '${TRANSFER_DIR}' \
  --output-h5 '${TRANSFER_TRAINREADY_SCRATCH}' \
  --label-key extinction_group \
  --split-mode order \
  --train-percent 80 \
  --val-percent 10 \
  --allow-partial
EOS
sbatch --parsable /tmp/repack_rruff_249k6_external.sh" | extract_jobid
)"
echo "[INFO] REPACK_JOB=${REPACK_JOB}"

echo "[INFO] Submitting Stampede merge job"
MERGE_JOB="$(
ssh -q "${STAMPede_HOST}" "cat <<'EOS' > /tmp/merge_rruff_349k_v1.sh
#!/bin/bash
#SBATCH -J merge-r349
#SBATCH -A CDA24014
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 02:00:00
#SBATCH -o /scratch/ALLOC3/UserE/merge_r349_%j.out
#SBATCH -e /scratch/ALLOC3/UserE/merge_r349_%j.err
set -euo pipefail
module load python/3.12.11
source '${STAMPede_VENV}/bin/activate'
python '${REMOTE_REPO_STAMPede}/Code/Generation/tacc/merge_trainready_hdf5.py' \
  --inputs '${LOCAL_100K_TRAINREADY}' '${TRANSFER_TRAINREADY_SCRATCH}' \
  --output '${MERGED_TRAINREADY_SCRATCH}'
EOS
sbatch --parsable --dependency=afterok:${REPACK_JOB} /tmp/merge_rruff_349k_v1.sh" | extract_jobid
)"
echo "[INFO] MERGE_JOB=${MERGE_JOB}"

echo "[INFO] Submitting Stampede copy-to-work2 job"
COPY_JOB="$(
ssh -q "${STAMPede_HOST}" "cat <<'EOS' > /tmp/copy_rruff_349k_v1_to_work.sh
#!/bin/bash
#SBATCH -J copy-r349
#SBATCH -A CDA24014
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -t 00:30:00
#SBATCH -o /scratch/ALLOC3/UserE/copy_r349_%j.out
#SBATCH -e /scratch/ALLOC3/UserE/copy_r349_%j.err
set -euo pipefail
cp '${MERGED_TRAINREADY_SCRATCH}' '${MERGED_TRAINREADY_WORK}'
ls -lh '${MERGED_TRAINREADY_WORK}'
EOS
sbatch --parsable --dependency=afterok:${MERGE_JOB} /tmp/copy_rruff_349k_v1_to_work.sh" | extract_jobid
)"
echo "[INFO] COPY_JOB=${COPY_JOB}"

echo "[INFO] Submitting Vista stage job"
STAGE_JOB="$(
ssh -q "${VISTA_HOST}" "cat <<'EOS' > /tmp/stage_rruff_349k_v1.sh
#!/bin/bash
#SBATCH -J stage-r349k
#SBATCH -A CDA24014
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -t 01:00:00
#SBATCH -o /scratch/ALLOC3/UserE/stage_rruff_349k_%j.out
#SBATCH -e /scratch/ALLOC3/UserE/stage_rruff_349k_%j.err
set -euo pipefail
module load gcc/13.2.0 python3/3.11.8
source /scratch/ALLOC3/UserE/project-venv/bin/activate
src='${MERGED_TRAINREADY_WORK}'
dst='${MERGED_TRAINREADY_VISTA_SCRATCH}'
tmp=\"${dst}.tmp\"
for _ in \$(seq 1 180); do
  if [ -f \"\$src\" ]; then
    break
  fi
  sleep 20
done
if [ ! -f \"\$src\" ]; then
  echo '[ERROR] Timed out waiting for merged train-ready file on /work2'
  exit 1
fi
rm -f \"\$tmp\" \"\$dst\"
cp \"\$src\" \"\$tmp\"
python - <<PY
import h5py, os
path = os.path.expanduser('${MERGED_TRAINREADY_VISTA_SCRATCH}.tmp')
with h5py.File(path, 'r') as h:
    print('train', h['X_train'].shape[0], 'val', h['X_val'].shape[0], 'test', h['X_test'].shape[0])
PY
mv \"\$tmp\" \"\$dst\"
ls -lh \"\$dst\"
EOS
sbatch --parsable /tmp/stage_rruff_349k_v1.sh" | extract_jobid
)"
echo "[INFO] STAGE_JOB=${STAGE_JOB}"

echo "[INFO] Submitting Vista training job"
TRAIN_JOB="$(
ssh -q "${VISTA_HOST}" "cd '${REMOTE_REPO_VISTA}/Code/ViT_NVIDIA' && sbatch --parsable --dependency=afterok:${STAGE_JOB} vista_rruff_conditioned_349k_from_wwuvp1kj_physpe_coord.sh" | extract_jobid
)"
echo "[INFO] TRAIN_JOB=${TRAIN_JOB}"

printf '\n[INFO] Submitted chain:\\n'
printf '  Stampede repack : %s\\n' "${REPACK_JOB}"
printf '  Stampede merge  : %s\\n' "${MERGE_JOB}"
printf '  Stampede copy   : %s\\n' "${COPY_JOB}"
printf '  Vista stage     : %s\\n' "${STAGE_JOB}"
printf '  Vista train     : %s\\n' "${TRAIN_JOB}"
