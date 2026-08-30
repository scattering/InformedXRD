#!/bin/bash
set -euo pipefail

cd $SCRATCH/ai-diffraction/Code/ViT_NVIDIA

fractions=(1_2 1_4 1_8)

extract_jobid() {
  grep -Eo 'Submitted batch job [0-9]+' | awk '{print $4}' | tail -n 1
}

for frac in "${fractions[@]}"; do
  echo "Submitting EG scaling run for fraction ${frac}"
  eg_submit=$(SCALE_MODE=eg SCALE_FRACTION="${frac}" sbatch vista_uniform_2m_scaling_train.sh)
  echo "${eg_submit}"
  eg_jobid=$(echo "${eg_submit}" | extract_jobid)

  echo "Submitting SG scaling run for fraction ${frac}"
  sg_submit=$(SCALE_MODE=sg SCALE_FRACTION="${frac}" sbatch vista_uniform_2m_scaling_train.sh)
  echo "${sg_submit}"
  sg_jobid=$(echo "${sg_submit}" | extract_jobid)

  echo "Submitting SG->EG collapse eval for fraction ${frac} after jobs ${sg_jobid} and ${eg_jobid}"
  UNIFORM_SG_JOB_ID="${sg_jobid}" UNIFORM_EG_JOB_ID="${eg_jobid}" SCALE_FRACTION="${frac}" \
    sbatch --dependency=afterok:"${sg_jobid}":"${eg_jobid}" vista_uniform_2m_scaling_sg_eval_from_job.sh
done
