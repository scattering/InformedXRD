#!/bin/bash
set -euo pipefail

cd /scratch/ALLOC3/UserE/project/Code/ViT_NVIDIA

ABLATION_PREFIX="${ABLATION_PREFIX:-splitablate_v2}"
MODES="${MODES:-on off}"
SEEDS="${SEEDS:-1337 2027 31415}"

for mode in $MODES; do
  for seed in $SEEDS; do
    stage1_cfg="/scratch/ALLOC3/UserE/project/Code/ViT_NVIDIA/config_uniform_premaint_1381k_splitablation_${mode}.json"
    stage2_cfg="/scratch/ALLOC3/UserE/project/Code/ViT_NVIDIA/config_rruff_conditioned_2346k_splitablation_${mode}.json"
    tag="${ABLATION_PREFIX}_${mode}_s${seed}"

    stage1_out="/scratch/ALLOC3/UserE/${tag}_stage1_config.json"
    stage1_meta="/scratch/ALLOC3/UserE/${tag}_stage1_meta.json"
    stage2_out="/scratch/ALLOC3/UserE/${tag}_stage2_config.json"
    stage2_meta="/scratch/ALLOC3/UserE/${tag}_stage2_meta.json"

    stage1_job="$(sbatch --parsable \
      --job-name "${tag}_u1381k" \
      --export=ALL,BASE_CONFIG="$stage1_cfg",OUTPUT_CONFIG="$stage1_out",META_OUT="$stage1_meta",JOB_TAG="${tag}_u1381k",SEED="$seed" \
      vista_split_ablation_train_with_meta.sh | tail -n 1)"

    stage2_job="$(sbatch --parsable \
      --dependency=afterok:${stage1_job} \
      --job-name "${tag}_r2346k" \
      --export=ALL,BASE_CONFIG="$stage2_cfg",OUTPUT_CONFIG="$stage2_out",META_OUT="$stage2_meta",JOB_TAG="${tag}_r2346k",SEED="$seed",RESUME_META="$stage1_meta" \
      vista_split_ablation_train_with_meta.sh | tail -n 1)"

    eval_job="$(sbatch --parsable \
      --dependency=afterok:${stage2_job} \
      --job-name "${tag}_eval" \
      --export=ALL,META_IN="$stage2_meta" \
      vista_split_ablation_eval_from_meta.sh | tail -n 1)"

    echo "${tag}: stage1=${stage1_job} stage2=${stage2_job} eval=${eval_job}"
  done
done
