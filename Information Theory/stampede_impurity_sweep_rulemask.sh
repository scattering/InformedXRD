#!/bin/bash
#SBATCH -J it_impurity
#SBATCH -A PHY26002
#SBATCH -p spr
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -o /work2/ALLOC3/UserE/logs/%x_%j.out
#SBATCH -e /work2/ALLOC3/UserE/logs/%x_%j.err

set -euo pipefail

ROOT="${ROOT:-/scratch/ALLOC3/UserE/project}"
NPZ="${NPZ:-/work2/ALLOC3/UserE/information_theory_outputs/gate3_rule_mask_ladder_s100.npz}"
RUNG="${RUNG:-background_noise_like}"
RATES="${RATES:-0,0.02,0.05,0.1,0.2}"
WEIGHTING="${WEIGHTING:-inverse_norm}"
NBOOT="${NBOOT:-200}"
OUT_JSON="${OUT_JSON:-/work2/ALLOC3/UserE/information_theory_outputs/impurity_sweep_rulemask_${RUNG}.json}"
ALL_RUNGS="${ALL_RUNGS:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p /work2/ALLOC3/UserE/logs

cd "${ROOT}"

python3 "Information Theory/impurity_sweep_rulemask.py" \
  --npz "${NPZ}" \
  --rates "${RATES}" \
  --weighting "${WEIGHTING}" \
  --heldout-boot "${NBOOT}" \
  --out-json "${OUT_JSON}" \
  $([ "${ALL_RUNGS}" = "1" ] && printf '%s' "--all-rungs" || printf '%s' "--rung ${RUNG}") \
  ${EXTRA_ARGS}
