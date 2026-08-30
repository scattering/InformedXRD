#!/bin/bash

set -euo pipefail

base=${XRDBENCH_CLEAN_BASE:-$SCRATCH/xrdbench_clean_second_set_20260726}
script=${base}/tools/vista_xrdbench_clean_second_set_gb.sh

if [[ ! -x "${script}" ]]; then
  printf 'Missing staged submission script: %s\n' "${script}" >&2
  exit 1
fi

sbatch --parsable "${script}" | tail -n 1
