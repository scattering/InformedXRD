#!/bin/bash
#SBATCH -J r209-mp-overlap
#SBATCH -A PHY26002
#SBATCH -p spr
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -o /work2/ALLOC3/UserE/logs/r209_mp_overlap_%j.out
#SBATCH -e /work2/ALLOC3/UserE/logs/r209_mp_overlap_%j.err

set -euo pipefail

SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}
REPO=${REPO:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}
WORKROOT=${WORKROOT:-/work2/ALLOC3/UserE/rruff209_mp_overlap}
CASES_JSON=${CASES_JSON:-${WORKROOT}/rruff209_cases.json}
MP_JSONL=${MP_JSONL:-/work2/ALLOC3/UserE/stampede3/reference_data/mp_theoretical_candidates_20260427.jsonl}
PROCESSED_DIR=${PROCESSED_DIR:-${WORKROOT}/processed_xy}
ENV_PY=${ENV_PY:-/work2/ALLOC3/UserE/stampede3/conda/envs/ansatz/bin/python}
NCDIST_DIR=${NCDIST_DIR:-${WORKROOT}/ncdist}
SELLING_DIR=${SELLING_DIR:-${SCRIPT_DIR}}
OUT_JSON=${OUT_JSON:-${WORKROOT}/rruff209_mp_overlap_results.json}
MAX_CASES=${MAX_CASES:-0}
SKIP_BUILD=${SKIP_BUILD:-1}

mkdir -p "${WORKROOT}" "$(dirname "${OUT_JSON}")" /work2/ALLOC3/UserE/logs

module load gcc/13.2.0 cmake

if [ ! -x "${ENV_PY}" ]; then
  echo "missing Python interpreter: ${ENV_PY}" >&2
  exit 1
fi
if ! "${ENV_PY}" - <<'PY'
import pybind11
print(pybind11.get_cmake_dir())
PY
then
  "${ENV_PY}" -m pip install --user pybind11
fi

if [ ! -d "${NCDIST_DIR}" ]; then
  git clone https://github.com/yayahjb/ncdist.git "${NCDIST_DIR}"
fi

pushd "${SELLING_DIR}" >/dev/null
if [ "${SKIP_BUILD}" != "1" ] || ! ls s6dist*.so >/dev/null 2>&1; then
  cmake -S . -B build \
    -DNCDIST_DIR="${NCDIST_DIR}" \
    -Dpybind11_DIR="$("${ENV_PY}" -c 'import pybind11; print(pybind11.get_cmake_dir())')"
  cmake --build build -j
  cp build/s6dist*.so .
fi
popd >/dev/null

export PYTHONPATH="${SELLING_DIR}:${PYTHONPATH:-}"
"${ENV_PY}" "${SELLING_DIR}/run_rruff209_mp_overlap.py" \
  --cases-json "${CASES_JSON}" \
  --mp-jsonl "${MP_JSONL}" \
  --output-json "${OUT_JSON}" \
  --processed-dir "${PROCESSED_DIR}" \
  --max-cases "${MAX_CASES}"
