#!/bin/bash
#SBATCH -J pycrysfml_spr_build
#SBATCH -o pycrysfml_spr_build.%j.out
#SBATCH -e pycrysfml_spr_build.%j.err
#SBATCH -p spr
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 02:00:00
#SBATCH -A PHY26002

set -euo pipefail

echo "== Job context =="
hostname
date
whoami
pwd

WORKROOT="${WORKROOT:-$SCRATCH/pycrysfml_spr}"
REPO_ROOT="${REPO_ROOT:-$WORKROOT/crysfml}"
VENV_DIR="${VENV_DIR:-$WORKROOT/venv}"
FULLPROF_URL="${FULLPROF_URL:-https://www.ill.eu/sites/fullprof/downloads/FullProf_Suite_March2026_Linux64_ifx.tgz}"
FULLPROF_TARBALL="${FULLPROF_TARBALL:-$WORKROOT/$(basename "$FULLPROF_URL")}"
FULLPROF_DIR="${FULLPROF_DIR:-$WORKROOT/fullprof}"
CRYSFML_REMOTE="${CRYSFML_REMOTE:-https://code.ill.fr/scientific-software/crysfml.git}"

mkdir -p "$WORKROOT" "$FULLPROF_DIR" "$HOME/.ssh"

echo "== Modules =="
module purge
module load gcc/13.2.0
module load cmake/4.1.1
module load python/3.12.11
module list

echo "== Toolchain =="
which python3
python3 --version
which gcc
gcc --version | head -n 1
which gfortran
gfortran --version | head -n 1
which cmake
cmake --version | head -n 2

echo "== Python env =="
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install numpy meson ninja scikit-build-core f90wrap pybind11 spglib pyxtal pymatgen
python -m pip list

echo "== ILL host key =="
ssh-keyscan -H code.ill.fr >> "$HOME/.ssh/known_hosts" 2>/dev/null || true

echo "== Clone CrysFML =="
rm -rf "$REPO_ROOT.tmp"
if git clone "$CRYSFML_REMOTE" "$REPO_ROOT.tmp"; then
  rm -rf "$REPO_ROOT"
  mv "$REPO_ROOT.tmp" "$REPO_ROOT"
else
  echo "CrysFML clone failed"
  exit 10
fi

echo "== Repo snapshot =="
find "$REPO_ROOT" -maxdepth 3 -type f | sed -n '1,200p'

echo "== FullProf =="
if [ ! -f "$FULLPROF_TARBALL" ]; then
  curl -L "$FULLPROF_URL" -o "$FULLPROF_TARBALL"
fi
tar -xzf "$FULLPROF_TARBALL" -C "$FULLPROF_DIR" --strip-components=1 || tar -xzf "$FULLPROF_TARBALL" -C "$FULLPROF_DIR"
find "$FULLPROF_DIR" -maxdepth 2 -type d | sed -n '1,80p'
export FULLPROF="$FULLPROF_DIR"
echo "FULLPROF=$FULLPROF"

echo "== Build/install attempts =="
cd "$REPO_ROOT"

if [ -f pyproject.toml ]; then
  python -m pip install -v .
elif [ -f setup.py ]; then
  python -m pip install -v .
elif [ -d Python_API ] && [ -f Python_API/pyproject.toml ]; then
  cd Python_API
  python -m pip install -v .
elif [ -d Python_API ] && [ -f Python_API/setup.py ]; then
  cd Python_API
  python -m pip install -v .
else
  echo "No obvious Python build entrypoint found"
  find . -maxdepth 4 \( -name pyproject.toml -o -name setup.py -o -name CMakeLists.txt -o -name README.md \) | sed -n '1,200p'
  exit 20
fi

echo "== Smoke test =="
python - <<'PY'
import os
print("FULLPROF", os.environ.get("FULLPROF"))
import pycrysfml
from pycrysfml import cfml_py_utilities
print("pycrysfml", pycrysfml.__file__)
print("cfml_py_utilities", cfml_py_utilities)
PY

echo "== Done =="
