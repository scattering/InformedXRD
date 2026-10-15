#!/bin/bash
#SBATCH -J gen1pat
#SBATCH -p spr
#SBATCH -A PHY26002
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 00:20:00
#SBATCH -o /home1/ALLOC3/UserE/gen1pat.%j.out
#SBATCH -e /home1/ALLOC3/UserE/gen1pat.%j.err

set -euo pipefail

WORKDIR="$HOME/project-generation-smoke"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

if [ ! -d "$HOME/codex/pycrysfml-test/src/pycrysfml" ]; then
  echo "Missing existing pycrysfml-test build tree"
  exit 2
fi

module purge
module load gcc/13.2.0
module load python/3.12.11

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install numpy spglib pyxtal pymatgen

export PYCRYSFML_LIB="$HOME/codex/pycrysfml-test/src/pycrysfml/libcrysfml.so"
export PYCRYSFML_MODS="$HOME/codex/pycrysfml-test/src/pycrysfml"
export PYTHONPATH="$HOME/codex/pycrysfml-test/src:$PYTHONPATH"

python "$WORKDIR/test_generate_one_pattern.py"
