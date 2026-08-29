# Install

Install requirements.txt

# Configure environment and dependencies

Recommended environment:

```
export WORKROOT=/work/ALLOC3/UserE/shared/pycrysfml_spr
export REPO_ROOT=$WORKROOT/crysfml
export FULLPROF=$WORKROOT/fullprof
export FULLPROF_DIR=$WORKROOT/fullprof
export EXT_MAP_CSV=$WORKROOT/FINAL_SPG_ExtG_CrysS_Table.csv

# Empirical priors for RRUFF-conditioned generation (Multilabel branch port)
export RRUFF_PRIOR_CSV=/work2/ALLOC3/UserE/rruff-benchmark/option1_metadata_qc_full.csv
export ICSD_CHEMISTRY_CSV=/work2/ALLOC3/UserE/reference_data/ICSD_With_Bravais.csv
```

`xrd_generator.py` and `train_streaming.sh` default to the paths above when these variables are unset. Override any of them individually as needed.

Optional generation tuning variables (all have defaults):
- `EG_SAMPLING_MODE` — `rruff` (default) or `uniform`
- `PRIOR_MODE` — `robust` (default) or `clean` (stricter RRUFF row filter)
- `SAMPLE_RETRY_LIMIT` — int, default 3
- `FAILURE_LOG_LIMIT` — int, default 200 (0 = disabled); writes to `FAILURE_LOG_PATH`
- `USE_EXACT_GP` — `0` (default RFF) or `1` (exact GP, slower)
- `CHEMISTRY_MARGINAL_TEMPER`, `CHEMISTRY_PAIR_TEMPER`, `CHEMISTRY_SMOOTH` — chemistry prior temperature controls
If you want your own venv:
```
python3 -m venv $SCRATCH/my_pycrysfml_env
source $SCRATCH/my_pycrysfml_env/bin/activate
pip install --upgrade pip setuptools wheel
pip install numpy meson ninja scikit-build-core f90wrap pybind11 spglib pyxtal pymatgen
cd $REPO_ROOT
pip install -v .
```

# Ensure correct modules are loaded (Python 3.12)

Currently Loaded Modules:
```
  1) ucc/1.6.0   2) ucx/1.20.0   3) cuda/12.4 (g)   4) nvidia/24.5   5) python/3.12.11   6) openmpi/5.0.8
```