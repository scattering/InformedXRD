# Portable HPC Data Generation Checklist

This is the non-TACC version of the setup. It is the shortest list of things you need to reproduce synthetic diffraction generation on another HPC system.

## Goal

Get this working:

- generate synthetic XRD patterns with the repo generator
- support uniform extinction-group sampling
- support second phase / impurity augmentation
- write resumable shard HDF5 files

The generator is:

- [/Users/UserE/project/Code/Generation/gen_like_paper.py](/Users/UserE/project/Code/Generation/gen_like_paper.py)

## Minimum Software Stack

You need:

- Python 3.11 or 3.12
- a C/C++ compiler
- a Fortran compiler
- CMake or the build system required by the `crysfml` tree you use

Python packages needed:

- `numpy`
- `scipy`
- `pandas`
- `h5py`
- `tqdm`
- `ase`
- `pyxtal`
- `spglib`
- `pymatgen`

And one of:

- `pycrysfml`
- `CFML_api`

## External Dependencies

You need both:

1. ILL `crysfml`
2. `FullProf`

Why `FullProf` still matters:

- the repo generation path expects `FULLPROF`
- even if you end up using the `CFML_api` fallback, keeping the FullProf tables available avoids confusion and matches the working setup

## Files You Need From This Repo

At minimum, copy these:

- [/Users/UserE/project/Code/Generation/gen_like_paper.py](/Users/UserE/project/Code/Generation/gen_like_paper.py)
- [/Users/UserE/project/Code/Generation/tacc/cfml_api_single_pattern.py](/Users/UserE/project/Code/Generation/tacc/cfml_api_single_pattern.py)
- [/Users/UserE/project/Code/Generation/tacc/repack_uniform_shards_trainready.py](/Users/UserE/project/Code/Generation/tacc/repack_uniform_shards_trainready.py)
- [/Users/UserE/project/Code/Post_Processing/FINAL_SPG_ExtG_CrysS_Table.csv](/Users/UserE/project/Code/Post_Processing/FINAL_SPG_ExtG_CrysS_Table.csv)

If you want the exact launch patterns we used, also copy:

- [/Users/UserE/project/Code/Generation/tacc/run_gen_like_paper_uniform_spr.sh](/Users/UserE/project/Code/Generation/tacc/run_gen_like_paper_uniform_spr.sh)
- [/Users/UserE/project/Code/Generation/tacc/run_gen_like_paper_uniform_array_spr.sh](/Users/UserE/project/Code/Generation/tacc/run_gen_like_paper_uniform_array_spr.sh)

## Directory Layout That Works Well

Use separate directories for:

- repo source
- generated data
- external crystallography stack

For example:

```bash
$SCRATCH/project
$SCRATCH/ai_diffraction_generated
$SCRATCH/pycrysfml_build
```

Inside the crystallography stack directory, keep:

```bash
$SCRATCH/pycrysfml_build/venv
$SCRATCH/pycrysfml_build/crysfml
$SCRATCH/pycrysfml_build/fullprof
```

## Environment Variables To Set

The generator relies on environment variables.

Minimum useful set:

```bash
export FULLPROF=/path/to/fullprof
export EXT_MAP_CSV=/path/to/FINAL_SPG_ExtG_CrysS_Table.csv
export TOTAL_SAMPLES=4096
export EG_SAMPLING_MODE=uniform
export GENMAP_NUM_SHARDS=1
export GENMAP_SHARD_ID=0
export BATCH_ID=test
export BASE_OUTPUT_HDF5=/path/to/output/uniform_eg_impurity_shard
export NUM_PROCESSES_OVERRIDE=1
export RESUME_GENERATION=1
```

For large runs:

```bash
export TOTAL_SAMPLES=2000000
export GENMAP_NUM_SHARDS=100
export BATCH_ID=uniform2m_v1
export BASE_OUTPUT_HDF5=/path/to/output/uniform_eg_impurity_shard
```

## Build Strategy

Build on compute nodes, not login nodes.

The exact TACC script is:

- [/Users/UserE/project/Code/Generation/tacc/build_pycrysfml_spr.sh](/Users/UserE/project/Code/Generation/tacc/build_pycrysfml_spr.sh)

On another cluster, the same logic applies:

1. create a venv
2. install Python deps
3. clone `crysfml`
4. install/build the Python bindings
5. stage `FullProf`
6. run a smoke import

## Smoke Tests

Do these in order:

1. Confirm Python imports:

```python
import pyxtal, spglib, pymatgen, ase, h5py
```

2. Confirm crystallography backend:

```python
import pycrysfml
```

or

```python
import CFML_api
```

3. Confirm one isolated powder-pattern calculation:

- use [/Users/UserE/project/Code/Generation/tacc/cfml_api_single_pattern.py](/Users/UserE/project/Code/Generation/tacc/cfml_api_single_pattern.py)

4. Confirm one small generator run:

```bash
TOTAL_SAMPLES=32 GENMAP_NUM_SHARDS=1 GENMAP_SHARD_ID=0 NUM_PROCESSES_OVERRIDE=1 python gen_like_paper.py
```

Only scale after all four of those work.

## Important Stability Rule

Do not rely on repeated in-process `CFML_api` powder calls if you see allocator crashes.

What was stable for us:

- keep one high-level generator process per shard lane
- call the low-level powder calculation in a subprocess

That isolation is already implemented in:

- [/Users/UserE/project/Code/Generation/tacc/cfml_api_single_pattern.py](/Users/UserE/project/Code/Generation/tacc/cfml_api_single_pattern.py)

## Parallelism Strategy

Best pattern:

- shard-level parallelism across many jobs or tasks
- one Python generator process per shard lane
- avoid a large stateful local process pool

Use more shards, not more shared interpreter complexity.

Good starting point:

- `NUM_PROCESSES_OVERRIDE=1`
- several shard lanes per batch job
- many independent batch jobs if the scheduler allows it

## Checkpoint / Resume

This is mandatory for long runs.

The generator already writes progress into each shard:

- `tasks_processed`
- `successes_written`

Keep:

```bash
export RESUME_GENERATION=1
```

That allows:

- job resubmission after failure
- partial completion without losing a shard

## Uniform Sampling

For the multilabel / split-head target, use:

```bash
export EG_SAMPLING_MODE=uniform
```

This is the right choice if you want:

- full 99-EG coverage
- better rare-rule exposure
- less prior collapse than natural-mineral distributions

## Repacking

After shard generation, repack to a standard train-ready HDF5:

- `X_train`, `y_train`
- `X_val`, `y_val`
- `X_test`, `y_test`

Use:

- [/Users/UserE/project/Code/Generation/tacc/repack_uniform_shards_trainready.py](/Users/UserE/project/Code/Generation/tacc/repack_uniform_shards_trainready.py)

Typical invocation:

```bash
python repack_uniform_shards_trainready.py \
  --input-dir /path/to/shards \
  --output-h5 /path/to/trainready.hdf5 \
  --pattern "*.hdf5" \
  --train-percent 90 \
  --val-percent 5 \
  --split-mode order \
  --allow-partial \
  --min-written 1
```

## Failure Modes To Expect

1. `pycrysfml` imports, but the repo’s older wrapper API is missing
2. `CFML_api` works once, then repeated calls corrupt memory
3. HDF5 files are locked while active writers hold them
4. local multiprocessing looks faster but is less stable

If any of those appear:

- keep shard writes
- keep resume enabled
- isolate low-level calls
- repack from whatever readable shards you have

## Minimal Portable Recipe

If you only want the shortest recipe:

1. Install Python deps in a venv.
2. Build or install `crysfml` bindings on a compute node.
3. Install `FullProf` and set `FULLPROF`.
4. Run one single-pattern smoke test.
5. Run `gen_like_paper.py` with `EG_SAMPLING_MODE=uniform`, `NUM_PROCESSES_OVERRIDE=1`.
6. Scale out with shard jobs.
7. Repack shards into one train-ready HDF5.

That is the minimum path that should transfer to another HPC system with the fewest surprises.
