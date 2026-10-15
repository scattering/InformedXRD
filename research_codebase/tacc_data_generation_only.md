# Data Generation Setup On TACC

This note is only about synthetic diffraction generation. It does not cover ViT training.

The working path on TACC was:

- build the ILL `crysfml` Python stack on a compute node, not a login node
- keep `FullProf` available because the generation code expects `FULLPROF`
- use the repo's `Code/Generation/gen_like_paper.py`
- run generation on CPU nodes (`spr`)
- write shard HDF5 files with checkpoint/resume
- scale out across many shard-producing Slurm jobs

## What Actually Worked

The generator we ended up using is:

- [/Users/UserE/project/Code/Generation/gen_like_paper.py](/Users/UserE/project/Code/Generation/gen_like_paper.py)

This is the right script if you want the "paper-like" synthetic data:

- second phase / impurity mixing
- peak broadening / augmentation
- K-alpha splitting
- extinction-group aware labeling

We patched it so it now also supports:

- `EG_SAMPLING_MODE=uniform`
- shard checkpoint/resume
- per-sample retries
- subprocess-isolated `CFML_api` calls to avoid heap corruption

The key helper for that isolation is:

- [/Users/UserE/project/Code/Generation/tacc/cfml_api_single_pattern.py](/Users/UserE/project/Code/Generation/tacc/cfml_api_single_pattern.py)

## Why The Subprocess Helper Was Necessary

Repeated in-process `CFML_api` powder calls were not stable on `spr`. They eventually hit allocator errors.

What worked was:

- keep the main generator in Python
- but run each low-level `CFML_api` powder calculation in a fresh subprocess

That is slower per call, but it is robust enough for large-scale shard generation.

## Minimum Requirements

You need:

- a clone of this repo
- Python with:
  - `numpy`
  - `h5py`
  - `pandas`
  - `scipy`
  - `pyxtal`
  - `spglib`
  - `pymatgen`
  - `ase`
  - either `pycrysfml` or `CFML_api`
- a `FullProf` install
- access to:
  - [/Users/UserE/project/Code/Post_Processing/FINAL_SPG_ExtG_CrysS_Table.csv](/Users/UserE/project/Code/Post_Processing/FINAL_SPG_ExtG_CrysS_Table.csv)

## Build CrysFML On Compute Nodes

Do not build on a login node.

The working TACC build script is:

- [/Users/UserE/project/Code/Generation/tacc/build_pycrysfml_spr.sh](/Users/UserE/project/Code/Generation/tacc/build_pycrysfml_spr.sh)

What it does:

- creates a venv under `$SCRATCH/pycrysfml_spr/venv`
- clones ILL `crysfml`
- downloads FullProf
- builds/install the Python package
- runs a smoke import

Important detail:

- on TACC, HTTPS to `https://code.ill.fr/scientific-software/crysfml.git` was more reliable than SSH

Typical pattern:

```bash
cd /scratch/$USER/project/Code/Generation/tacc
sbatch build_pycrysfml_spr.sh
```

Artifacts expected afterward:

- venv:
  - `$SCRATCH/pycrysfml_spr/venv`
- FullProf:
  - `$SCRATCH/pycrysfml_spr/fullprof`
- built wheel / source tree:
  - `$SCRATCH/pycrysfml_spr/crysfml`

## Smoke Test The Powder Path

Before large generation, confirm that a single powder-pattern call works.

Useful files:

- [/Users/UserE/project/Code/Generation/tacc/test_generate_one_pattern.py](/Users/UserE/project/Code/Generation/tacc/test_generate_one_pattern.py)
- [/Users/UserE/project/Code/Generation/tacc/run_generate_one_pattern_spr.sh](/Users/UserE/project/Code/Generation/tacc/run_generate_one_pattern_spr.sh)

If `pycrysfml.cfml_py_utilities` does not import cleanly, the patched generator can fall back to `CFML_api`.

## Single-Job Generation

The simplest generator launcher is:

- [/Users/UserE/project/Code/Generation/tacc/run_gen_like_paper_uniform_spr.sh](/Users/UserE/project/Code/Generation/tacc/run_gen_like_paper_uniform_spr.sh)

This is good for:

- smoke tests
- one shard
- one node

Typical use:

```bash
cd /scratch/$USER/project/Code/Generation/tacc
sbatch run_gen_like_paper_uniform_spr.sh
```

Important env vars:

```bash
REPO_DIR=/scratch/$USER/project
WORKROOT=/scratch/$USER/pycrysfml_spr
VENV_DIR=$WORKROOT/venv
FULLPROF_DIR=$WORKROOT/fullprof
OUT_DIR=/scratch/$USER/ai_diffraction_generated
TOTAL_SAMPLES=4096
GENMAP_NUM_SHARDS=1
GENMAP_SHARD_ID=0
BATCH_ID=test
NUM_PROCESSES_OVERRIDE=1
```

## Large-Scale Sharded Generation

The main launcher is:

- [/Users/UserE/project/Code/Generation/tacc/run_gen_like_paper_uniform_array_spr.sh](/Users/UserE/project/Code/Generation/tacc/run_gen_like_paper_uniform_array_spr.sh)

Despite the name, the safest mode was not a huge Slurm array. What worked best was:

- one Slurm job per node
- several shard lanes inside that job
- one Python generator process per lane

Why:

- the generator is CPU-heavy
- `CFML_api` is sensitive
- shard-level parallelism is safer than a big local multiprocessing pool

Important env vars:

```bash
TOTAL_SAMPLES=2000000
NUM_SHARDS=100
RUN_TAG=uniform2m_v1
BATCH_ID=uniform2m_v1
BASE_OUTPUT_HDF5=/scratch/$USER/ai_diffraction_generated/uniform_10m/uniform_eg_impurity_shard
NUM_PROCESSES_OVERRIDE=1
SHARDS_PER_JOB=8
ARRAY_TASK_ID_OVERRIDE=0
ARRAY_TASK_COUNT_OVERRIDE=4
RESUME_GENERATION=1
```

The launch pattern we ended up using was multiple ordinary `sbatch` jobs, for example four jobs, each handling a disjoint block of shard lanes:

```bash
ARRAY_TASK_COUNT_OVERRIDE=4 ARRAY_TASK_ID_OVERRIDE=0 sbatch run_gen_like_paper_uniform_array_spr.sh
ARRAY_TASK_COUNT_OVERRIDE=4 ARRAY_TASK_ID_OVERRIDE=1 sbatch run_gen_like_paper_uniform_array_spr.sh
ARRAY_TASK_COUNT_OVERRIDE=4 ARRAY_TASK_ID_OVERRIDE=2 sbatch run_gen_like_paper_uniform_array_spr.sh
ARRAY_TASK_COUNT_OVERRIDE=4 ARRAY_TASK_ID_OVERRIDE=3 sbatch run_gen_like_paper_uniform_array_spr.sh
```

This avoided the Slurm array issues we saw on TACC.

## Uniform Extinction-Group Sampling

For the 37-bit target, we wanted uniform coverage across all 99 EGs.

The generator now supports:

```bash
export EG_SAMPLING_MODE=uniform
```

This is not just iid uniform random. The code uses a balanced block schedule so partial output stays much more uniform as a run progresses.

## Checkpoint / Resume

This was essential.

Each shard HDF5 stores progress metadata:

- `tasks_processed`
- `successes_written`
- shard identity / seed info

If a job dies, rerunning the same shard resumes instead of starting over.

The generator code for this is in:

- [/Users/UserE/project/Code/Generation/gen_like_paper.py](/Users/UserE/project/Code/Generation/gen_like_paper.py)

The important env var is:

```bash
export RESUME_GENERATION=1
```

## Repacking Shards Into A Train-Ready HDF5

Once enough shards are written, repack them into one HDF5 with:

- `X_train`, `y_train`
- `X_val`, `y_val`
- `X_test`, `y_test`

Scripts:

- [/Users/UserE/project/Code/Generation/tacc/repack_uniform_shards_trainready.py](/Users/UserE/project/Code/Generation/tacc/repack_uniform_shards_trainready.py)
- [/Users/UserE/project/Code/Generation/tacc/run_repack_uniform_shards_spr.sh](/Users/UserE/project/Code/Generation/tacc/run_repack_uniform_shards_spr.sh)

The repacker supports:

- ordered split assignment
- partial-shard ingestion
- skipping locked files while generation is still active

Typical example:

```bash
python repack_uniform_shards_trainready.py \
  --input-dir /scratch/$USER/ai_diffraction_generated/uniform_10m \
  --output-h5 /scratch/$USER/ai_diffraction_generated/uniform2m_v1_partial_trainready_ordered.hdf5 \
  --pattern "*.hdf5" \
  --train-percent 90 \
  --val-percent 5 \
  --split-mode order \
  --allow-partial \
  --min-written 1
```

## What To Copy To Another HPC System

At minimum, copy:

- the repo, especially:
  - `Code/Generation/gen_like_paper.py`
  - `Code/Generation/tacc/cfml_api_single_pattern.py`
  - `Code/Generation/tacc/run_gen_like_paper_uniform_spr.sh`
  - `Code/Generation/tacc/run_gen_like_paper_uniform_array_spr.sh`
  - `Code/Generation/tacc/repack_uniform_shards_trainready.py`
  - `Code/Post_Processing/FINAL_SPG_ExtG_CrysS_Table.csv`

Then adapt:

- Slurm partition / account / walltime
- module names
- compiler / Python versions
- `WORKROOT`, `VENV_DIR`, `FULLPROF_DIR`

## TACC-Specific Lessons That Likely Generalize

1. Build and test `crysfml` on compute nodes, not login nodes.
2. Prefer one generator process per shard lane.
3. Do not rely on a large local `multiprocessing.Pool` around `CFML_api`.
4. Keep shard checkpointing from day one.
5. Repack from shards into one train-ready HDF5 for training.
6. If you need high throughput, scale by shard fanout across jobs and nodes, not by forcing more stateful work into one Python process.

## If You Only Want The Minimum Working Sequence

1. Clone repo to scratch.
2. Run `build_pycrysfml_spr.sh` on a compute node.
3. Run `run_generate_one_pattern_spr.sh` to confirm the powder path.
4. Run `run_gen_like_paper_uniform_spr.sh` for a small smoke test.
5. Run `run_gen_like_paper_uniform_array_spr.sh` across several jobs for shard production.
6. Run `repack_uniform_shards_trainready.py` once enough shards exist.

That is the shortest path that actually worked for us.
