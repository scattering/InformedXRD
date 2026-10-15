# Running `gen_rruff_conditioned.py` On Another Cluster

This note documents the current recommended way to run the corrected `rruff_conditioned` generator on a non-TACC Slurm cluster.

## Current Recommendation

If you want more useful data while the Vista training queue turns over, prioritize:

1. `rruff_conditioned`
2. corrected `uniform_conditioned` only after that

Reason:

- the immediate question is whether the corrected realistic generator improves the defended real benchmark
- more `rruff_conditioned` data answers that directly
- more `uniform_conditioned` data is still useful later for larger backbone pretraining, but it is not the current bottleneck

## Current Generator State

The current local working-tree version of:

- [/Users/UserE/project/Code/Generation/gen_rruff_conditioned.py](/Users/UserE/project/Code/Generation/gen_rruff_conditioned.py)

includes the fixes that mattered in profiling:

- filtered chemistry support
- exclusion of short-half-life elements
- exclusion of PyXtal-unsupported elements with missing radius metadata
- ICSD-based marginal + pairwise chemistry prior
- Wyckoff-compatible `numIons` by construction
- main-phase requirement for a general-position orbit when available
- relaxed impurity symmetry acceptance
- `spglib` validation at `symprec=1e-3`

That is the version you should run elsewhere. Do not use an older committed copy unless you first sync the current file.

## Required Inputs

You need:

- a checkout of the repo
- a Python environment with:
  - `h5py`
  - `numpy`
  - `pandas`
  - `spglib`
  - `ase`
  - `pyxtal`
  - `scipy`
  - `tqdm`
  - `pymatgen`
- `pycrysfml` or `CFML_api`
- a FullProf install
- extinction mapping CSV:
  - [`FINAL_SPG_ExtG_CrysS_Table.csv`](/Users/UserE/project/Code/Post_Processing/FINAL_SPG_ExtG_CrysS_Table.csv)
- RRUFF nuisance prior CSV:
  - `option1_metadata_qc_full.csv`
- ICSD chemistry CSV:
  - `ICSD_With_Bravais.csv`

## Generic Launcher

Use:

- [/Users/UserE/project/Code/Generation/run_gen_rruff_conditioned_array_generic.sh](/Users/UserE/project/Code/Generation/run_gen_rruff_conditioned_array_generic.sh)

This is the cluster-agnostic version of the TACC array launcher. It keeps the same shard/lane logic but removes TACC-specific account, partition, and module assumptions.

You will still need to edit:

- `#SBATCH` lines
- any `module load ...` lines

to match the target cluster.

## Recommended First Run

For a first run on another cluster:

- `TOTAL_SAMPLES=100000`
- `NUM_SHARDS=16`
- `--array=0-1`
- `SHARDS_PER_JOB=8`
- `EG_SAMPLING_MODE=rruff`
- `PRIOR_MODE=robust`
- `NUM_PROCESSES_OVERRIDE=1`
- `RESUME_GENERATION=1`

This is the same layout that worked for the corrected `100k` TACC pilot.

## Example Submission

```bash
sbatch --array=0-1 \
  --export=ALL,REPO_DIR=/path/to/project,WORKROOT=/path/to/workroot,VENV_DIR=/path/to/venv,FULLPROF_DIR=/path/to/fullprof,OUT_DIR=/path/to/output/rruff_conditioned_100k,TOTAL_SAMPLES=100000,NUM_SHARDS=16,RUN_TAG=rruff_cond_100k_other,EG_SAMPLING_MODE=rruff,PRIOR_MODE=robust,RRUFF_PRIOR_CSV=/path/to/option1_metadata_qc_full.csv,ICSD_CHEMISTRY_CSV=/path/to/ICSD_With_Bravais.csv,EXT_MAP_CSV=/path/to/FINAL_SPG_ExtG_CrysS_Table.csv,SHARDS_PER_JOB=8,NUM_PROCESSES_OVERRIDE=1,RESUME_GENERATION=1 \
  /path/to/project/Code/Generation/run_gen_rruff_conditioned_array_generic.sh
```

## Important Environment Variables

- `REPO_DIR`: repo root
- `WORKROOT`: root containing your venv / local pycrysfml assets
- `VENV_DIR`: Python environment
- `FULLPROF_DIR`: FullProf install root
- `OUT_DIR`: output directory for HDF5 shards
- `TOTAL_SAMPLES`: total dataset size across all shards
- `NUM_SHARDS`: number of HDF5 shards
- `RUN_TAG`: batch tag used in output naming
- `EG_SAMPLING_MODE`: `rruff` or `uniform`
- `PRIOR_MODE`: use `robust`
- `RRUFF_PRIOR_CSV`: nuisance prior table
- `ICSD_CHEMISTRY_CSV`: chemistry prior table
- `EXT_MAP_CSV`: SG->EG mapping
- `SHARDS_PER_JOB`: shard lanes per array task
- `NUM_PROCESSES_OVERRIDE`: keep at `1`
- `RESUME_GENERATION`: set to `1`

## If You Need To Run Without The Array Launcher

The direct invocation is:

```bash
cd /path/to/project/Code/Generation

export FULLPROF=/path/to/fullprof
export EG_SAMPLING_MODE=rruff
export PRIOR_MODE=robust
export TOTAL_SAMPLES=100000
export GENMAP_NUM_SHARDS=16
export GENMAP_SHARD_ID=0
export BATCH_ID=rruff_cond_100k_other
export BASE_OUTPUT_HDF5=/path/to/output/rruff_conditioned_shard
export EXT_MAP_CSV=/path/to/FINAL_SPG_ExtG_CrysS_Table.csv
export RRUFF_PRIOR_CSV=/path/to/option1_metadata_qc_full.csv
export ICSD_CHEMISTRY_CSV=/path/to/ICSD_With_Bravais.csv
export RESUME_GENERATION=1

python gen_rruff_conditioned.py
```

Repeat that with different `GENMAP_SHARD_ID` values to fill the shard set.

## Operational Notes

- The corrected generator is now dominated by powder generation cost, not PyXtal compatibility failure.
- `RESUME_GENERATION=1` is important. Timed-out shards can be resumed.
- Do not mix shards from older pre-fix runs with the current corrected generator.
- If you want the other cluster to match TACC exactly, sync the current working-tree versions of:
  - [/Users/UserE/project/Code/Generation/gen_rruff_conditioned.py](/Users/UserE/project/Code/Generation/gen_rruff_conditioned.py)
  - [/Users/UserE/project/Code/Generation/run_gen_rruff_conditioned_array_generic.sh](/Users/UserE/project/Code/Generation/run_gen_rruff_conditioned_array_generic.sh)

## Expected Throughput

On the corrected TACC profile runs, the best `100`-sample profile landed around:

- `~1566 samples/hour`

That implies roughly:

- `~4 hours` walltime for `100k` with `16` shards in parallel

The exact result on another cluster will depend mostly on:

- CPU speed
- PyCrysFML / FullProf behavior on that system
- filesystem performance for the shard writes
