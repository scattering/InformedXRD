# Datasets And Generation Notes

## Status Warning

This file is now primarily a historical working note from the earlier UserC-shared-directory phase of the project. It still contains references to legacy datasets that were important during exploration, but it should \textbf{not} be treated as the authoritative provenance document for the current manuscript revision.

For the current manuscript-facing provenance, use:

- [current_data_provenance_2026-04-03.md](/Users/UserE/project/docs/review_notes/current_data_provenance_2026-04-03.md)
- [reproducibility_packet_2026-04-01.md](/Users/UserE/project/docs/review_notes/reproducibility_packet_2026-04-01.md)
- [artifact_manifest_2026-04-01.md](/Users/UserE/project/docs/review_notes/artifact_manifest_2026-04-01.md)

## Current Dataset Roles

### 1. Vista Training Baseline

Current large training dataset used on Vista:

- [rruff_aug_10M_trainready_repacked.hdf5](/work2/ALLOC2/UserC/shared_dir/rruff_aug_10M_trainready_repacked.hdf5)

Location:

- `/work2/ALLOC2/UserC/shared_dir/rruff_aug_10M_trainready_repacked.hdf5`

What it is:

- large HDF5 used for the current ViT training runs on Vista
- approximate split sizes:
  - `X_train`: `(9000000, 8501)`
  - `y_train`: `(9000000,)`
  - `X_val`: `(500000, 8501)`
  - `y_val`: `(500000,)`
  - `X_test`: `(500000, 8501)`
  - `y_test`: `(500000,)`

Important caveat:

- this dataset only covers `70` of the `99` canonical extinction groups

### 2. Final Real-Data Evaluation Set

The small real RRUFF dataset that matches the current `8501`-point training input is:

- [RRUFF_low_bkg_full_intensity_cleaned_maxnorm.hdf5](/work2/ALLOC2/UserC/shared_dir/RRUFFTesting/RRUFF_low_bkg_full_intensity_cleaned_maxnorm.hdf5)

Location:

- `/work2/ALLOC2/UserC/shared_dir/RRUFFTesting/RRUFF_low_bkg_full_intensity_cleaned_maxnorm.hdf5`

Contents:

- `X_test`: `(1222, 8501)`
- `y_test`: `(1222,)`

This is the most likely final real-data test set for the current ViT pipeline because:

- it is small
- it appears to be real RRUFF test data
- it matches the `8501` input length used in the current Vista training code

### 3. Other RRUFF Testing Files In The Same Tree

These also exist under the same shared directory tree:

- [RRUFF_norm_uniform_sampling_5_90_deg_1000_intensity.hdf5](/work2/ALLOC2/UserC/shared_dir/RRUFFTesting/RRUFF_norm_uniform_sampling_5_90_deg_1000_intensity.hdf5)
  - `X_test`: `(2795, 1000)`
  - `y_test`: `(2795,)`
- [RRUFF_norm_reduced_range_10_80_deg_700_intensity.hdf5](/work2/ALLOC2/UserC/shared_dir/RRUFFTesting/RRUFF_norm_reduced_range_10_80_deg_700_intensity.hdf5)
  - `X_test`: `(2795, 700)`
  - `y_test`: `(2795,)`

These do not match the current `8501`-point Vista training setup.

### 4. In-Progress Uniform Synthetic Dataset

Current synthetic generation plan:

- generate in `2,000,000`-sample batches first
- keep sampling uniform across all `99` extinction groups
- keep the repo-style impurity / second-phase / paper-style augmentation path
- later combine multiple `2M` batches toward a `10,000,000`-sample synthetic corpus

Current output root on Stampede3 `spr`:

- `/scratch/ALLOC3/UserE/ai_diffraction_generated/uniform_10m`

Current active batch:

- Slurm array job `2957979`
- partition `spr`
- first target batch size: `2,000,000` examples
- `64` active shard workers total (`4` array tasks x `16` shard workers per job)

Planned train-ready repack output for the first batch:

- `/scratch/ALLOC3/UserE/ai_diffraction_generated/uniform2m_v1_trainready.hdf5`

## Why We Need The Uniform Synthetic Dataset

The current RRUFF-based training distribution is geologically skewed and incomplete for the `37`-bit BCE head:

- only `70/99` extinction groups are present
- some rule bits are missing entirely
- rare rule bits can die under BCE because the easiest solution is to predict `0` almost always
- operator learning becomes entangled with common geological contexts instead of learning the algebra cleanly

The uniform synthetic dataset is intended to fix that by forcing coverage of the full crystallographic rule space.

## Generation On Stampede3 `spr`

### Goal

Use the repo’s realistic generator path, not the simpler toy generator. The desired generator is:

- [gen_like_paper.py](/Users/UserE/project/Code/Generation/gen_like_paper.py)

Why this script:

- it uses pyxtal structure generation
- it uses pycrysfml/CrysFML for powder generation
- it includes impurity / second-phase mixing
- it includes K-alpha splitting
- it includes paper-style augmentation and background noise

### What Was Added

The generator was patched to support:

- `EG_SAMPLING_MODE=uniform`
- fallback to `CFML_api` when `pycrysfml.cfml_py_utilities` is unavailable
- interpolation of fallback output onto the repo’s fixed `8501`-point `2theta` grid
- serial execution when `NUM_PROCESSES_OVERRIDE=1`

Relevant files:

- [gen_like_paper.py](/Users/UserE/project/Code/Generation/gen_like_paper.py)
- [run_gen_like_paper_uniform_spr.sh](/Users/UserE/project/Code/Generation/tacc/run_gen_like_paper_uniform_spr.sh)
- [run_gen_like_paper_uniform_array_spr.sh](/Users/UserE/project/Code/Generation/tacc/run_gen_like_paper_uniform_array_spr.sh)

### What Works So Far

On Stampede3 `spr`, we confirmed:

- the generator starts in uniform `99`-EG mode
- the repo-style impurity path is being exercised
- the fallback backend can get past the earlier fixed-grid mismatch
- isolated subprocess execution of the `CFML_api` pattern step avoids the earlier heap corruption
- a serial smoke test completed end to end and wrote an HDF5 shard
- shard-level resume/checkpointing now works

Successful smoke test:

- job `2957948`
- partition `spr`
- state: `COMPLETED`
- elapsed: `00:01:17`
- output shard:
  - `/scratch/ALLOC3/UserE/ai_diffraction_generated/uniform_eg_impurity_batchsmoke5_02182025.hdf5`

Checkpoint/resume validation:

- initial tiny shard job: `2957950`
- resume job on the same shard file: `2957960`
- result on disk:
  - `intensity` shape: `(4, 8501)`
  - `tasks_processed = 5`
  - `successes_written = 4`
  - successful `task_id` values: `[0, 1, 3, 4]`

This confirms:

- failed generation attempts do not count as completed samples
- rerunning the same shard resumes from the last attempted task
- the shard stops only when the target number of written samples is reached

### Current Status

The main remaining work is scale-out and repacking, not basic generator stability.

At this point, the known-safe generation mode is:

- one generator process per Slurm task
- one isolated subprocess per `CFML_api` powder-pattern call

This is being scaled with shard-level parallelism, not a large local Python `multiprocessing.Pool`.

## Repacking Into Vista-Ready Training Data

The generator shards do not use the same schema as the Vista trainer. Generator shards contain:

- `2theta`
- `intensity`
- `space_group`
- `extinction_group`
- `crystal_system`
- shard progress metadata in HDF5 attributes

The Vista trainer expects a train-ready HDF5 layout:

- `X_train`, `y_train`
- `X_val`, `y_val`
- `X_test`, `y_test`

To bridge this, the following repack path now exists:

- repack script:
  - [repack_uniform_shards_trainready.py](/Users/UserE/project/Code/Generation/tacc/repack_uniform_shards_trainready.py)
- `spr` launcher:
  - [run_repack_uniform_shards_spr.sh](/Users/UserE/project/Code/Generation/tacc/run_repack_uniform_shards_spr.sh)

Important repack behavior:

- only complete shards are ingested
- output is resumable
- processed shard paths / ids / split assignments are recorded in the output HDF5
- shard-to-split assignment is stable by `shard_id % 100`
- default split is `90/5/5` across train / val / test

This means the first finished `2M` batch can be converted directly into a standard HDF5 and trained on Vista without another trainer rewrite.

## Intended Scaling Strategy Once Stable

When the generator is stable, the correct scale-out model is:

1. one generator process per Slurm task
2. shard-level parallelism across many `spr` jobs or a Slurm array
3. one HDF5 shard per task
4. each shard carries progress metadata so failed jobs can resume
5. repack completed shards into a train-ready HDF5 for Vista

This is preferred over a large local Python `multiprocessing.Pool`, because the low-level crystallography bindings do not currently look fork-safe.

## Recommended Next Steps

1. let the first `2M` uniform batch finish on `spr`
2. repack the completed shards into `uniform2m_v1_trainready.hdf5`
3. train on Vista from that train-ready HDF5
4. evaluate on:
   - the large RRUFF-based training/validation baseline
   - the real-data test set `RRUFF_low_bkg_full_intensity_cleaned_maxnorm.hdf5`
