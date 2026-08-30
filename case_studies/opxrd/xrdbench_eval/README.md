# XRDBench Models on the opXRD Cu-Kalpha Surface

## Scope

This directory records an evaluation of six XRDBench architectures on the
strict opXRD `10-80 degree` Cu-Kalpha surface. The evaluation reuses the frozen
checkpoints previously trained on the public SimXRD `train_1` split:

- `597,845` training patterns;
- `230` space-group targets;
- `10-80 degree` two-theta support;
- `0.02-degree` spacing and `3,501` input points;
- nominal maximum intensity `100`; and
- fixed wavelength `1.54184 Angstrom`.

The opXRD input contains `98` unique patterns spanning `28` space groups and
`23` extinction groups. Its `0.01-degree` grid is exactly decimated to the
SimXRD `0.02-degree` grid with no interpolation or endpoint fill. Each
decimated pattern is independently normalized to maximum `100`.

Retraining was not repeated because neither the SimXRD training corpus nor the
model input contract changed. Reusing the same checkpoints also makes this a
direct third-surface comparison with the clean RRUFF-473 and RRUFF-325
evaluations.

## Vista Run

- Job: `868916`
- Partition/account: `gh` / `<ALLOCATION>`
- Compute node: `c608-132`
- State: `COMPLETED`
- Elapsed time: `38 seconds`
- Remote root:
  `$SCRATCH/xrdbench_opxrd_community_20260727`
- Remote result:
  `$SCRATCH/xrdbench_opxrd_community_20260727/runs/job_868916`

The job was initially queued on `gb`, but Slurm estimated a `09:05` start
because a 16-node job occupied nearly the whole partition. The pending job was
rerouted in place to `gh` and started immediately. All model inference ran on
the allocated compute node.

## Results

| Model | SG Top-1/3/5 | EG Top-1/3/5 | Wrong EG distance `<=2` | Mean wrong EG distance |
|---|---:|---:|---:|---:|
| CNN10 | `6.12 / 14.29 / 24.49` | `5.10 / 19.39 / 38.78` | `29.03%` | `3.258` |
| CNN11 | `7.14 / 17.35 / 24.49` | `6.12 / 20.41 / 26.53` | `31.52%` | `3.174` |
| BiLSTM | `2.04 / 12.24 / 18.37` | `5.10 / 15.31 / 22.45` | `23.66%` | `3.441` |
| BiGRU | `3.06 / 12.24 / 20.41` | `4.08 / 17.35 / 24.49` | `20.21%` | `3.394` |
| Transformer | `1.02 / 21.43 / 27.55` | `4.08 / 22.45 / 31.63` | `17.02%` | `3.745` |
| PatchTST | `9.18 / 17.35 / 22.45` | `10.20 / 20.41 / 34.69` | `32.95%` | `3.034` |

PatchTST has the highest EG Top-1, while CNN10 has the highest EG Top-5. The
models remain weak in absolute terms, and their ordering depends on the
ranking depth.

## Wavelength Stratum

All six models score higher on the `33` weighted-average records than on the
`65` records that explicitly store a Kalpha1/Kalpha2 pair:

| Model | Explicit-pair EG Top-1/3/5 | Weighted-average EG Top-1/3/5 |
|---|---:|---:|
| CNN10 | `1.54 / 13.85 / 33.85` | `12.12 / 30.30 / 48.48` |
| CNN11 | `3.08 / 12.31 / 16.92` | `12.12 / 36.36 / 45.45` |
| BiLSTM | `1.54 / 9.23 / 13.85` | `12.12 / 27.27 / 39.39` |
| BiGRU | `1.54 / 10.77 / 15.38` | `9.09 / 30.30 / 42.42` |
| Transformer | `1.54 / 16.92 / 21.54` | `9.09 / 33.33 / 51.52` |
| PatchTST | `6.15 / 10.77 / 30.77` | `18.18 / 39.39 / 42.42` |

This difference should not be attributed to wavelength representation alone:
the two small strata also differ in source and label composition. It is,
however, a reason to preserve wavelength mode as a mandatory benchmark
stratum.

## DAG Result

The exact benchmark-level uniform-wrong-label null has:

- mean wrong-label distance `2.876`;
- `32.62%` of wrong labels within distance `<=2`; and
- descendant/ancestor/branch fractions `10.57 / 32.61 / 56.79%`.

For the six models, mean wrong-case distances range from `3.034` to `3.745`.
Each is larger than its model-error-conditioned null (`2.820-2.840`).
Wrong-case distance locality therefore does not improve on the matched null on
this surface. The predictions are strongly shifted toward descendant,
lower-symmetry groups, but that directionality can also reflect systematic
low-symmetry collapse and should not be presented as evidence of good
classification.

## Test-Marginal Diagnostic

An input-blind predictor allowed to use the opXRD test-set label frequencies
would score `22.45 / 53.06 / 66.33%` Top-1/3/5. This is not a deployable
baseline because it uses the evaluation marginal, but it quantifies how
strongly imbalanced and small the surface is. The six models all fall below
this diagnostic, reinforcing that Top-k accuracy should be accompanied by
pattern-use tests and source/label strata.

## Files

`job_868916/summary.json` contains the compact model table.
Each model directory contains:

- `eval.json`: complete SG/EG and stratum metrics;
- `posterior.npz`: SG and EG posterior arrays;
- `failure.json`: topology-analysis input;
- `topology.json`: observed DAG result;
- `topology_bootstrap.json`: bootstrap intervals; and
- `topology_error_conditioned_null.json`: the exact matched null.

`job_868916/manifest.json` contains sizes and SHA-256 hashes for the generated
run files. All `45` listed files were verified after transfer from Vista.
