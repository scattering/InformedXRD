# opXRD Single-Phase Cu-Kalpha Community Surface

## Scope

This directory contains candidate benchmark views derived from the public
opXRD archive. The wavelength rule matches the RRUFF source convention:

- accept an explicit Cu Kalpha1/Kalpha2 pair near
  `1.540562/1.544390 Angstrom`, in either order;
- accept the conventional 2:1 weighted-average wavelength,
  `1.541838 Angstrom`; and
- reject monochromated Kalpha1-only, monochromated Kalpha2-only, and
  non-Cu measurements.

Every retained record declares exactly one phase and contains a lattice and
atomic basis from which `pymatgen` derives a space group. Space groups are
mapped to the same 99 extinction groups used by RRUFF-473 and RRUFF-325.

## Candidate Audit

Starting from `92,552` opXRD records:

- `1,069` declare exactly one phase;
- `908` contain enough structural information to derive symmetry;
- `229` satisfy the RRUFF-compatible Cu-Kalpha rule:
  - `127` record the Kalpha1/Kalpha2 pair;
  - `102` record the weighted-average wavelength;
- `2` have non-strictly-increasing two-theta axes and are rejected;
- `2` repeated patterns are removed by exact pattern hashing; and
- `225` unique, valid-axis candidates remain before angular-window and
  native-resolution filtering.

The complete audit surface, including duplicate and failure metadata, is
`opxrd_cukalpha_candidates.csv`.

## Crystal Communities

The stored opXRD structures were projected into the frozen ICSD structural
map from:

> Nguyen et al., "Computed materials proposals depart from the structural
> memory of experimental discovery," arXiv:2606.30967.

The production embedding uses the same 213-dimensional structure descriptor,
three rounds of Voronoi-neighbor message passing, frozen 32-dimensional PCA
frame, Louvain communities, and per-community 95th-percentile centroid-distance
thresholds as the arXiv work.

- `210/227` valid-axis records are within the published 256-site external
  projection scope;
- these records map to `111` ICSD communities;
- `110` are in-basin under the per-community threshold; and
- `100` are frontier-like.

The remaining `17` structures contain more than 256 atomic sites. They remain
in the candidate and angular-window manifests with explicit projection errors.
They are not silently treated as structural outliers.

Community membership replaces the RRUFF-specific mineral-family identifier as
the structural grouping variable. It does not by itself establish that the
measured pattern agrees with the supplied structure. A direct Pawley/Le Bail
profile-consistency audit remains the appropriate measurement-quality step.
Frontier-like community status is retained as a stratum, not used as an
automatic exclusion rule.

## Fixed Angular Views

All released views:

- require measured support across the complete stated interval;
- require a median native step no larger than `0.05 degrees`;
- interpolate to a `0.01-degree` grid;
- apply per-pattern maximum normalization; and
- retain one representative from each exact duplicate-pattern group.

No endpoint extrapolation is used in these files.

| Window | Records | Community assigned | SGs | EGs | Grid points |
|---|---:|---:|---:|---:|---:|
| 5-90 degrees | 13 | 12 | 8 | 7 | 8,501 |
| 5-80 degrees | 39 | 37 | 12 | 11 | 7,501 |
| 10-90 degrees | 50 | 49 | 23 | 20 | 8,001 |
| 10-80 degrees | 98 | 95 | 28 | 23 | 7,001 |
| 15-80 degrees | 139 | 136 | 35 | 28 | 6,501 |

The `10-80 degree` view is the best current compromise. The `5-90 degree`
view is directly aligned with the frozen RRUFF grid but is too small and too
narrow in label coverage to support a primary benchmark. Starting at
`15 degrees` retains more records but discards a substantial low-angle region
that is important for systematic-absence evidence.

The builder can optionally allow at most one native input bin of constant edge
fill with `--endpoint-native-bin-tolerance 1`. This is not used in the files in
this directory.

## Synthetic-Training Consequence

The wavelength-compatible opXRD records use the same Cu-Kalpha convention as
RRUFF. No new wavelength simulation is required.

A model evaluated on the `10-80 degree` view should be trained on that same
angular interval. Existing `5-90 degree` synthetic patterns can be cropped and
resampled deterministically, so the diffraction patterns do not need to be
simulated again. Comparable checkpoints still need to be retrained on the
cropped input representation.

## Benchmark Design

The opXRD surface should be reported separately from RRUFF-473 and RRUFF-325.
The sources differ in phase semantics, instrument distribution, angular
coverage, and structural metadata. A joint summary may report a source-balanced
macro-average, but source-specific scores should remain primary.

## Reproduction

Run from the NeurIPS revision repository:

```bash
$HOME/opt/anaconda3/envs/crystal-communities/bin/python \
  docs/review_notes/opxrd_metadata_2026-04-02/build_opxrd_cukalpha_community_surface.py \
  --archive "<DATA_ROOT>/AI and Diffraction/OpXRD/opxrd.zip" \
  --sg-eg-table Code/Post_Processing/FINAL_SPG_ExtG_CrysS_Table.csv \
  --icsd-features $HOME/crystal-communities/notes/features.npy \
  --community-assignments $HOME/crystal-communities/notes/icsd_community_assignments/community_assignments_labels3.csv \
  --community-thresholds $HOME/crystal-communities/notes/per_community_thresholds_fullmap_p95.json \
  --crystal-community-scripts $HOME/crystal-communities/scripts \
  --output-dir docs/review_notes/opxrd_metadata_2026-04-02/opxrd_cukalpha_community_surface_2026-07-27 \
  --n-jobs 8
```
