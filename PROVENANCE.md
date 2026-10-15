# Provenance

The paper states that components reused from the public companion release
(Baggett et al.) are marked with their provenance. This file is that marking:
it lists, for every top-level path, whether it is **adapted from the public
Baggett et al. release** (the upstream `paper-ai-diffraction` reproducibility
repository, frozen 2026-05-01) or **specific to InformedXRD** (this paper).
A third section describes `research_codebase/`, the full research codebase
from which the curated paths were extracted.

## Adapted from the public Baggett et al. release

| Path | Role |
|---|---|
| `src/` | ViT model implementation, calibration/split-validity/topology evaluation code |
| `scripts/` | RRUFF-473/325 benchmark builders, prior-asset export, figure/table scripts |
| `assets/lookups/` | Extinction-group mapping tables (first released in machine-usable form with that work; verified and canonicalized here) |
| `assets/topology/` | Condensed *t*-subgroup DAG adjacency |
| `assets/reviewer_examples/`, `assets/figure_data/` | Reviewer example patterns and figure data |
| `results/` | Calibration metrics, failure modes, split validity, bootstrap CI JSONs for the four ViT checkpoints |
| `notebooks/reviewer/` | Reviewer quickstart and full RRUFF-325 evaluation notebooks |
| `configs/`, `reproducibility/`, `docs/` (except this file), environment files, `pyproject.toml`, `LICENSE` | Packaging and documentation of the upstream release |

The four ViT checkpoints on the Hugging Face model repository
(`xrd_model_82ept35h`, `xrd_model_9rwv1qly`, `xrd_model_cscjfdwk`,
`xrd_model_wwuvp1kj`) are likewise from the public Baggett et al. release.

## Specific to InformedXRD (this paper)

| Path | Role |
|---|---|
| `evaluation/nulls/` | DAG-metric reference distributions: exact uniform nulls, model-matched (error-conditioned) nulls, and their generator |
| `evaluation/permutation/` | Label-permutation (rank-lift) test of pattern dependence |
| `evaluation/mcnemar/` | Exact paired McNemar tests across the six XRDBench architectures |
| `evaluation/rescoring_example/` | Worked example collapsing 230-way space-group outputs into the 99-way extinction-group space |
| `evaluation/hex_trig/` | Reflection-condition enumeration for the hexagonal/trigonal extinction-class pair |
| `case_studies/cnn_control/` | 1D residual CNN control: standalone implementation, loader, DAG evaluation, convergence test |
| `case_studies/xrdbench/` | Six XRDBench architectures trained under the SimXRD recipe and rescored in extinction-group space |
| `case_studies/sg_vs_eg/` | Matched space-group vs extinction-group target comparison and scaling study |
| `case_studies/opxrd/` | opXRD metadata audit, Cu-Kα community surface construction, and six-model evaluation |
| `README.md`, `PROVENANCE.md`, `docs/RESEARCH_CODEBASE_MAP.csv` | This packaging |

## Full research codebase: `research_codebase/`

`research_codebase/` is the complete working codebase of the project (the
`scattering/ai-diffraction` repository), added for the camera-ready release
as a verbatim copy of the snapshot that was prepared for the NeurIPS 2026 AI
for Science workshop submission. It contains the ViT, CNN and transformer
training code, the synthetic-pattern generation pipeline, the
structure-database pipeline, the derivation of the extinction-group mapping
(`Code/Post_Processing/`), the information-theory analyses
(`Information Theory/`), the interpretability experiments, and the analysis
notebooks. The curated directories listed above were extracted from it.
Where a file exists in both places, the top-level copy is the paper-facing,
path-scrubbed version and is authoritative;
`docs/RESEARCH_CODEBASE_MAP.csv` lists each curated file with its
counterpart in `research_codebase/` and whether the two are byte-identical
or differ (scrubbed or adapted).

Redactions in `research_codebase/` are retained from that snapshot: user
names, e-mail addresses, institutions, HPC allocation identifiers, a
Weights & Biases entity, and two Google Drive folder identifiers are
replaced by neutral placeholders (`UserA`–`UserF`, `OrgA`/`OrgB`,
`ALLOC1`–`ALLOC3`, `ANONYMIZED-…`), matching the `$SCRATCH` /
`<ALLOCATION>` policy of the curated directories; no quantitative artifact
depends on the redacted values. Two changes were made relative to the
snapshot: the outputs of `Code/Database_Pipeline/structures.ipynb`, which
logged per-record ICSD collection-code filenames, were cleared (source cells
unchanged), consistent with the policy below of shipping no per-structure
ICSD records; and the copyright holder in `research_codebase/LICENSE` was
restored from the anonymized placeholder to that of the upstream repository.

## Third-party inputs and licenses

- **RRUFF**: source archive for RRUFF-473/RRUFF-325 (benchmark payloads on
  Hugging Face reference frozen scan identifiers), following the same
  redistribution posture as the frozen upstream release.
- **opXRD** (Hollarek et al.): source archive for the 98-record surface,
  distributed openly via Zenodo without use restrictions; the derived
  resampled surfaces and record manifests in `case_studies/opxrd/` are
  released with attribution to that archive. Record identifiers are in
  `case_studies/opxrd/community_surface/*_manifest.csv`.
- **SimXRD / XRDBench** (MIT License): training corpus contract and
  architecture recipe for the six-model study (unchanged public `train_1`
  contract). `case_studies/xrdbench/overrides/train.py` is a modified copy of
  the upstream training script; the upstream MIT license text is included at
  `case_studies/xrdbench/overrides/UPSTREAM_LICENSE`. The SimXRD corpus
  itself is not redistributed here; released checkpoints are weights trained
  by the authors of this work.
- **EXPO extinction-symbol table** (derived from *International Tables for
  Crystallography*): underlying tabulation parsed, canonicalized, and inverted
  into `assets/lookups/`.
- **ICSD structural community map** (Nguyen et al.): frozen projection used to
  assign structural families on the opXRD surface. Consistent with the
  release's policy of shipping no raw ICSD structures or per-structure
  records, only derived community labels and aggregate diagnostics are
  included; per-record ICSD anchor identifiers were redacted from the
  released overlap audit tables.

## Path redaction

Cluster launchers and result JSONs reference the original compute environment
through `$SCRATCH` placeholders; Slurm accounts are redacted as
`<ALLOCATION>`, and local archive roots as `<DATA_ROOT>`/`$HOME`. No
quantitative artifact depends on the redacted values.
