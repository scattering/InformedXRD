# Provenance

The paper states that components reused from the public companion release
(Baggett et al.) are marked with their provenance. This file is that marking:
it lists, for every top-level path, whether it is **adapted from the public
Baggett et al. release** (the upstream `paper-ai-diffraction` reproducibility
repository, frozen 2026-05-01) or **specific to InformedXRD** (this paper).

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
| `README.md`, `PROVENANCE.md` | This packaging |

## Third-party inputs

- **RRUFF**: source archive for RRUFF-473/RRUFF-325 (benchmark payloads on
  Hugging Face reference frozen scan identifiers).
- **opXRD** (Hollarek et al.): source archive for the 98-record surface;
  record identifiers are in `case_studies/opxrd/community_surface/*_manifest.csv`.
- **SimXRD / XRDBench**: training corpus contract and architecture recipe for
  the six-model study (unchanged public `train_1` contract).
- **EXPO extinction-symbol table** (derived from *International Tables for
  Crystallography*): underlying tabulation parsed, canonicalized, and inverted
  into `assets/lookups/`.
- **ICSD structural community map** (Nguyen et al.): frozen projection used to
  assign structural families on the opXRD surface.

## Path redaction

Cluster launchers and result JSONs reference the original compute environment
through `$SCRATCH` placeholders; Slurm accounts are redacted as
`<ALLOCATION>`, and local archive roots as `<DATA_ROOT>`/`$HOME`. No
quantitative artifact depends on the redacted values.
