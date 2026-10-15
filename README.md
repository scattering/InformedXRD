# InformedXRD

Reproducibility repository for **InformedXRD: Reproducible Benchmarks and
Physics-Informed Evaluation for Powder Diffraction Symmetry Classification**
(Edward G. Friedman, Elizabeth J. Baggett, Abhishek Shetty, Derrick Chan-Sew,
Vanellsa Acha, Harshitha Dwaracherla, Paul Kienzle, William Ratcliff).
The same repository serves both venues of the paper:

- NeurIPS 2026, Track on Evaluations and Datasets;
- NeurIPS 2026 Workshop *AI for Science: Verification in the Age of AI
  Scientists* (poster).

Companion artifacts: the benchmark payloads are on Hugging Face at
[informedxrd/informedxrd-benchmarks](https://huggingface.co/datasets/informedxrd/informedxrd-benchmarks)
(with a Croissant file carrying core and Responsible-AI fields at
`metadata/croissant_rai.jsonld`), and the case-study checkpoints at
[informedxrd/informedxrd-checkpoints](https://huggingface.co/informedxrd/informedxrd-checkpoints).
The companion paper *Attention Is Not All You Need for Diffraction*
(Baggett et al., [arXiv:2604.23811](https://arxiv.org/abs/2604.23811)) has
its own public repository,
[scattering/paper-ai-diffraction](https://github.com/scattering/paper-ai-diffraction).

InformedXRD is evaluation infrastructure for powder-X-ray-diffraction (PXRD)
machine learning — deliberately not a new model architecture. It has four parts:

1. a machine-readable mapping from the **230 space groups to the 99 extinction
   groups**, the finest classes powder diffraction can identify from systematic
   absences alone;
2. a **standardized benchmark-construction method** whose inclusion rules are
   specified as code and reproduce from a frozen snapshot, instantiated as
   **RRUFF-473**, **RRUFF-325**, and a 98-record **opXRD** surface;
3. **evaluation tools**: prior-only frequency baselines (deployable and oracle
   variants), a label-permutation test of pattern dependence, and a stratified
   reporting protocol;
4. a **hierarchy-aware error metric** — hop distance on the condensed
   *translationengleiche* subgroup graph over extinction groups — reported
   against exact null reference distributions.

## Quick start: the worked rescoring example

Rescore a 230-way space-group model in extinction-group space and reproduce a
reported results row end to end (the PatchTST row of the opXRD table:
10.20 / 20.41 / 34.69 Top-1/3/5):

```bash
pip install numpy
python evaluation/rescoring_example/rescore_sg_to_eg.py
```

The script loads released 230-way probabilities, collapses them into the 99-way
extinction-group space with `assets/lookups/canonical_extinction_to_space_group.csv`,
scores Top-k, and cross-checks against the released collapsed probabilities.

## Repository layout

| Path | Contents |
|---|---|
| `assets/lookups/` | The extinction-group mapping artifact: canonical 99-class table, 230-row space-group → extinction-group/crystal-system lookup |
| `assets/topology/` | Condensed *t*-subgroup DAG adjacency over extinction groups |
| `scripts/` | Benchmark construction: RRUFF-473 reconstruction from a frozen snapshot, RRUFF-325 derivation, case metadata, prior-asset export |
| `src/` | Core model/evaluation code surface (calibration, split validity, topology analyses) |
| `results/` | Compact paper-backed JSON artifacts (calibration metrics, failure modes, bootstrap CIs) |
| `notebooks/reviewer/` | Reviewer quickstart and full RRUFF-325 evaluation notebooks |
| `evaluation/nulls/` | Hierarchy-metric reference distributions: exact uniform nulls and model-matched (error-conditioned) nulls, plus the generator `analyze_topological_null.py` |
| `evaluation/permutation/` | Label-permutation (rank-lift) test: `rank_lift_null.py` and cluster launchers |
| `evaluation/mcnemar/` | Exact paired McNemar tests over the six XRDBench architectures on RRUFF-473/325 |
| `evaluation/rescoring_example/` | Worked example: collapse 230-way outputs to 99-way extinction groups and reproduce a reported row |
| `evaluation/hex_trig/` | Explicit reflection-condition enumeration for the hexagonal/trigonal extinction-class pair (P - - c vs P - c -) |
| `case_studies/cnn_control/` | 1D residual CNN control: implementation, loader, DAG evaluation, convergence test artifacts |
| `case_studies/xrdbench/` | XRDBench six-architecture study: training/eval/export scripts under the SimXRD recipe, rescored on RRUFF-473/325 |
| `case_studies/sg_vs_eg/` | Matched space-group vs extinction-group target comparison: config templates, scaling-study launchers, evaluation scripts, analysis notebook |
| `case_studies/opxrd/` | opXRD instance: metadata audit, Cu-Kα community surface (manifests = exact record lists + `.npz` surfaces), six-model evaluation (`job_868916`) |
| `docs/` | Benchmark, evaluation, figure, and training documentation from the upstream release; `docs/RESEARCH_CODEBASE_MAP.csv` maps curated files to their origin in the research codebase |
| `research_codebase/` | The full research codebase (ViT/CNN/transformer training, synthetic-pattern generation, structure-database pipeline, extinction-group mapping derivation, information-theory analyses, interpretability, notebooks) from which the curated directories above were extracted; see its README and [PROVENANCE.md](PROVENANCE.md) |

## Paper-claim → artifact map

| Paper section | Artifact |
|---|---|
| §3 / App. I — extinction-group mapping | `assets/lookups/canonical_extinction_to_space_group.csv`, `assets/lookups/FINAL_SPG_ExtG_CrysS_Table.csv` |
| §4.2 / App. A — RRUFF-473 / RRUFF-325 construction | `scripts/reconstruct_rruff_473.py`, `scripts/build_rruff_325_from_473.py` (payloads on Hugging Face) |
| §4.3 / App. H — opXRD surface | `case_studies/opxrd/metadata_audit/`, `case_studies/opxrd/community_surface/` (manifests are the exact record lists) |
| §5.1 / §6.3 — prior-only baselines, permutation test | `results/reviewer/ext_group_priors.*`, `evaluation/permutation/` |
| §5.2 / App. C — DAG metric and null distributions | `assets/topology/extinction_group_adjacency.json`, `evaluation/nulls/` |
| §6.1 / App. E — CNN control and convergence test | `case_studies/cnn_control/` |
| §6.2 — six XRDBench architectures, McNemar | `case_studies/xrdbench/`, `evaluation/mcnemar/` |
| App. B — SG-vs-EG matched comparison and scaling | `case_studies/sg_vs_eg/` |
| App. G — hex/trig pair enumeration | `evaluation/hex_trig/enumerate_hex_trig_conditions.py` |
| App. H — opXRD six-model evaluation | `case_studies/opxrd/xrdbench_eval/job_868916/` |
| App. J — worked rescoring example | `evaluation/rescoring_example/rescore_sg_to_eg.py` |

Provenance of every directory (what is adapted from the public companion
release vs. specific to this paper) is listed in [PROVENANCE.md](PROVENANCE.md).

## External artifacts

Benchmark payloads and model checkpoints are hosted on Hugging Face:

- Dataset (benchmark payloads, reviewer subset):
  [informedxrd/informedxrd-benchmarks](https://huggingface.co/datasets/informedxrd/informedxrd-benchmarks)
- Models (case-study checkpoints):
  [informedxrd/informedxrd-checkpoints](https://huggingface.co/informedxrd/informedxrd-checkpoints)

Expected filenames and roles are listed in
[reproducibility/checkpoint_manifest.csv](reproducibility/checkpoint_manifest.csv)
and [reproducibility/dataset_manifest.csv](reproducibility/dataset_manifest.csv).

## Install

For lightweight analysis and the worked example, `numpy` suffices. For the full
evaluation surface:

```bash
conda env create -f environment.yml
conda activate neurips-ai-diffraction
pip install -e .
```

For checkpoint evaluation, use `environment-train-eval.yml` instead. Cluster
launcher scripts (`*_vista_*.sh`, `vista_*.sh`, `stampede_*.sh`) are released
for auditability of the exact submitted jobs; paths are parameterized by
`$SCRATCH` and the Slurm account is redacted.

## Notes

- Shell launchers and JSON result files reference cluster paths through
  `$SCRATCH` placeholders; every quantitative claim in the paper is
  reproducible from the committed JSON/CSV artifacts plus the Hugging Face
  payloads, without cluster access.
- `docs/UPSTREAM_README.md` is the README of the upstream frozen release this
  repository builds on.

## Citation

```bibtex
@inproceedings{friedman2026informedxrd,
  title     = {InformedXRD: Reproducible Benchmarks and Physics-Informed Evaluation
               for Powder Diffraction Symmetry Classification},
  author    = {Friedman, Edward G. and Baggett, Elizabeth J. and Shetty, Abhishek and
               Chan-Sew, Derrick and Acha, Vanellsa and Dwaracherla, Harshitha and
               Kienzle, Paul and Ratcliff, William},
  booktitle = {NeurIPS 2026 Track on Evaluations and Datasets},
  year      = {2026}
}
```
