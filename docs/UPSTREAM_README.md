# neurips-ai-diffraction

Public reproducibility repository for the NeurIPS-facing AI diffraction release.

This repo contains the cleaned code and compact artifacts needed to reproduce the
benchmarking, topology, and provenance layers of the transformer-based diffraction
work. It is intentionally limited to source, notebooks, small lookup assets, and
compact JSON outputs.

What is included:

- benchmark-construction and subset-building scripts for `RRUFF-473` and `RRUFF-325`
- core ViT model code
- calibration, split-validity, and topology analysis utilities
- compact result JSONs used for paper-facing summaries
- small bundled lookup and reviewer-support assets
- a compact aggregate extinction-group prior for calibration, with no raw ICSD
  structures or per-structure records

What is not included:

- model checkpoints
- large HDF5 training or benchmark datasets
- raw ICSD-derived structures or raw synthetic training corpora
- cluster-specific historical launchers and prior-work supplemental notebooks
- private operational paths and environment-specific local state

External artifact repos:

- Hugging Face dataset repo: `informedxrd/informedxrd-benchmarks`
- Hugging Face model repo: `informedxrd/informedxrd-checkpoints`

The reviewer-facing reproduction path is the reviewer notebooks, compact JSON
outputs, Hugging Face benchmark payloads, and the paper-facing evaluation and
figure scripts. Training campaign inputs and launch wrappers are not part of this
public artifact.

Checkpoints are expected under:

```text
external/checkpoints/
```

The expected filenames and roles are listed in:

- [reproducibility/checkpoint_manifest.csv](reproducibility/checkpoint_manifest.csv)

External dataset requirements and example filenames are listed in:

- [reproducibility/dataset_manifest.csv](reproducibility/dataset_manifest.csv)

The corresponding external artifact locations are:

- Dataset payloads and reviewer subset: [informedxrd/informedxrd-benchmarks](https://huggingface.co/datasets/informedxrd/informedxrd-benchmarks)
- Case-study checkpoints: [informedxrd/informedxrd-checkpoints](https://huggingface.co/informedxrd/informedxrd-checkpoints)

## Install

For figure/table regeneration and lightweight analysis:

```bash
conda env create -f environment.yml
conda activate neurips-ai-diffraction
pip install -e .
```

For checkpoint evaluation:

```bash
conda env create -f environment-train-eval.yml
conda activate neurips-ai-diffraction-train-eval
pip install -e .
```

## Key Docs

- [docs/BENCHMARKS.md](docs/BENCHMARKS.md)
- [docs/EVALUATION.md](docs/EVALUATION.md)
- [docs/FIGURES.md](docs/FIGURES.md)
- [docs/TRAINING.md](docs/TRAINING.md)
- [docs/REVIEWER_NOTEBOOK.md](docs/REVIEWER_NOTEBOOK.md)

## Repo Contract

- `src/` contains the public transformer/evaluation code surface.
- `results/` contains compact paper-backed JSON artifacts only.
- `assets/` contains small bundled lookups and reviewer examples only.
- checkpoint binaries and full HDF5 benchmarks stay external to the repo.
- training campaign inputs and launch wrappers are archived outside this public reviewer artifact.
