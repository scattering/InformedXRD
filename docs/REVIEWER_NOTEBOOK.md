# Reviewer Support

This repo includes compact reviewer-facing assets under:

- `assets/reviewer_examples/`
- `results/reviewer/`

These are intended to support lightweight inspection of benchmark-derived examples
without redistributing the full benchmark HDF5 files.

Included reviewer assets:

- one correct-case CSV and paired metadata JSON
- one failure-case CSV and paired metadata JSON
- compact prior CSV/JSON
- compact precomputed `RRUFF-325` benchmark summary JSON

Reviewer notebooks:

- `notebooks/reviewer/reviewer_quickstart.ipynb`
  - inspects the bundled reviewer assets only
  - plots the two example traces
  - summarizes the precomputed `RRUFF-325` inference JSON
- `notebooks/reviewer/reviewer_full_evaluation_rruff325.ipynb`
  - runs a bounded end-to-end calibration evaluation on `RRUFF-325`
  - requires a released checkpoint plus local `RRUFF-325`
  - uses the bundled compact aggregate prior JSON by default

## Local Quickstart

```bash
conda env create -f environment-train-eval.yml
conda activate neurips-ai-diffraction-train-eval
pip install -e .
mkdir -p external/checkpoints
```

Place a compatible released checkpoint under `external/checkpoints/`, then open the
relevant reviewer, provenance, or evaluation notebook from the repo root.

## Notes

- This public repo does not redistribute the full benchmark HDF5 files.
- Reviewer-facing compact JSONs are included precisely so the benchmark logic can
  be inspected without requiring the full private data layer.
- `results/reviewer/ext_group_priors.json` is an aggregate extinction-group prior
  used for calibration. It contains no raw ICSD structures, raw diffraction
  records, formulas, atomic coordinates, or per-structure records. The raw
  trainready corpus used to derive it is not redistributed.
