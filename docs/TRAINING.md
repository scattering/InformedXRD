# Training

This public repository is a reviewer-facing evaluation artifact, not a training
release. It keeps the model and dataset code needed to load released checkpoints
and rerun evaluation, but it does not publish the private training campaign inputs
or launch wrappers.

For checkpoint evaluation:

```bash
conda env create -f environment-train-eval.yml
conda activate neurips-ai-diffraction-train-eval
pip install -e .
```

## Training Scope

This public artifact does not include:

- training launch wrappers
- private training configs
- initialization checkpoints that are not in the Hugging Face model repo
- large HDF5 trainready datasets

Reviewer-facing released filenames and roles are documented in:

- [checkpoint_manifest.csv](../reproducibility/checkpoint_manifest.csv)
- [dataset_manifest.csv](../reproducibility/dataset_manifest.csv)

The broader private training lineage used additional initialization checkpoints
and trainready HDF5 corpora that are not part of this NeurIPS artifact release.

## Included Code Surface

- [dataset.py](../src/paper_ai_diffraction/core/dataset.py)
- [model.py](../src/paper_ai_diffraction/core/model.py)
- [inference.py](../src/paper_ai_diffraction/core/inference.py)

The full training campaign provenance remains in the private archive.
