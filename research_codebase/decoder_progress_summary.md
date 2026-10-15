# Decoder And Training Summary

## Current Setup

- Model backbone: ViT unchanged, `patch_size=25`
- Supervision:
  - `37`-bit multilabel head for crystallographic rules
  - auxiliary `99`-way extinction-group head
- Training data:
  - `/work2/ALLOC2/UserC/shared_dir/rruff_aug_10M_trainready_repacked.hdf5`
  - about `9,000,000` training spectra in the full split
  - only `70` of the `99` canonical extinction groups are present in RRUFF
- Distributed training target:
  - Vista `gh`
  - `2` nodes
  - `1` GH200 GPU per node

## What Changed

### Multilabel Formulation

The original `99`-way categorical extinction-group target was replaced by a structured `37`-bit target:

- `7` crystal-system bits
- `5` Bravais-lattice bits
- `25` operator bits

Each canonical extinction group is mapped to a fixed `37`-bit template in:

- [extinction_multilabel.py](/Users/UserE/project/Code/ViT_NVIDIA/extinction_multilabel.py)

### Hybrid Objective

The training objective now has two parts:

- multilabel `BCEWithLogitsLoss` on the `37` rule bits
- auxiliary `CrossEntropyLoss` on the extinction-group id

This was necessary because the pure multilabel decoder was learning useful rule structure, but was much weaker than direct EG classification.

### Decoder Upgrades

The offline decoder was extended to support:

- Euclidean template matching
- Bernoulli log-likelihood template scoring
- seen-class masking to the `70` EGs present in RRUFF
- hierarchical decoding using predicted crystal system and lattice
- Bayesian prior weighting using empirical train-set EG frequencies

Offline evaluation code:

- [evaluate_decoder_variants.py](/Users/UserE/project/Code/ViT_NVIDIA/evaluate_decoder_variants.py)

## Main Findings

### 3-Epoch Checkpoint, Same Measured Model

Checkpoint:

- `/scratch/ALLOC3/UserE/ai_diffraction_models/xrd_model_c6nuopyg_latest.pth`

Test set:

- `65536` samples

Results:

- `euclidean_all99`: top-1 `2.89%`, top-3 `4.34%`, top-5 `7.51%`
- `bernoulli_all99`: top-1 `2.89%`, top-3 `4.36%`, top-5 `10.52%`
- `bernoulli_seen70`: top-1 `3.62%`, top-3 `6.84%`, top-5 `16.42%`
- `hier_bernoulli_all99`: top-1 `2.85%`, top-3 `5.36%`, top-5 `13.25%`
- `hier_bernoulli_seen70`: top-1 `4.11%`, top-3 `10.53%`, top-5 `16.98%`
- `bayes025_hier_seen70`: top-1 `6.02%`, top-3 `21.17%`, top-5 `24.88%`
- `bayes050_hier_seen70`: top-1 `9.68%`, top-3 `25.16%`, top-5 `26.82%`
- `bayes100_hier_seen70`: top-1 `15.73%`, top-3 `25.71%`, top-5 `28.10%`
- `aux_head`: top-1 `20.59%`, top-3 `42.04%`, top-5 `56.44%`

### Interpretation

The decoder gap is real, but it is not just a training failure.

The auxiliary head is stronger because it learns:

- the joint extinction-group distribution
- the RRUFF geological class prior
- the fact that the `29` missing extinction groups should never be predicted

The pure multilabel head is weaker because standard BCE treats the `37` bits as independent, which allows physically inconsistent or high-entropy mixtures.

The Bayesian decoder showed that a large part of the gap was due to missing class priors in the template-matching stage. Adding the empirical log-prior raised template-decoded top-1 from `4.11%` to `15.73%` on the same checkpoint, without retraining.

## 10-Epoch Vista Run

Resumed run:

- job `632397`
- W&B: [smart-dragon-8](https://wandb.ai/ANONYMIZED-ENTITY/project/runs/co7a87po)
- partition: `gh`
- nodes: `c619-052,c634-082`
- state: `COMPLETED`
- elapsed: `00:13:08`

This run resumed from:

- `/scratch/ALLOC3/UserE/ai_diffraction_models/xrd_model_c6nuopyg_latest.pth`

and continued from epoch `4` through epoch `10`.

Final reported metrics from the training log:

- Validation Decode Accuracy:
  - epoch 4: `3.65%`
  - epoch 5: `3.02%`
  - epoch 6: `4.07%`
  - epoch 7: `3.76%`
  - epoch 8: `5.58%`
  - epoch 9: `4.57%`
  - epoch 10: `5.23%`
- Validation Aux EG Accuracy:
  - epoch 4: `18.26%`
  - epoch 5: `19.60%`
  - epoch 6: `20.25%`
  - epoch 7: `20.58%`
  - epoch 8: `20.70%`
  - epoch 9: `20.99%`
  - epoch 10: `21.22%`

Final test metrics:

- template decode top-1: `5.40%`
- template decode top-3: `6.90%`
- template decode top-5: `11.05%`
- auxiliary EG top-1: `21.74%`
- auxiliary EG top-3: `43.98%`
- auxiliary EG top-5: `57.96%`

## Where We Are

1. The infrastructure is working.
   - Vista environment, W&B, checkpointing, and 2-node DDP are all functioning.

2. The model is learning.
   - The auxiliary head is now above `21%` top-1 on the RRUFF test subset.

3. The decoder is the main bottleneck.
   - Pure template decode still trails the auxiliary head substantially.
   - Bayesian hierarchical decoding closes a large fraction of that gap offline.

4. More epochs helped, but they did not solve the decoder mismatch by themselves.
   - Going from `3` to `10` epochs improved template top-1 from about `2.9%` to `5.4%`.
   - The auxiliary head improved more modestly, from about `20.6%` to `21.7%`.

## Recommended Next Steps

1. Re-evaluate the `10`-epoch checkpoint with the Bayesian hierarchical decoder.
   - This is the most immediate next measurement.

2. The split-head formulation is now implemented for the next uniform-data runs:
   - system head: softmax + cross-entropy
   - lattice head: softmax + cross-entropy
   - operator head: sigmoid + BCE
   - auxiliary `99`-way EG head retained
   - current Vista config for the first uniform batch:
     - [config_uniform2m_vista_large.json](/Users/UserE/project/Code/ViT_NVIDIA/config_uniform2m_vista_large.json)

3. Next, train that split-head model on the uniform synthetic data once the first `2M` batch is repacked.

4. After that, revisit generated or streamed data to cover the missing `29` extinction groups and the missing rule bits at larger scale.

## Vista Status

As of the last check:

- no jobs were currently queued or running for user `UserE` on Vista
- the most recent large training job, `632397`, completed successfully
- the most recent decoder eval jobs, `632416` and `632449`, also completed successfully on `gh-dev`
