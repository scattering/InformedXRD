# Hierarchy-metric reference distributions

Reference distributions for the hierarchy-aware DAG metric, computed on the
condensed *t*-subgroup graph (`assets/topology/extinction_group_adjacency.json`).
The generator is `analyze_topological_null.py`, which uses the graph utilities
in `src/paper_ai_diffraction/topology/analyze_topological_error_distance.py`.

## Uniform nulls (exact)

For each benchmark true label, enumerate all 98 incorrect extinction-group
predictions with equal weight (the failure JSON supplies only the benchmark
true labels; model predictions are not used). RRUFF-325: 31,850 wrong-label
pairs, mean distance 2.944, 30.65% within two hops. RRUFF-473: 46,354 pairs.

Reproduce from the repository root (`pip install numpy`):

```bash
PYTHONPATH=src/paper_ai_diffraction/topology python3 evaluation/nulls/analyze_topological_null.py \
  --graph-json assets/topology/extinction_group_adjacency.json \
  --failure-json results/mixed2500k_compare_325_failure_modes_655279.json \
  --model-key mixed2500k_655279_aux_bayes_t5 \
  --output-json rruff325_uniform_wrong_null.json
```

The committed `rruff325_uniform_wrong_null.json` reproduces exactly under this
command (mean, locality, directionality, and the full distance histogram).

## Model-matched (error-conditioned) nulls

Same enumeration restricted to the cases each model got wrong; add
`--condition-on-model-errors`. The `*_error_conditioned_null.json` files cover
the six XRDBench architectures and CNN checkpoints on RRUFF-473, CNN10 and
PatchTST on RRUFF-325, and the three RRUFF-325 case-study models:

```bash
PYTHONPATH=src/paper_ai_diffraction/topology python3 evaluation/nulls/analyze_topological_null.py \
  --graph-json assets/topology/extinction_group_adjacency.json \
  --failure-json results/po_compare_325_failure_modes_cscjfdwk.json \
  --model-key cscjfdwk_aux_bayes_t5 \
  --condition-on-model-errors \
  --output-json po_only_rruff325_uniform_wrong_error_conditioned_null.json
```

The committed model-matched nulls for the preferred-orientation-only ViT,
the final mixed-data ViT, and the epoch-6 CNN control were regenerated with
this command from the released failure-mode artifacts and match the paper's
reference-distribution table.

## Frequency-matched null and most-frequent-group reference

The frequency-matched null (wrong label drawn from the benchmark's own
extinction-group marginal) and the most-frequent-group predictor reference
are reported in the paper but their generator is not part of this script;
their artifacts will be added alongside it.
