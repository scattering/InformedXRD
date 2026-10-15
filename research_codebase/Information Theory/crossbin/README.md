# Cross-bin error decomposition (reproducible)

Decomposes each model's wrong extinction-group predictions into:
- **cross-bin** (the absence pattern itself was misread -- measurement-limited), and
- **within-bin** (absences read correctly, wrong Laue class -- the irreducible floor).

EGs share an absence bin iff they have the same lattice + glide/screw operators
(same systematic absences), regardless of crystal system. The map is built from
the canonical extinction symbols (`absence_bins.py`), **standard library only --
no gemmi**, so it is environment-independent and reproducible.

## Run
```
python run_crossbin.py            # all inputs/*.json -> outputs/crossbin_summary.json
```

## Inputs
`inputs/*.json`: compact per-pattern `[{"true_eg": int, "pred_eg": int}, ...]`
(canonical EG ids 1..99) for each model on RRUFF-473. Derived from the posteriors
/ failure-mode artifacts archived on Zenodo.

## Result (RRUFF-473, cross-bin % of wrong predictions)
| model | cross-bin |
|---|---|
| our model (stage2c) | 100.0% |
| XRDBench CNN11 / BiGRU / Transformer | 100.0 / 99.0 / 99.5% |
| Schopmans ResNet-50 (sqrt F / T) | 100.0 / 99.5% |
| Schopmans ResNet-101 (sqrt F / T) | 96.7 / 93.6% |

Across every architecture, **93.6-100%** of errors are cross-bin: the dominant
failure is misreading the absence pattern, not Laue-class confusion.
