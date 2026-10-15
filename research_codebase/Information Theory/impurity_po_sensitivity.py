"""
impurity_po_sensitivity.py
==========================

Quick analytic test of the leading hypothesis for the floor/ceiling gap: that the
~20% mask-channel Bayes ceiling is too high because the simulator omits the two
nuisances that actually destroy ABSENCE information — impurity and preferred
orientation.

Rather than regenerate structures, we apply these as operators directly on the
EXISTING clean per-family masks and recompute the held-out Bayes ceiling vs flip
rate:

  * IMPURITY  flips a family that is currently `absent` -> `present` with prob p.
    This is the diagnostic killer: it drops a peak where the true phase is
    forbidden, erasing the EG signature (absent -> present).
  * PREFERRED ORIENTATION flips a family currently `present` -> `absent` with
    prob p. It suppresses allowed families (present -> absent erasure), which can
    mimic a higher-symmetry EG.

If a modest impurity/PO rate pulls the ceiling from ~20% down toward the real
cross-model ~15% (Schopmann) or ~10% (our model), that supports "the simulator
under-degrades; real models are near their true ceiling; architecture is not the
bottleneck." If the ceiling barely moves, the gap is more likely extraction loss
or headroom.

States: 0 = present, 1 = absent, 2 = unresolved (matching the mask encoding).
Input: labels (N,), masks (N, F) of per-family states, n_classes.
Deps: numpy.
"""
from __future__ import annotations
import numpy as np

PRESENT, ABSENT, UNRESOLVED = 0, 1, 2


def _encode(masks, n_states=3):
    F = masks.shape[1]
    w = n_states ** np.arange(F)
    return (masks * w).sum(axis=1)


def _bayes_topk_heldout(labels, symbols, n_classes, ks, alpha=None,
                        test_frac=0.5, n_boot=200, seed=0):
    labels = np.asarray(labels, np.int64); symbols = np.asarray(symbols, np.int64)
    n_sym = int(symbols.max()) + 1
    if alpha is None:
        alpha = 1.0 / n_sym
    rng = np.random.default_rng(seed); N = len(labels); kmax = max(ks)
    acc = {k: [] for k in ks}
    for _ in range(n_boot):
        perm = rng.permutation(N); cut = int(N * (1 - test_frac))
        tr, te = perm[:cut], perm[cut:]
        C = np.zeros((n_sym, n_classes)); np.add.at(C, (symbols[tr], labels[tr]), 1.0)
        C += alpha; post = C / C.sum(1, keepdims=True)
        pred = np.argsort(-post[symbols[te]], axis=1)[:, :kmax]
        for k in ks:
            acc[k].append((pred[:, :k] == labels[te][:, None]).any(1).mean())
    return {k: (float(np.mean(acc[k])), np.percentile(acc[k], [2.5, 97.5])) for k in ks}


def apply_impurity(masks, p, rng):
    m = masks.copy()
    flip = (m == ABSENT) & (rng.random(m.shape) < p)
    m[flip] = PRESENT
    return m


def apply_po(masks, p, rng):
    m = masks.copy()
    flip = (m == PRESENT) & (rng.random(m.shape) < p)
    m[flip] = ABSENT
    return m


def sensitivity(labels, clean_masks, n_classes, rates=(0.0, 0.05, 0.1, 0.2, 0.4),
                ks=(1, 5), seed=0):
    labels = np.asarray(labels, np.int64)
    clean_masks = np.asarray(clean_masks, np.int64)
    rng = np.random.default_rng(seed)
    for name, op in (("impurity (absent->present)", apply_impurity),
                     ("pref. orient (present->absent)", apply_po)):
        print(f"\n{name}:")
        print(f"  {'rate':>5} {'Top-1':>8} {'Top-5':>8}")
        for p in rates:
            m = op(clean_masks, p, rng) if p > 0 else clean_masks
            res = _bayes_topk_heldout(labels, _encode(m), n_classes, ks)
            t1 = res[1][0] * 100; t5 = res[5][0] * 100
            print(f"  {p:>5.2f} {t1:>7.1f}% {t5:>7.1f}%")


if __name__ == "__main__":
    # synthetic clean channel: informative per-family masks over 6 families.
    rng = np.random.default_rng(0)
    NC, F = 99, 6
    # each EG gets a characteristic per-family pattern (mostly absent, some present)
    base = rng.integers(0, 3, (NC, F))
    lab = np.repeat(np.arange(NC), 400)
    masks = base[lab].copy()
    # clean-channel noise so it's not deterministic
    n = rng.random(masks.shape) < 0.08
    masks[n] = rng.integers(0, 3, n.sum())
    print("Sensitivity of the mask-channel Bayes ceiling to absence-corrupting nuisances")
    print("(synthetic stand-in; run on your real clean per-family masks)")
    sensitivity(lab, masks, NC)
    print("\nRead: if a modest impurity/PO rate drags Top-1 from its clean value")
    print("down toward the real ~15%/~10%, the simulator was under-degrading and")
    print("architecture is likely not the bottleneck.")
