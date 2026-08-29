#!/usr/bin/env python3
"""
analyze_rule_mask_physical_v3.py

Fixes the cardinality problem in v2 (which still hit ~2000+ symbols on degraded
data) by handling the `missing` state correctly, and adds the modeling distinction
that actually matters under nuisance.

KEY CHANGE: `missing` is MARGINALIZED, not encoded.
  - a serial axis reads NONE only if order-1 is CONFIRMED present; a screw only if
    order-1 is CONFIRMED absent and a higher order is CONFIRMED present; if the
    diagnostic reflection is `missing`, the feature is a single shared UNDETERMINED
    value -- never a spurious period. (v2's bug: a dropped (1,0,0) made order-2
    look like a 2_1 screw, cataloguing dropout as symmetry.)

TWO MODES:
  - per-pattern  : classify each degraded pattern's own signature. Honest,
    leakage-free, held-out with matched-fold null. THE operational number.
  - aggregated   : pool a group's patterns into a confirmed present/absent
    consensus, then extract categories. This pools BY EG LABEL, so it is a LEAKY
    ORACLE -- it measures whether the rule families are recoverable IN PRINCIPLE
    when you don't read them off a single shot, NOT a fair classifier accuracy.
    Reported as an exact ceiling, explicitly labeled.

Also runs the six-prototype mask through the identical scorer, and reports the
diagnostics that explain the result: per-feature UNDETERMINED rate and the
`missing` rate on the high-index families the rule mask depends on.

States: 0 present, 1 absent, 2 missing. Deps: numpy.
"""
from __future__ import annotations
import argparse, itertools, json
from pathlib import Path
import numpy as np

PRESENT, ABSENT, MISSING = 0, 1, 2


def _perm_closure(base):
    s = set()
    for hkl in base:
        for p in itertools.permutations(range(3)):
            s.add((hkl[p[0]], hkl[p[1]], hkl[p[2]]))
    return sorted(s)


RULE_BASE = [
    (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0), (5, 0, 0), (6, 0, 0),
    (1, 1, 0), (2, 1, 0), (2, 2, 0), (3, 1, 0), (4, 2, 0), (3, 3, 0), (4, 4, 0),
    (1, 1, 1), (2, 1, 1), (2, 2, 1), (3, 2, 1),
]
RULE_REFLS = _perm_closure(RULE_BASE)
SIX = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)]

# serial category codes
S_UNDET, S_NONE, S2, S3, S4, S6, S_OTHER = range(7)
# glide category codes
G_UNDET, G_NONE, G_N, G_NONE_OR_N, G_AXIAL, G_D, G_PRESENT_UNDET = range(7)
# centering codes
C_UNDET, C_P, C_I, C_F, C_C, C_OTHER = range(6)


def _consensus(states, hkls):
    """min over symmetric reflections: present if any present, else absent if any
    absent, else missing."""
    vals = [states[h] for h in hkls if h in states]
    if not vals:
        return MISSING
    if PRESENT in vals:
        return PRESENT
    if ABSENT in vals:
        return ABSENT
    return MISSING


def serial_category(states, axis):
    def st(n):
        h = [0, 0, 0]; h[axis] = n
        return states.get(tuple(h), MISSING)
    s1 = st(1)
    if s1 == MISSING:
        return S_UNDET
    if s1 == PRESENT:
        return S_NONE                      # order-1 present -> no screw
    for n, code in [(2, S2), (3, S3), (4, S4), (6, S6)]:
        if st(n) == PRESENT:
            return code                    # smallest confirmed-present order = period
    return S_OTHER                         # screw present, period unresolved


def glide_category(states, zero_axis):
    ax = [a for a in range(3) if a != zero_axis]
    def ip(u, v):
        h = [0, 0, 0]; h[ax[0]] = u; h[ax[1]] = v
        return tuple(h)
    s11 = _consensus(states, [ip(1, 1)])
    s21 = _consensus(states, [ip(2, 1), ip(1, 2)])
    if s11 == MISSING:
        return G_UNDET
    if s11 == PRESENT:
        if s21 == MISSING:
            return G_NONE_OR_N             # present but type undetermined
        return G_NONE if s21 == PRESENT else G_N
    # s11 absent -> glide present
    if s21 == MISSING:
        return G_PRESENT_UNDET
    return G_AXIAL if s21 == PRESENT else G_D


def centering_category(states):
    s111 = _consensus(states, [(1, 1, 1)])
    s211 = _consensus(states, [(2, 1, 1), (1, 2, 1), (1, 1, 2)])
    s221 = _consensus(states, [(2, 2, 1), (2, 1, 2), (1, 2, 2)])
    if s111 == MISSING or s211 == MISSING:
        return C_UNDET
    if (s111, s211) == (PRESENT, PRESENT):
        return C_P
    if s111 == ABSENT and s211 == PRESENT:
        return C_I
    if s111 == PRESENT and s211 == ABSENT:
        if s221 == ABSENT:
            return C_F
        if s221 == PRESENT:
            return C_C
        return C_OTHER
    return C_OTHER


def physical_signature(states):
    serials = tuple(sorted(serial_category(states, a) for a in range(3)))
    glides = tuple(sorted(glide_category(states, a) for a in range(3)))
    return serials + glides + (centering_category(states),)


def states_from_row(row):
    return {hkl: int(c) for hkl, c in zip(RULE_REFLS, np.asarray(row).tolist())}


def six_signature(states):
    return tuple(states.get(h, MISSING) for h in SIX)


# ----------------------- bound-safe held-out + null ------------------------- #
def heldout_with_null(labels, symbols, n_classes, alpha=1.0, test_frac=0.5,
                      n_boot=200, ks=(1, 5), seed=0):
    labels = np.asarray(labels, np.int64); symbols = np.asarray(symbols, np.int64)
    n_sym = int(symbols.max()) + 1 if len(symbols) else 1
    rng = np.random.default_rng(seed); N = len(labels); kmax = max(ks)

    def run(lab):
        accs = {k: [] for k in ks}
        for _ in range(n_boot):
            perm = rng.permutation(N); cut = int(N * (1 - test_frac))
            tr, te = perm[:cut], perm[cut:]
            p_tr = np.bincount(lab[tr], minlength=n_classes).astype(float)
            p_tr = p_tr / p_tr.sum()
            C = np.zeros((n_sym, n_classes))
            np.add.at(C, (symbols[tr], lab[tr]), 1.0)
            post = C + alpha * p_tr[None, :]
            post /= post.sum(1, keepdims=True)
            pred = np.argsort(-post[symbols[te]], axis=1)[:, :kmax]
            for k in ks:
                accs[k].append((pred[:, :k] == lab[te][:, None]).any(1).mean())
        return accs

    obs_samples = run(labels)
    null_samples = run(rng.permutation(labels))
    out = {
        "n_symbols": int(n_sym),
        "heldout": {},
        "null": {},
        "lift_points": {},
        "lift_ci95_points": {},
    }
    for k in ks:
        obs = np.asarray(obs_samples[k], dtype=np.float64)
        null = np.asarray(null_samples[k], dtype=np.float64)
        lift = 100.0 * (obs - null)
        out["heldout"][f"top{k}"] = float(obs.mean())
        out["null"][f"top{k}"] = float(null.mean())
        out["lift_points"][f"top{k}"] = float(lift.mean())
        out["lift_ci95_points"][f"top{k}"] = [float(x) for x in np.percentile(lift, [2.5, 97.5])]
    return out


def compare_masks_shared_folds(labels, symbols_a, symbols_b, n_classes, alpha=1.0,
                               test_frac=0.5, n_boot=200, ks=(1, 5), seed=0):
    labels = np.asarray(labels, np.int64)
    symbols_a = np.asarray(symbols_a, np.int64)
    symbols_b = np.asarray(symbols_b, np.int64)
    n_sym_a = int(symbols_a.max()) + 1 if len(symbols_a) else 1
    n_sym_b = int(symbols_b.max()) + 1 if len(symbols_b) else 1
    rng = np.random.default_rng(seed)
    N = len(labels)
    kmax = max(ks)

    heldout_diff = {k: [] for k in ks}
    lift_diff = {k: [] for k in ks}

    for _ in range(n_boot):
        perm = rng.permutation(N)
        cut = int(N * (1 - test_frac))
        tr, te = perm[:cut], perm[cut:]
        p_tr = np.bincount(labels[tr], minlength=n_classes).astype(float)
        p_tr = p_tr / p_tr.sum()
        shuffled = rng.permutation(labels)

        def topk_samples(symbols, n_sym, lab_train, lab_test):
            C = np.zeros((n_sym, n_classes))
            np.add.at(C, (symbols[tr], lab_train[tr]), 1.0)
            post = C + alpha * p_tr[None, :]
            post /= post.sum(1, keepdims=True)
            pred = np.argsort(-post[symbols[te]], axis=1)[:, :kmax]
            return {
                k: (pred[:, :k] == lab_test[te][:, None]).any(1).mean()
                for k in ks
            }

        obs_a = topk_samples(symbols_a, n_sym_a, labels, labels)
        obs_b = topk_samples(symbols_b, n_sym_b, labels, labels)
        null_a = topk_samples(symbols_a, n_sym_a, shuffled, shuffled)
        null_b = topk_samples(symbols_b, n_sym_b, shuffled, shuffled)

        for k in ks:
            heldout_diff[k].append(100.0 * (obs_a[k] - obs_b[k]))
            lift_diff[k].append(100.0 * ((obs_a[k] - null_a[k]) - (obs_b[k] - null_b[k])))

    out = {"heldout_gain_points": {}, "heldout_gain_ci95_points": {}, "lift_gain_points": {}, "lift_gain_ci95_points": {}}
    for k in ks:
        held = np.asarray(heldout_diff[k], dtype=np.float64)
        lift = np.asarray(lift_diff[k], dtype=np.float64)
        out["heldout_gain_points"][f"top{k}"] = float(held.mean())
        out["heldout_gain_ci95_points"][f"top{k}"] = [float(x) for x in np.percentile(held, [2.5, 97.5])]
        out["lift_gain_points"][f"top{k}"] = float(lift.mean())
        out["lift_gain_ci95_points"][f"top{k}"] = [float(x) for x in np.percentile(lift, [2.5, 97.5])]
    return out


def exact_oracle(eg_to_sig, n_classes, prior=None):
    """Deterministic EG->signature: exact distinct count and Top-k ceiling
    (uniform prior unless given). This is the LEAKY-ORACLE recoverability bound."""
    from collections import Counter
    egs = sorted(eg_to_sig)
    bins = Counter(eg_to_sig.values())
    N = len(egs)
    top1 = sum(min(1, b) for b in bins.values()) / N
    top5 = sum(min(5, b) for b in bins.values()) / N
    return {"n_distinct": len(bins), "top1": top1, "top5": top5}


# --------------------------------------------------------------------------- #
def analyze(labels, rows_per_sample, n_classes, n_boot=200):
    """rows_per_sample: (N, n_refls) array of states for each sample."""
    labels = np.asarray(labels, np.int64)
    states = [states_from_row(r) for r in rows_per_sample]

    # ---- per-pattern (honest) ----
    phys = [physical_signature(s) for s in states]
    six = [six_signature(s) for s in states]
    def to_ids(sigs):
        ids = {}
        for k in sigs:
            ids.setdefault(k, len(ids))
        return np.array([ids[k] for k in sigs], np.int64), len(ids)
    phys_sym, n_phys = to_ids(phys)
    six_sym, n_six = to_ids(six)

    pp_rule = heldout_with_null(labels, phys_sym, n_classes, n_boot=n_boot)
    pp_six = heldout_with_null(labels, six_sym, n_classes, n_boot=n_boot)
    paired_gain = compare_masks_shared_folds(labels, phys_sym, six_sym, n_classes, n_boot=n_boot)

    # ---- aggregated leaky oracle (pool by EG) ----
    eg_rows = {}
    for e, r in zip(labels, rows_per_sample):
        eg_rows.setdefault(int(e), []).append(np.asarray(r))
    def consensus_states(rows):
        arr = np.array(rows)
        cons = np.full(arr.shape[1], MISSING)
        cons[(arr == ABSENT).any(0)] = ABSENT
        cons[(arr == PRESENT).any(0)] = PRESENT
        return {hkl: int(c) for hkl, c in zip(RULE_REFLS, cons)}
    eg_phys = {e: physical_signature(consensus_states(rs)) for e, rs in eg_rows.items()}
    eg_six = {e: six_signature(consensus_states(rs)) for e, rs in eg_rows.items()}
    agg_rule = exact_oracle(eg_phys, n_classes)
    agg_six = exact_oracle(eg_six, n_classes)

    # ---- diagnostics ----
    undet = {
        "serial_undet_rate": float(np.mean([S_UNDET in p[:3] for p in phys])),
        "glide_undet_rate": float(np.mean([G_UNDET in p[3:6] for p in phys])),
        "centering_undet_rate": float(np.mean([p[6] == C_UNDET for p in phys])),
    }
    arr = np.array(rows_per_sample)
    maxidx = np.array([max(abs(h) for h in r) for r in RULE_REFLS])
    hi = maxidx >= 4
    miss_hi = float((arr[:, hi] == MISSING).mean())
    miss_lo = float((arr[:, ~hi] == MISSING).mean())

    return {
        "n_samples": int(len(labels)),
        "per_pattern": {
            "rule_mask": {**pp_rule},
            "six_family": {**pp_six},
            "rule_minus_six": paired_gain,
        },
        "aggregated_leaky_oracle": {
            "_caveat": "pools by EG label; in-principle recoverability, NOT a fair classifier accuracy",
            "rule_mask": agg_rule,
            "six_family": agg_six,
        },
        "diagnostics": {
            "rule_feature_undetermined_rates": undet,
            "missing_rate_high_index_families(|hkl|>=4)": miss_hi,
            "missing_rate_low_index_families": miss_lo,
        },
    }


def analyze_npz(npz_path, n_classes, rung, n_boot):
    d = np.load(npz_path, allow_pickle=True)
    labels = np.asarray(d["labels"], np.int64)
    vocab = np.asarray(d[f"rule_vocab_{rung}"], dtype=object)
    sample_symbols = np.asarray(d[f"rule_symbols_{rung}"], np.int64)
    rows = np.array([np.asarray(vocab[int(i)], dtype=np.int64) for i in sample_symbols])
    res = analyze(labels, rows, n_classes, n_boot=n_boot)
    res["rung"] = rung
    return res


def _self_test():
    """Faithful test: clean rule-mask rows from REAL space-group symmetry (gemmi),
    mapped to EG, then degraded with index-dependent dropout. Checks the properties
    that matter: scorer bound-safety, and that aggregation recovers clean structure.
    Per-pattern cardinality is reported, not asserted (it reflects dropout combinatorics)."""
    import gemmi
    from extinction_family_analysis import load_map
    rng = np.random.default_rng(0)
    m = load_map()
    NC = 99
    nref = len(RULE_REFLS)
    # clean absence row per SG (0 present, 1 absent), via symmetry
    clean = {}
    for sg in m["sg"].astype(int).unique():
        ops = gemmi.SpaceGroup(int(sg)).operations()
        clean[int(sg)] = np.array([ABSENT if ops.is_systematically_absent(h) else PRESENT
                                   for h in RULE_REFLS], dtype=np.int64)
    eg_of = {int(r.sg): int(r.eg) for r in m.itertuples()}
    maxidx = np.array([max(abs(h) for h in r) for r in RULE_REFLS])
    pmiss = np.clip(0.10 + 0.12 * (maxidx - 1), 0, 0.85)  # high-index dropped more
    sgs = list(clean)
    labels, rows = [], []
    for _ in range(9900):
        sg = sgs[rng.integers(len(sgs))]
        row = clean[sg].copy()
        row[rng.random(nref) < pmiss] = MISSING          # dropout only (no flips: absences are robust)
        labels.append(eg_of[sg] - 1); rows.append(row)   # EG 1..99 -> 0..98
    res = analyze(np.array(labels), np.array(rows), NC, n_boot=40)
    print(json.dumps(res, indent=2))
    n_phys = res["per_pattern"]["rule_mask"]["n_symbols"]
    agg = res["aggregated_leaky_oracle"]["rule_mask"]["n_distinct"]
    print(f"\nper-pattern rule-mask symbols: {n_phys}")
    print(f"aggregated (pooled) rule-mask distinct EG-signatures: {agg} (clean ceiling ~63)")
    for mk in ("rule_mask", "six_family"):
        r = res["per_pattern"][mk]
        assert r["heldout"]["top1"] >= r["null"]["top1"] - 1e-9, "held-out >= null"
    assert agg >= 40, "aggregation should recover most of the clean ~63-class structure"
    print("OK: held-out >= null (both masks); aggregation recovers clean structure.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz"); ap.add_argument("--rung", default="background_noise_like")
    ap.add_argument("--n-classes", type=int, default=99)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--all-rungs", action="store_true")
    ap.add_argument("--out-json", default="")
    a = ap.parse_args()
    if not a.npz:
        _self_test(); return
    if a.all_rungs:
        rung_names = ["ideal_mask", "structure_factor_like", "broadening_overlap_like", "rruff_like", "background_noise_like"]
        res = {r: analyze_npz(a.npz, a.n_classes, r, a.n_boot) for r in rung_names}
    else:
        res = analyze_npz(a.npz, a.n_classes, a.rung, a.n_boot)
    out = json.dumps(res, indent=2); print(out)
    if a.out_json:
        Path(a.out_json).write_text(out)


if __name__ == "__main__":
    main()
