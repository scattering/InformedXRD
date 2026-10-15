#!/usr/bin/env python3
"""
rule_mask_v2.py

Targeted redesign driven by the v3 diagnostics (glide ~55% undetermined,
centering ~50%, high-index missing ~44%). Two changes:

1. RULE-FITTING classifiers instead of fixed-reflection reads. v3 returned
   UNDETERMINED whenever a *specific* diagnostic reflection was missing. v2 instead
   FITS the condition to whatever reflections survived:
     - glide: find the parity rule (none/axial/n/d) consistent with the observed
       present/absent in-plane reflections; UNDET only if nothing usable survived.
     - centering: fit P/one-face/I/F to ALL observed reflections (centering
       constrains every hkl, so redundancy is maximal) -> very robust.
   Serial stays as in v3 (it already survives, ~17% undetermined).

2. LOW-ORDER, HIGH-MULTIPLICITY probe set. Drops the fragile high-index families
   (4,2,0 / 3,3,0 / 4,4,0) that go missing first; keeps low-order zonal/general
   reflections whose orbits give redundant tests of the same condition.

Evaluated on the IDENTICAL bound-safe held-out/null scorer as v3, head-to-head
against v1 (= v3's encoding). Deps: numpy, gemmi (for the faithful test).
"""
from __future__ import annotations
import argparse, itertools, json
import numpy as np
from analyze_rule_mask_physical_v3 import (
    heldout_with_null, compare_masks_shared_folds, serial_category, physical_signature as v1_signature,
    six_signature, states_from_row, PRESENT, ABSENT, MISSING, RULE_REFLS as V1_REFLS,
    S_UNDET,
)

# glide codes (after orientation merge of axial_u/axial_v)
GV_UNDET, GV_NONE, GV_AXIAL, GV_N, GV_D, GV_PRESENT_UNDET = range(6)
# centering codes
CV_UNDET, CV_P, CV_ONEFACE, CV_I, CV_F, CV_OTHER = range(6)


def _perm_closure(base):
    s = set()
    for hkl in base:
        for p in itertools.permutations(range(3)):
            s.add((hkl[p[0]], hkl[p[1]], hkl[p[2]]))
    return sorted(s)


# low-order, high-multiplicity probe set (no 4,2,0 / 3,3,0 / 4,4,0 / 5,0,0)
RULE_BASE_V2 = [
    (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0), (6, 0, 0),   # serial period probes
    (1, 1, 0), (2, 1, 0), (2, 2, 0), (3, 1, 0), (3, 2, 0),   # low-order zonal (glide)
    (1, 1, 1), (2, 1, 1), (2, 2, 1), (3, 1, 1),              # low-order general (centering)
]
RULE_REFLS_V2 = _perm_closure(RULE_BASE_V2)


# --------------------------- rule-fitting glide ----------------------------- #
def glide_category_v2(states, zero_axis):
    ax = [a for a in range(3) if a != zero_axis]
    obs = []  # (u, v, state)
    for hkl, s in states.items():
        if hkl[zero_axis] != 0:
            continue
        u, v = hkl[ax[0]], hkl[ax[1]]
        if u == 0 and v == 0:
            continue
        if s in (PRESENT, ABSENT):
            obs.append((u, v, s))
    if not obs:
        return GV_UNDET
    rules = {
        "axial_u": lambda u, v: u % 2 == 0,
        "axial_v": lambda u, v: v % 2 == 0,
        "n":       lambda u, v: (u + v) % 2 == 0,
        "d":       lambda u, v: (u + v) % 4 == 0,
    }
    consistent = []
    for name, allowed in rules.items():
        ok = all((allowed(u, v) and s == PRESENT) or (not allowed(u, v) and s == ABSENT)
                 for (u, v, s) in obs)
        # require at least one ABSENT it explains (else indistinguishable from none)
        if ok and any(s == ABSENT for (_, _, s) in obs):
            consistent.append(name)
    has_absent = any(s == ABSENT for (_, _, s) in obs)
    if not has_absent:
        return GV_NONE                       # nothing forbidden observed
    if not consistent:
        return GV_PRESENT_UNDET              # glide present, type not pinned
    cset = set(consistent)
    if cset <= {"axial_u", "axial_v"}:
        return GV_AXIAL
    if cset == {"n"}:
        return GV_N
    if cset == {"d"}:
        return GV_D
    return GV_PRESENT_UNDET                   # multiple rules consistent -> ambiguous


# ------------------------- rule-fitting centering --------------------------- #
def centering_category_v2(states):
    obs = [(h, s) for h, s in states.items() if s in (PRESENT, ABSENT) and any(h)]
    if not obs:
        return CV_UNDET
    rules = {
        "P": lambda h: True,
        "A": lambda h: (h[1] + h[2]) % 2 == 0,
        "B": lambda h: (h[0] + h[2]) % 2 == 0,
        "C": lambda h: (h[0] + h[1]) % 2 == 0,
        "I": lambda h: (h[0] + h[1] + h[2]) % 2 == 0,
        "F": lambda h: (h[0] % 2 == h[1] % 2 == h[2] % 2),
    }
    consistent = [name for name, allowed in rules.items()
                  if all((allowed(h) and s == PRESENT) or (not allowed(h) and s == ABSENT)
                         for (h, s) in obs)]
    cset = set(consistent)
    has_absent = any(s == ABSENT for (_, s) in obs)
    if not has_absent:
        return CV_P if "P" in cset else CV_UNDET
    if not cset:
        return CV_OTHER
    if cset == {"F"}:
        return CV_F
    if cset == {"I"}:
        return CV_I
    if cset <= {"A", "B", "C"}:
        return CV_ONEFACE
    return CV_OTHER


def signature_v2(states):
    serials = tuple(sorted(serial_category(states, a) for a in range(3)))
    glides = tuple(sorted(glide_category_v2(states, a) for a in range(3)))
    return serials + glides + (centering_category_v2(states),)


def states_from_row_refls(row, refls):
    return {hkl: int(c) for hkl, c in zip(refls, np.asarray(row).tolist())}


# ----------------------------- comparison ---------------------------------- #
def _ids(sigs):
    d = {}
    for k in sigs:
        d.setdefault(k, len(d))
    return np.array([d[k] for k in sigs], np.int64), len(d)


def compare(labels, rows_v1, rows_v2, n_classes, n_boot=80):
    labels = np.asarray(labels, np.int64)
    st1 = [states_from_row(r) for r in rows_v1]
    st2 = [states_from_row_refls(r, RULE_REFLS_V2) for r in rows_v2]

    sig1 = [v1_signature(s) for s in st1]
    sig2 = [signature_v2(s) for s in st2]
    sym1, n1 = _ids(sig1); sym2, n2 = _ids(sig2)

    r1 = heldout_with_null(labels, sym1, n_classes, n_boot=n_boot)
    r2 = heldout_with_null(labels, sym2, n_classes, n_boot=n_boot)
    gain = compare_masks_shared_folds(labels, sym2, sym1, n_classes, n_boot=n_boot)

    # undetermined rates (v3 codes: G_UNDET==0, C_UNDET==0, S_UNDET==0)
    g1 = float(np.mean([0 in [c for c in p[3:6]] for p in sig1]))  # v3 G_UNDET==0
    c1 = float(np.mean([p[6] == 0 for p in sig1]))                 # v3 C_UNDET==0
    g2 = float(np.mean([GV_UNDET in p[3:6] for p in sig2]))
    c2 = float(np.mean([p[6] == CV_UNDET for p in sig2]))
    s1 = float(np.mean([S_UNDET in p[:3] for p in sig1]))
    s2 = float(np.mean([S_UNDET in p[:3] for p in sig2]))

    return {
        "v1_rule_mask": {"n_symbols": n1, **r1["heldout"], "null": r1["null"],
                         "serial_undet": s1, "glide_undet": g1, "centering_undet": c1},
        "v2_rule_mask": {"n_symbols": n2, **r2["heldout"], "null": r2["null"],
                         "serial_undet": s2, "glide_undet": g2, "centering_undet": c2},
        "gain_v2_over_v1": {
            "top1_pts": 100 * (r2["heldout"]["top1"] - r1["heldout"]["top1"]),
            "top5_pts": 100 * (r2["heldout"]["top5"] - r1["heldout"]["top5"]),
        },
        "gain_v2_over_v1_bootstrap": gain,
    }


def compare_npz(npz_path, rung, n_classes, n_boot):
    d = np.load(npz_path, allow_pickle=True)
    labels = np.asarray(d["labels"], np.int64)
    vocab_v1 = np.asarray(d[f"rule_vocab_{rung}"], dtype=object)
    sym_v1 = np.asarray(d[f"rule_symbols_{rung}"], np.int64)
    rows_v1 = np.array([np.asarray(vocab_v1[int(i)], dtype=np.int64) for i in sym_v1])

    vocab_v2 = np.asarray(d[f"rule_vocab_v2_{rung}"], dtype=object)
    sym_v2 = np.asarray(d[f"rule_symbols_v2_{rung}"], np.int64)
    rows_v2 = np.array([np.asarray(vocab_v2[int(i)], dtype=np.int64) for i in sym_v2])
    out = compare(labels, rows_v1, rows_v2, n_classes, n_boot=n_boot)
    out["rung"] = rung
    return out


def _faithful_test():
    import gemmi
    from extinction_family_analysis import load_map
    rng = np.random.default_rng(0)
    m = load_map(); NC = 99
    def clean_rows(refls):
        out = {}
        for sg in m["sg"].astype(int).unique():
            ops = gemmi.SpaceGroup(int(sg)).operations()
            out[int(sg)] = np.array([ABSENT if ops.is_systematically_absent(h) else PRESENT
                                     for h in refls], np.int64)
        return out
    clean1 = clean_rows(V1_REFLS); clean2 = clean_rows(RULE_REFLS_V2)
    eg_of = {int(r.sg): int(r.eg) for r in m.itertuples()}
    def pmiss(refls):
        mx = np.array([max(abs(x) for x in r) for r in refls])
        return np.clip(0.10 + 0.12 * (mx - 1), 0, 0.85)
    pm1, pm2 = pmiss(V1_REFLS), pmiss(RULE_REFLS_V2)
    sgs = list(clean1)
    labels, R1, R2 = [], [], []
    for _ in range(9900):
        sg = sgs[rng.integers(len(sgs))]
        r1 = clean1[sg].copy(); r1[rng.random(len(r1)) < pm1] = MISSING
        r2 = clean2[sg].copy(); r2[rng.random(len(r2)) < pm2] = MISSING
        labels.append(eg_of[sg] - 1); R1.append(r1); R2.append(r2)
    res = compare(np.array(labels), np.array(R1), np.array(R2), NC, n_boot=60)
    print(json.dumps(res, indent=2))
    v1, v2 = res["v1_rule_mask"], res["v2_rule_mask"]
    print(f"\nglide undet: v1 {v1['glide_undet']:.2f} -> v2 {v2['glide_undet']:.2f}")
    print(f"centering undet: v1 {v1['centering_undet']:.2f} -> v2 {v2['centering_undet']:.2f}")
    print(f"gain v2 over v1: Top-1 {res['gain_v2_over_v1']['top1_pts']:+.2f}pts, "
          f"Top-5 {res['gain_v2_over_v1']['top5_pts']:+.2f}pts")
    assert v2["top1"] >= v2["null"]["top1"] - 1e-9, "held-out >= null"
    assert v2["glide_undet"] <= v1["glide_undet"] + 1e-9, "v2 should not worsen glide undet"
    assert v2["centering_undet"] <= v1["centering_undet"] + 1e-9, "v2 should not worsen centering undet"
    print("OK: v2 reduces undetermined rates and stays bound-safe.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz"); ap.add_argument("--rung", default="background_noise_like")
    ap.add_argument("--all-rungs", action="store_true")
    ap.add_argument("--n-classes", type=int, default=99); ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--out-json", default="")
    a = ap.parse_args()
    if not a.npz:
        _faithful_test(); return
    if a.all_rungs:
        rung_names = ["ideal_mask", "structure_factor_like", "broadening_overlap_like", "rruff_like", "background_noise_like"]
        res = {r: compare_npz(a.npz, r, a.n_classes, a.n_boot) for r in rung_names}
    else:
        res = compare_npz(a.npz, a.rung, a.n_classes, a.n_boot)
    out = json.dumps(res, indent=2)
    print(out)
    if a.out_json:
        with open(a.out_json, "w") as f:
            f.write(out)


if __name__ == "__main__":
    main()
