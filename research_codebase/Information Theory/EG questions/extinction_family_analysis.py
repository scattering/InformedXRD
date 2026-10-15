"""
extinction_family_analysis.py
==============================

Two no-sampling, no-window analyses built directly from space-group symmetry
(gemmi systematic-absence tests), to answer:

  (A) SIX-FAMILY FAILURE SUMMARY. Why does the implemented six-prototype mask
      (reflections 100,010,001,110,101,011) fail to be a canonical EG invariant?
      For each space group we compute the exact six-bit absence signature, group
      by extinction group (EG), and find which EGs are internally inconsistent
      (their member SGs disagree). We stratify the inconsistent EGs by crystal
      system and identify the culprit reflections. This reproduces, from first
      principles, the ~57-inconsistent-EG finding and explains it.

  (B) CANONICAL RULE-LEVEL CHANNEL. Instead of six fixed low-index reflections,
      we use the full systematic-absence pattern over a box of reflections,
      made orientation-invariant by canonicalizing over the 6 axis permutations
      (so different settings of the same EG map together). This is the EG-defining
      signature. We verify it is (nearly) constant within each EG and count how
      many of the 99 EGs it separates -- the clean design reference against which
      the six-family mask is a lossy, brittle projection.

Inputs: FINAL_SPG_ExtG_CrysS_Table.csv (SG -> EG -> crystal system).
Deps: gemmi, numpy, pandas.
"""
from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
import gemmi

UPLOADS = "/mnt/user-data/uploads"
OUT = "/mnt/user-data/outputs"

SIX = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)]
SIX_NAMES = ["100", "010", "001", "110", "101", "011"]


# --------------------------------------------------------------------------- #
# symmetry helpers
# --------------------------------------------------------------------------- #
def sg_absence_fn(sg_number):
    ops = gemmi.SpaceGroup(sg_number).operations()
    return lambda hkl: ops.is_systematically_absent(hkl)


def six_family_signature(sg_number):
    f = sg_absence_fn(sg_number)
    return tuple(int(f(hkl)) for hkl in SIX)


def canonical_box_signature(sg_number, nmax=6):
    """Orientation-invariant absence signature over a reflection box,
    canonicalized over the 6 axis permutations (lexicographically minimal)."""
    f = sg_absence_fn(sg_number)
    rng = range(0, nmax + 1)
    box = [(h, k, l) for h in rng for k in rng for l in rng if (h or k or l)]
    # absence dict
    absent = {hkl: int(f(hkl)) for hkl in box}
    best = None
    for perm in itertools.permutations(range(3)):
        sig = tuple(absent[(hkl[perm[0]], hkl[perm[1]], hkl[perm[2]])] for hkl in box)
        if best is None or sig < best:
            best = sig
    return best


# --------------------------------------------------------------------------- #
# load mapping
# --------------------------------------------------------------------------- #
def load_map():
    df = pd.read_csv(f"{UPLOADS}/FINAL_SPG_ExtG_CrysS_Table.csv")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"Space Group": "sg", "Extinction Group": "eg",
                            "Crystal System": "system"})
    return df[["sg", "eg", "system"]]


def entropy_bits(p):
    p = np.asarray(p, float); p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def channel_stats(eg_to_symbol, prior=None):
    """eg_to_symbol: dict EG -> hashable signature. Returns MI / distinct / collisions
    under a prior over EGs (default uniform)."""
    egs = sorted(eg_to_symbol)
    syms = [eg_to_symbol[e] for e in egs]
    uniq = {s: i for i, s in enumerate(sorted(set(syms), key=lambda x: (len(str(x)), str(x))))}
    inv = np.array([uniq[s] for s in syms])
    n = len(egs)
    if prior is None:
        prior = np.full(n, 1.0 / n)
    prior = np.asarray(prior, float); prior /= prior.sum()
    pm = np.zeros(len(uniq)); np.add.at(pm, inv, prior)
    H_M = entropy_bits(pm); H_G = entropy_bits(prior)
    collisions = {}
    for s, i in uniq.items():
        members = [egs[j] for j in range(n) if inv[j] == i]
        if len(members) > 1:
            collisions[i] = members
    return {"n_eg": n, "n_distinct": len(uniq), "H_G_bits": H_G,
            "MI_bits": H_M, "H_G_given_M_bits": H_G - H_M,
            "effective_candidates": float(2 ** (H_G - H_M)),
            "n_colliding_eg": sum(len(v) for v in collisions.values()),
            "collisions": collisions}


# --------------------------------------------------------------------------- #
def main():
    m = load_map()
    egs = sorted(m["eg"].unique())
    sys_of_eg = {e: sorted(m[m.eg == e]["system"].unique()) for e in egs}

    # precompute signatures per SG
    six_by_sg = {int(r.sg): six_family_signature(int(r.sg)) for r in m.itertuples()}
    canon_by_sg = {int(r.sg): canonical_box_signature(int(r.sg)) for r in m.itertuples()}

    # ---- (A) six-family consistency ----
    six_inconsistent = []
    six_eg_majority = {}
    for e in egs:
        sgs = m[m.eg == e]["sg"].astype(int).tolist()
        sigs = [six_by_sg[s] for s in sgs]
        from collections import Counter
        c = Counter(sigs)
        maj_sig, maj_n = c.most_common(1)[0]
        six_eg_majority[e] = maj_sig
        maj_frac = maj_n / len(sigs)
        if len(c) > 1:
            # culprit families: which of the 6 vary across member SGs
            arr = np.array(sigs)
            culprits = [SIX_NAMES[j] for j in range(6) if len(set(arr[:, j])) > 1]
            six_inconsistent.append({
                "eg": e, "crystal_system": "/".join(sys_of_eg[e]),
                "n_sgs": len(sgs), "n_distinct_six_sig": len(c),
                "majority_fraction": round(maj_frac, 3),
                "culprit_reflections": ",".join(culprits)})

    six_stats = channel_stats(six_eg_majority)

    # ---- (B) canonical consistency ----
    canon_inconsistent = []
    canon_eg_symbol = {}
    for e in egs:
        sgs = m[m.eg == e]["sg"].astype(int).tolist()
        sigs = {canon_by_sg[s] for s in sgs}
        # represent EG by the (single) canonical sig if consistent, else the min
        canon_eg_symbol[e] = min(sigs)
        if len(sigs) > 1:
            canon_inconsistent.append({"eg": e,
                                       "crystal_system": "/".join(sys_of_eg[e]),
                                       "n_distinct_canon_sig": len(sigs)})
    canon_stats = channel_stats(canon_eg_symbol)

    # ---- report ----
    print("=" * 70)
    print("(A) SIX-FAMILY MASK  (reflections 100 010 001 110 101 011)")
    print("=" * 70)
    print(f"  EGs total                         : {len(egs)}")
    print(f"  EGs internally INCONSISTENT       : {len(six_inconsistent)} / {len(egs)}")
    print(f"  distinct majority patterns        : {six_stats['n_distinct']}")
    print(f"  MI (uniform prior)                : {six_stats['MI_bits']:.3f} bits")
    print(f"  effective candidates (majority)   : {six_stats['effective_candidates']:.2f}")
    print(f"  EGs in collision bins             : {six_stats['n_colliding_eg']}")
    if six_inconsistent:
        fail = pd.DataFrame(six_inconsistent)
        print("\n  inconsistent EGs by crystal system:")
        by_sys = fail.groupby("crystal_system").size().sort_values(ascending=False)
        tot = m.groupby("system")["eg"].nunique()
        for syst, cnt in by_sys.items():
            base = tot.get(syst.split("/")[0], "?")
            print(f"    {syst:14} {cnt:3d} inconsistent  (of ~{base} EGs in system)")
        print("\n  most-frequent culprit reflections among inconsistent EGs:")
        allc = ",".join(fail["culprit_reflections"]).split(",")
        for refl, cnt in Counter([x for x in allc if x]).most_common():
            print(f"    {refl}: {cnt}")

    print("\n" + "=" * 70)
    print("(B) CANONICAL RULE-LEVEL SIGNATURE  (full absence box, axis-canonical)")
    print("=" * 70)
    print(f"  EGs total                         : {len(egs)}")
    print(f"  EGs INCONSISTENT under canonical  : {len(canon_inconsistent)} / {len(egs)}")
    print(f"  distinct canonical patterns       : {canon_stats['n_distinct']}")
    print(f"  MI (uniform prior)                : {canon_stats['MI_bits']:.3f} bits  "
          f"(H_G = {canon_stats['H_G_bits']:.3f})")
    print(f"  effective candidates              : {canon_stats['effective_candidates']:.2f}")
    print(f"  EGs in collision bins             : {canon_stats['n_colliding_eg']}")
    if canon_stats["collisions"]:
        print("  residual canonical collisions (genuinely absence-identical EGs):")
        for i, members in list(canon_stats["collisions"].items())[:10]:
            print(f"    EGs {members}")

    print("\n" + "=" * 70)
    print("DECOMPOSITION")
    print("=" * 70)
    print(f"  pigeonhole bound (6 binary families)   : <= 64 patterns for 99 EGs")
    print(f"  six-family realized (majority)         : {six_stats['n_distinct']} patterns, "
          f"~{six_stats['effective_candidates']:.1f} eff. candidates")
    print(f"  canonical full-absence signature       : {canon_stats['n_distinct']} patterns, "
          f"~{canon_stats['effective_candidates']:.1f} eff. candidates")
    print("  => the six low-index prototypes are a lossy AND non-invariant projection;")
    print("     the canonical absence pattern is the EG-defining (near-injective) object.")

    # ---- save ----
    if six_inconsistent:
        pd.DataFrame(six_inconsistent).sort_values(
            ["crystal_system", "majority_fraction"]).to_csv(
            f"{OUT}/six_family_failure_summary.csv", index=False)
    rows = [{"eg": e, "crystal_system": "/".join(sys_of_eg[e]),
             "canonical_signature_hash": hash(canon_eg_symbol[e]) & 0xFFFFFFFF}
            for e in egs]
    pd.DataFrame(rows).to_csv(f"{OUT}/canonical_eg_signatures.csv", index=False)
    print(f"\nsaved: six_family_failure_summary.csv, canonical_eg_signatures.csv")


if __name__ == "__main__":
    main()
