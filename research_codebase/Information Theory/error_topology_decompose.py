"""
error_topology_decompose.py
===========================

Decompose extinction-group classifier errors into the two mechanisms the
information-theory analysis distinguishes:

  * WITHIN-BIN  (true & predicted EG share the canonical absence pattern):
      the absence pattern was read correctly; the wrong call is a Laue-class
      confusion -> the IRREDUCIBLE, Laue-limited error (only point-group /
      full-pattern info can fix it).
  * CROSS-BIN   (true & predicted EG have different absence patterns):
      the absence pattern itself was misread -> the MEASUREMENT-limited error
      (resolution, counting statistics, overlap, impurity/PO).

Prediction from the noiseless ceilings: since a clean absence reading would give
~66% Top-1 but real models give ~13%, CROSS-BIN (measurement) should dominate the
error budget; the WITHIN-BIN fraction is the realized Laue-limited floor.

Also reports the crystal-system-change rate (a one-line proxy, since absence-twins
are always cross-system) and, if a directionality label is supplied per error,
cross-tabs mechanism against ancestor/descendant/branch-jump.

Usage:
    tag = ErrorTagger()                       # builds EG -> absence-bin/Laue/system
    summary = tag.decompose(true_egs, pred_egs)   # arrays of EG ids (1..99)

Deps: gemmi, numpy, pandas. Reuses extinction_family_analysis.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import numpy as np
import gemmi
from extinction_family_analysis import load_map, canonical_box_signature


class ErrorTagger:
    def __init__(self, nmax=8):
        m = load_map()
        egs = sorted(m["eg"].unique())
        box = {int(r.sg): canonical_box_signature(int(r.sg), nmax) for r in m.itertuples()}

        def collapse(d):
            out = {}
            for e in egs:
                sgs = m[m.eg == e]["sg"].astype(int).tolist()
                out[e] = Counter(d[s] for s in sgs).most_common(1)[0][0]
            return out

        self.absence = collapse(box)
        self.laue = collapse({int(s): gemmi.SpaceGroup(int(s)).laue_str() for s in m["sg"]})
        self.system = {e: sorted(m[m.eg == e]["system"].unique())[0] for e in egs}
        # absence-pattern -> integer bin id
        uniq = {s: i for i, s in enumerate(sorted(set(self.absence.values()),
                                                  key=lambda x: str(x)))}
        self.bin_of = {e: uniq[self.absence[e]] for e in egs}
        self.egs = egs

    def decompose(self, true_egs, pred_egs, directions=None):
        true_egs = list(true_egs); pred_egs = list(pred_egs)
        wrong = [(t, p) for t, p in zip(true_egs, pred_egs) if t != p]
        n_wrong = len(wrong)
        within = [(t, p) for t, p in wrong if self.bin_of[t] == self.bin_of[p]]
        cross = [(t, p) for t, p in wrong if self.bin_of[t] != self.bin_of[p]]
        sys_change = sum(1 for t, p in wrong if self.system[t] != self.system[p])
        laue_change = sum(1 for t, p in wrong if self.laue[t] != self.laue[p])

        out = {
            "n_total": len(true_egs),
            "n_wrong": n_wrong,
            "within_bin_Laue_limited": len(within),
            "within_bin_frac_of_wrong": len(within) / n_wrong if n_wrong else 0.0,
            "cross_bin_measurement_limited": len(cross),
            "cross_bin_frac_of_wrong": len(cross) / n_wrong if n_wrong else 0.0,
            "crystal_system_change_frac": sys_change / n_wrong if n_wrong else 0.0,
            "laue_class_change_frac": laue_change / n_wrong if n_wrong else 0.0,
        }
        if directions is not None:
            dirs = [d for (t, p), d in zip(zip(true_egs, pred_egs), directions) if t != p]
            ct = defaultdict(lambda: [0, 0])  # direction -> [within, cross]
            for (t, p), d in zip(wrong, dirs):
                ct[d][0 if self.bin_of[t] == self.bin_of[p] else 1] += 1
            out["by_direction"] = {d: {"within_bin": w, "cross_bin": c}
                                   for d, (w, c) in ct.items()}
        return out


if __name__ == "__main__":
    tag = ErrorTagger()
    n_bins = len(set(tag.bin_of.values()))
    in_collision = sum(1 for e in tag.egs
                       if list(tag.bin_of.values()).count(tag.bin_of[e]) > 1)
    print(f"EG -> absence-bin map ready: {len(tag.egs)} EGs, {n_bins} distinct absence bins,"
          f" {in_collision} EGs in collision bins.")
    print("Feed UserC's (true_eg, pred_eg) arrays to tag.decompose(...).\n")

    # synthetic demo: a model that mostly misreads absences (cross-bin),
    # with a minority of clean-absence/wrong-Laue (within-bin) errors.
    rng = np.random.default_rng(0)
    egs = tag.egs
    # collision-bin members, to inject within-bin errors
    binmap = defaultdict(list)
    for e in egs:
        binmap[tag.bin_of[e]].append(e)
    multi = {k: v for k, v in binmap.items() if len(v) > 1}
    true, pred = [], []
    for _ in range(473):
        t = int(rng.choice(egs)); true.append(t)
        r = rng.random()
        if r < 0.13:                       # correct
            pred.append(t)
        elif r < 0.13 + 0.15 and tag.bin_of[t] in multi:   # within-bin Laue confusion
            mates = [x for x in multi[tag.bin_of[t]] if x != t]
            pred.append(int(rng.choice(mates)))
        else:                              # cross-bin absence misread
            pred.append(int(rng.choice([x for x in egs if x != t])))
    s = tag.decompose(true, pred)
    print("demo decomposition (synthetic):")
    for k, v in s.items():
        print(f"  {k}: {round(v,4) if isinstance(v,float) else v}")
    assert s["within_bin_frac_of_wrong"] + s["cross_bin_frac_of_wrong"] == 1.0
    print("\nOK: tagger runs; replace synthetic arrays with UserC's true/pred EG pairs.")
