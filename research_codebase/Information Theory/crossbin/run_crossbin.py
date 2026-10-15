#!/usr/bin/env python3
"""
run_crossbin.py -- reproduce the cross-bin error decomposition for every model.

Reads the compact per-pattern prediction files in inputs/ (each a JSON list of
{"true_eg": int, "pred_eg": int}, EG ids 1..99) and reports, for each, the
fraction of wrong predictions that are cross-bin (absence-misread) vs within-bin
(Laue confusion), using the canonical gemmi-free absence-bin map.

    python run_crossbin.py                 # all inputs/*.json
    python run_crossbin.py inputs/foo.json # one file

Deps: numpy not required; standard library only.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from absence_bins import build_bin_map, decompose

HERE = Path(__file__).resolve().parent


def main(argv):
    bin_of = build_bin_map()
    files = [Path(a) for a in argv[1:]] or sorted((HERE / "inputs").glob("*.json"))
    out = {}
    print(f"{'model':<34}{'n':>5}{'corr':>6}{'wrong':>7}{'cross%':>9}{'within%':>9}")
    print("-" * 70)
    for f in files:
        rows = json.loads(Path(f).read_text())
        true = [r["true_eg"] for r in rows]
        pred = [r["pred_eg"] for r in rows]
        s = decompose(true, pred, bin_of)
        name = f.stem
        out[name] = s
        print(f"{name:<34}{s['n_total']:>5}{s['n_correct']:>6}{s['n_wrong']:>7}"
              f"{100*s['cross_bin_frac_of_wrong']:>8.1f}%{100*s['within_bin_frac_of_wrong']:>8.1f}%")
    fracs = [s["cross_bin_frac_of_wrong"] for s in out.values()]
    if fracs:
        print("-" * 70)
        print(f"cross-bin range across models: {100*min(fracs):.1f}% - {100*max(fracs):.1f}%")
    (HERE / "outputs").mkdir(exist_ok=True)
    (HERE / "outputs" / "crossbin_summary.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote outputs/crossbin_summary.json")


if __name__ == "__main__":
    main(sys.argv)
