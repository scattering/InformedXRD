#!/usr/bin/env python3
"""Enumerate the reflection conditions separating the hexagonal/trigonal
extinction-class pair P - - c and P - c -.

The paper's appendix on the hexagonal/trigonal pair states that, indexing
powder lines by the hexagonal invariant S = h^2 + h*k + k^2:

  * the P - c - absences fall on the S = 1, 4, 9, ... families at odd l
    (the h -h 0 l reflection family, c-glide perpendicular to <100>), and
  * the P - - c absences fall on the S = 3, 12, 27, ... families at odd l
    (the h h -2h l reflection family, c-glide perpendicular to <1-10>),

and that the two absence sets are the same pattern related by S -> 3S (the
30-degree basal rotation that rescales the in-plane cell edge by sqrt(3)).
This script makes that enumeration explicit and machine-checkable.

Class membership (space-group numbers):
  P - - c : 159, 163, 186, 190, 194
  P - c - : 158, 165, 185, 188, 193

For each class it enumerates all reflections (h, k, l) within a bound,
applies the class's systematic-absence condition, groups the absent
reflections by S, and verifies:
  1. P - c - absences occupy exactly the S = n^2 families (l odd);
  2. P - - c absences occupy exactly the S = 3 n^2 families (l odd);
  3. the map S -> 3S is a bijection between the two absence-family sets
     within the enumeration bound;
  4. the lowest-angle distinguishing reflections are 10.1 (S = 1, absent
     only in P - c -) and 11.1 (S = 3, absent only in P - - c).

Run:  python enumerate_hex_trig_conditions.py [--hk-max 6] [--l-max 6]
"""
import argparse
import json

P_DASH_DASH_C = [159, 163, 186, 190, 194]  # P - - c   (extinction group 73)
P_DASH_C_DASH = [158, 165, 185, 188, 193]  # P - c -   (extinction group 74)


def s_invariant(h: int, k: int) -> int:
    return h * h + h * k + k * k


def absent_p_dash_c_dash(h: int, k: int, l: int) -> bool:
    """c-glide perpendicular to <100>: h -h 0 l type reflections need l = 2n.

    In (h, k) coordinates the family is k == -h (equivalently, by the
    threefold symmetry, k == 0 or h == 0 with the remaining index free);
    absences are the members with odd l.
    """
    in_family = (k == -h) or (k == 0) or (h == 0)
    return in_family and (l % 2 == 1) and not (h == 0 and k == 0)


def absent_p_dash_dash_c(h: int, k: int, l: int) -> bool:
    """c-glide perpendicular to <1-10>: h h -2h l type reflections need l = 2n.

    In (h, k) coordinates the family is k == h (equivalently h == -2k or
    k == -2h under the threefold symmetry); absences are the members with odd l.
    """
    in_family = (k == h) or (h == -2 * k) or (k == -2 * h)
    return in_family and (l % 2 == 1) and not (h == 0 and k == 0)


def enumerate_absences(rule, hk_max: int, l_max: int) -> dict[int, list[tuple[int, int, int]]]:
    families: dict[int, list[tuple[int, int, int]]] = {}
    for h in range(-hk_max, hk_max + 1):
        for k in range(-hk_max, hk_max + 1):
            if h == 0 and k == 0:
                continue
            for l in range(1, l_max + 1):
                if rule(h, k, l):
                    families.setdefault(s_invariant(h, k), []).append((h, k, l))
    return families


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hk-max", type=int, default=6)
    ap.add_argument("--l-max", type=int, default=6)
    args = ap.parse_args()

    fam_pcd = enumerate_absences(absent_p_dash_c_dash, args.hk_max, args.l_max)
    fam_pdc = enumerate_absences(absent_p_dash_dash_c, args.hk_max, args.l_max)

    s_pcd = sorted(fam_pcd)
    s_pdc = sorted(fam_pdc)

    squares = [n * n for n in range(1, args.hk_max + 1)]
    expected_pcd = [s for s in squares if s <= max(s_pcd)]
    expected_pdc = [3 * s for s in squares if 3 * s <= max(s_pdc)]

    check_1 = s_pcd == expected_pcd
    check_2 = s_pdc == expected_pdc
    check_3 = all(3 * s in fam_pdc for s in s_pcd if 3 * s <= max(s_pdc))
    # Lowest-angle distinguishing reflections: 10.1 (S=1) vs 11.1 (S=3).
    check_4 = ((1, 0, 1) in fam_pcd.get(1, [])) and ((1, 1, 1) in fam_pdc.get(3, []))

    report = {
        "classes": {
            "P - - c": {"space_groups": P_DASH_DASH_C, "absent_S_families_odd_l": s_pdc},
            "P - c -": {"space_groups": P_DASH_C_DASH, "absent_S_families_odd_l": s_pcd},
        },
        "checks": {
            "P-c- absences are exactly S = n^2 (l odd)": check_1,
            "P--c absences are exactly S = 3 n^2 (l odd)": check_2,
            "S -> 3S maps P-c- families onto P--c families": check_3,
            "lowest distinguishing lines are 10.1 (P-c-) vs 11.1 (P--c)": check_4,
        },
        "example_absent_reflections": {
            "P - c - (S=1)": fam_pcd.get(1, [])[:6],
            "P - - c (S=3)": fam_pdc.get(3, [])[:6],
        },
        "enumeration_bounds": {"hk_max": args.hk_max, "l_max": args.l_max},
    }
    print(json.dumps(report, indent=2, default=str))
    assert all(report["checks"].values()), "one or more enumeration checks failed"


if __name__ == "__main__":
    main()
