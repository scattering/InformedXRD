#!/usr/bin/env python3
"""Approximate RRUFF -> OrgA ICSD matching by name, space group, and loose cell consistency."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


SPECIES_SUFFIXES = (
    "ce", "la", "nd", "sm", "eu", "gd", "tb", "dy", "er", "yb", "lu", "y",
    "fe", "mg", "mn", "zn", "ca", "na", "k", "li", "cu", "ni", "co", "al",
    "ti", "pb", "ag", "sr", "ba", "as", "oh", "cl", "f",
)


def normalize_name(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def canonical_name_forms(text: str, known_names: set[str]) -> list[str]:
    base = normalize_name(text)
    forms = [base]
    for suffix in SPECIES_SUFFIXES:
        if base.endswith(suffix) and len(base) > len(suffix) + 3:
            stripped = base[: -len(suffix)]
            if stripped in known_names:
                forms.append(stripped)
    return list(dict.fromkeys(forms))


def normalize_sg_alias(text: str) -> str:
    return re.sub(r"[^a-z0-9-]", "", text.lower())


def relaxed_sg_alias(text: str) -> str:
    return normalize_sg_alias(text).replace("1", "")


def cell_volume(cell: list[float]) -> float:
    a, b, c, alpha, beta, gamma = cell
    ar, br, gr = map(math.radians, (alpha, beta, gamma))
    term = 1 + 2 * math.cos(ar) * math.cos(br) * math.cos(gr)
    term -= math.cos(ar) ** 2 + math.cos(br) ** 2 + math.cos(gr) ** 2
    term = max(term, 0.0)
    return a * b * c * math.sqrt(term)


def permute_cell(cell: list[float], perm: tuple[int, int, int]) -> list[float]:
    lengths = cell[:3]
    angles = cell[3:]
    old_to_new = {old: new for new, old in enumerate(perm)}
    new_lengths = [lengths[i] for i in perm]
    angle_pairs = {
        0: (1, 2),  # alpha opposite a
        1: (0, 2),  # beta opposite b
        2: (0, 1),  # gamma opposite c
    }
    old_angle_for_pair = {tuple(sorted(v)): k for k, v in angle_pairs.items()}
    new_angles = []
    for axis in range(3):
        pair = angle_pairs[axis]
        old_pair = tuple(sorted((perm[pair[0]], perm[pair[1]])))
        old_angle_idx = old_angle_for_pair[old_pair]
        new_angles.append(angles[old_angle_idx])
    return new_lengths + new_angles


def cell_distance(rruff_cell: list[float], icsd_cell: list[float]) -> dict[str, float]:
    rruff_vol = cell_volume(rruff_cell)
    best = None
    for perm in itertools.permutations((0, 1, 2)):
        cand = permute_cell(icsd_cell, perm)
        length_rel = [
            abs(rruff_cell[i] - cand[i]) / max(abs(rruff_cell[i]), 1e-6)
            for i in range(3)
        ]
        angle_abs = [abs(rruff_cell[3 + i] - cand[3 + i]) for i in range(3)]
        cand_vol = cell_volume(cand)
        volume_rel = abs(rruff_vol - cand_vol) / max(rruff_vol, 1e-6)
        score = sum(length_rel) / 3.0 + sum(angle_abs) / 180.0 + volume_rel
        entry = {
            "score": score,
            "length_rel_mean": sum(length_rel) / 3.0,
            "length_rel_max": max(length_rel),
            "angle_abs_mean": sum(angle_abs) / 3.0,
            "angle_abs_max": max(angle_abs),
            "volume_rel": volume_rel,
        }
        if best is None or entry["score"] < best["score"]:
            best = entry
    assert best is not None
    return best


@dataclass
class ICSDRow:
    idnum: int
    min_name: str
    chem_name: str
    sum_form: str
    sgr: str
    sgr_num: int | None
    cell: list[float]


def load_icsd(conn: sqlite3.Connection) -> tuple[dict[str, list[ICSDRow]], dict[str, set[int]], dict[str, set[int]]]:
    name_to_rows: dict[str, list[ICSDRow]] = defaultdict(list)
    exact_sg_to_nums: dict[str, set[int]] = defaultdict(set)
    relaxed_sg_to_nums: dict[str, set[int]] = defaultdict(set)

    for sgr, sgr_disp, sgr_num, orga_sgr_new in conn.execute(
        "SELECT SGR, SGR_DISP, SGR_NUM, OrgA_SGR_NEW FROM SPACE_GROUP"
    ):
        for alias in (sgr, sgr_disp, orga_sgr_new):
            if not alias:
                continue
            exact_sg_to_nums[normalize_sg_alias(alias)].add(int(sgr_num))
            relaxed_sg_to_nums[relaxed_sg_alias(alias)].add(int(sgr_num))

    for row in conn.execute(
        """
        SELECT IDNUM, MIN_NAME, CHEM_NAME, SUM_FORM, SGR, SGR_NUM,
               A_LEN, B_LEN, C_LEN, ALPHA, BETA, GAMMA
        FROM ICSD
        WHERE MIN_NAME IS NOT NULL AND TRIM(MIN_NAME) != ''
          AND A_LEN IS NOT NULL AND B_LEN IS NOT NULL AND C_LEN IS NOT NULL
          AND ALPHA IS NOT NULL AND BETA IS NOT NULL AND GAMMA IS NOT NULL
        """
    ):
        item = ICSDRow(
            idnum=int(row[0]),
            min_name=row[1],
            chem_name=row[2] or "",
            sum_form=row[3] or "",
            sgr=row[4] or "",
            sgr_num=int(row[5]) if row[5] is not None else None,
            cell=[float(row[6]), float(row[7]), float(row[8]), float(row[9]), float(row[10]), float(row[11])],
        )
        name_to_rows[normalize_name(item.min_name)].append(item)

    return name_to_rows, exact_sg_to_nums, relaxed_sg_to_nums


def score_match(
    rruff_name: str,
    rruff_sg: str,
    rruff_cell: list[float],
    candidate: ICSDRow,
    rruff_sgr_nums_exact: set[int],
    rruff_sgr_nums_relaxed: set[int],
) -> dict[str, object]:
    name_exact = normalize_name(rruff_name) == normalize_name(candidate.min_name)
    cell_metrics = cell_distance(rruff_cell, candidate.cell)
    sg_exact = candidate.sgr_num is not None and candidate.sgr_num in rruff_sgr_nums_exact
    sg_relaxed = candidate.sgr_num is not None and candidate.sgr_num in rruff_sgr_nums_relaxed
    score = cell_metrics["score"]
    if name_exact:
        score -= 0.50
    if sg_exact:
        score -= 0.35
    elif sg_relaxed:
        score -= 0.20
    return {
        "rruff_name": rruff_name,
        "rruff_sg": rruff_sg,
        "icsd_idnum": candidate.idnum,
        "icsd_min_name": candidate.min_name,
        "icsd_chem_name": candidate.chem_name,
        "icsd_sum_form": candidate.sum_form,
        "icsd_sgr": candidate.sgr,
        "icsd_sgr_num": candidate.sgr_num,
        "score": round(score, 6),
        "name_exact": name_exact,
        "sg_exact": sg_exact,
        "sg_relaxed": sg_relaxed,
        **{k: round(v, 6) for k, v in cell_metrics.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rruff-manifest", required=True)
    parser.add_argument("--icsd-sqlite", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    rruff_rows = json.loads(Path(args.rruff_manifest).read_text())
    conn = sqlite3.connect(args.icsd_sqlite)
    conn.row_factory = sqlite3.Row
    name_to_rows, exact_sg_to_nums, relaxed_sg_to_nums = load_icsd(conn)
    known_names = set(name_to_rows)

    all_matches = []
    summary = Counter()
    per_rruff = []

    for row in rruff_rows:
        rruff_name = row["name"]
        rruff_sg = row["space_group_symbol"]
        rruff_cell = [float(x) for x in row["cell"]]
        name_forms = canonical_name_forms(rruff_name, known_names)
        candidates = []
        for form in name_forms:
            candidates.extend(name_to_rows.get(form, []))
        dedup = {}
        for cand in candidates:
            dedup[cand.idnum] = cand
        candidates = list(dedup.values())

        sgr_exact = exact_sg_to_nums.get(normalize_sg_alias(rruff_sg), set())
        sgr_relaxed = relaxed_sg_to_nums.get(relaxed_sg_alias(rruff_sg), set())

        scored = [
            score_match(rruff_name, rruff_sg, rruff_cell, cand, sgr_exact, sgr_relaxed)
            for cand in candidates
        ]
        scored.sort(key=lambda x: (x["score"], -int(x["sg_exact"]), -int(x["name_exact"])))
        top = scored[: args.top_k]
        if top:
            summary["matched_rruff_rows"] += 1
            if top[0]["sg_exact"]:
                summary["top1_sg_exact"] += 1
            elif top[0]["sg_relaxed"]:
                summary["top1_sg_relaxed"] += 1
            if top[0]["length_rel_mean"] <= 0.05 and top[0]["volume_rel"] <= 0.15:
                summary["top1_loose_cell_consistent"] += 1
        else:
            summary["unmatched_rruff_rows"] += 1

        per_rruff.append(
            {
                "rid": row["rid"],
                "name": rruff_name,
                "space_group_symbol": rruff_sg,
                "cell": rruff_cell,
                "name_forms": name_forms,
                "candidate_count": len(scored),
                "top_matches": top,
            }
        )
        all_matches.extend({"rid": row["rid"], **match} for match in top)

    out_json = {
        "summary": summary,
        "rruff_rows": len(rruff_rows),
        "top_k": args.top_k,
        "results": per_rruff,
    }
    Path(args.out_json).write_text(json.dumps(out_json, indent=2))

    fieldnames = [
        "rid",
        "rruff_name",
        "rruff_sg",
        "icsd_idnum",
        "icsd_min_name",
        "icsd_chem_name",
        "icsd_sum_form",
        "icsd_sgr",
        "icsd_sgr_num",
        "score",
        "name_exact",
        "sg_exact",
        "sg_relaxed",
        "length_rel_mean",
        "length_rel_max",
        "angle_abs_mean",
        "angle_abs_max",
        "volume_rel",
    ]
    with Path(args.out_csv).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_matches:
            writer.writerow({k: row.get(k) for k in fieldnames})

    print(json.dumps({"summary": summary, "rruff_rows": len(rruff_rows), "top_k": args.top_k}, indent=2))


if __name__ == "__main__":
    main()
