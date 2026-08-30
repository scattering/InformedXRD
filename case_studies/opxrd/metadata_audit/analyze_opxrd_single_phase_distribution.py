#!/usr/bin/env python3
"""Characterize the 908 single-phase opXRD records with derivable symmetry."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from pymatgen.analysis.structure_matcher import ElementComparator, StructureMatcher
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from derive_opxrd_symmetry import (
    build_structure,
    decode_json_strings,
    parse_phase,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--sg-eg-table", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-records-csv", required=True, type=Path)
    parser.add_argument("--symprec", type=float, default=0.01)
    parser.add_argument("--angle-tolerance", type=float, default=5.0)
    parser.add_argument("--structure-ltol", type=float, default=0.05)
    parser.add_argument("--structure-stol", type=float, default=0.3)
    parser.add_argument("--structure-angle-tol", type=float, default=3.0)
    return parser.parse_args()


def load_sg_to_eg(path: Path) -> dict[int, int]:
    result: dict[int, int] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            result[int(row["Space Group"])] = int(row["Extinction Group"])
    if set(result) != set(range(1, 231)):
        raise ValueError("SG-to-EG table does not cover all 230 space groups")
    return result


def numeric_vector(value: Any) -> np.ndarray:
    value = decode_json_strings(value)
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or not len(array):
        raise ValueError("expected a nonempty one-dimensional numeric vector")
    return array


def hash_arrays(*arrays: np.ndarray, decimals: int | None = None) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        values = np.asarray(array, dtype=np.float64)
        if decimals is not None:
            values = np.round(values, decimals=decimals)
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def pattern_hashes(record: dict[str, Any]) -> tuple[str, str]:
    two_theta = numeric_vector(record.get("two_theta_values"))
    intensity = numeric_vector(record.get("intensities"))
    if len(two_theta) != len(intensity):
        raise ValueError("two-theta and intensity vectors have different lengths")

    raw_hash = hash_arrays(two_theta, intensity)
    maximum = float(np.max(np.abs(intensity)))
    normalized = intensity / maximum if maximum > 0 else intensity
    normalized_hash = hash_arrays(two_theta, normalized, decimals=8)
    return raw_hash, normalized_hash


def exact_structure_hash(structure) -> str:
    sites = []
    for site in structure:
        species = tuple(
            sorted(
                (str(specie), round(float(occupancy), 10))
                for specie, occupancy in site.species.items()
            )
        )
        coordinates = tuple(
            round(float(value % 1.0), 10) for value in site.frac_coords
        )
        sites.append((species, coordinates))
    payload = {
        "lattice": [
            round(float(value), 10) for value in structure.lattice.parameters
        ],
        "sites": sorted(sites),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def distribution_summary(counter: Counter[int], universe_size: int) -> dict[str, Any]:
    total = sum(counter.values())
    probabilities = np.asarray(list(counter.values()), dtype=np.float64) / total
    entropy = float(-np.sum(probabilities * np.log2(probabilities)))
    ranked = sorted(counter.items(), key=lambda item: (-item[1], item[0]))

    def top_share(k: int) -> float:
        return float(sum(count for _, count in ranked[:k]) / total)

    return {
        "n": total,
        "observed_classes": len(counter),
        "universe_classes": universe_size,
        "coverage_fraction": len(counter) / universe_size,
        "entropy_bits": entropy,
        "effective_classes": float(2**entropy),
        "normalized_entropy_over_universe": entropy / math.log2(universe_size),
        "herfindahl_index": float(np.sum(probabilities**2)),
        "top1_share": top_share(1),
        "top5_share": top_share(5),
        "top10_share": top_share(10),
        "counts": [
            {"class": int(label), "count": int(count), "fraction": count / total}
            for label, count in ranked
        ],
    }


def duplicate_summary(
    groups: dict[str, list[int]],
    rows: list[dict[str, Any]],
    *,
    minimum_size: int = 2,
) -> dict[str, Any]:
    duplicates = sorted(
        (indices for indices in groups.values() if len(indices) >= minimum_size),
        key=lambda indices: (-len(indices), [rows[i]["record"] for i in indices]),
    )
    redundant = sum(len(indices) - 1 for indices in duplicates)
    sg_conflicts = [
        indices
        for indices in duplicates
        if len({rows[index]["sg_number"] for index in indices}) > 1
    ]
    eg_conflicts = [
        indices
        for indices in duplicates
        if len({rows[index]["eg_number"] for index in indices}) > 1
    ]
    return {
        "duplicate_groups": len(duplicates),
        "records_in_duplicate_groups": sum(map(len, duplicates)),
        "redundant_records": redundant,
        "effective_unique_records": len(rows) - redundant,
        "largest_group_size": max(map(len, duplicates), default=1),
        "groups_with_sg_label_conflicts": len(sg_conflicts),
        "records_in_sg_label_conflicts": sum(map(len, sg_conflicts)),
        "groups_with_eg_label_conflicts": len(eg_conflicts),
        "records_in_eg_label_conflicts": sum(map(len, eg_conflicts)),
        "largest_groups": [
            {
                "size": len(indices),
                "records": [rows[i]["record"] for i in indices],
                "formulas": sorted(
                    {rows[index]["reduced_formula"] for index in indices}
                ),
                "sg_numbers": sorted(
                    {rows[index]["sg_number"] for index in indices}
                ),
                "eg_numbers": sorted(
                    {rows[index]["eg_number"] for index in indices}
                ),
            }
            for indices in duplicates[:20]
        ],
    }


class DisjointSet:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, first: int, second: int) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root != second_root:
            self.parent[second_root] = first_root


def structure_match_groups(
    structures: list[Any],
    rows: list[dict[str, Any]],
    matcher: StructureMatcher,
) -> dict[str, list[int]]:
    by_formula: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_formula[row["reduced_formula"]].append(index)

    sets = DisjointSet(len(rows))
    comparisons = 0
    matches = 0
    for indices in by_formula.values():
        for offset, first in enumerate(indices):
            for second in indices[offset + 1 :]:
                comparisons += 1
                if matcher.fit(structures[first], structures[second]):
                    sets.union(first, second)
                    matches += 1

    groups: dict[str, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        groups[str(sets.find(index))].append(index)
    groups["_comparisons"] = [comparisons]
    groups["_pair_matches"] = [matches]
    return groups


def representative_distribution(
    groups: dict[str, list[int]],
    rows: list[dict[str, Any]],
    label: str,
    universe_size: int,
    *,
    exclude_label_conflicts: bool = False,
) -> dict[str, Any]:
    representatives = []
    excluded_groups = 0
    excluded_records = 0
    for indices in groups.values():
        labels = {rows[index][label] for index in indices}
        if exclude_label_conflicts and len(labels) > 1:
            excluded_groups += 1
            excluded_records += len(indices)
            continue
        representatives.append(indices[0])

    summary = distribution_summary(
        Counter(rows[index][label] for index in representatives),
        universe_size,
    )
    summary["excluded_conflicting_groups"] = excluded_groups
    summary["excluded_records_in_conflicting_groups"] = excluded_records
    return summary


def main() -> None:
    args = parse_args()
    sg_to_eg = load_sg_to_eg(args.sg_eg_table)
    rows: list[dict[str, Any]] = []
    structures = []
    failures: Counter[str] = Counter()

    with zipfile.ZipFile(args.archive) as archive:
        for name in (item for item in archive.namelist() if item.endswith(".json")):
            with archive.open(name) as handle:
                record = json.load(handle)

            label = decode_json_strings(record.get("label"))
            if not isinstance(label, dict):
                continue
            phases = label.get("phases") or []
            if len(phases) != 1:
                continue

            phase = parse_phase(phases[0])
            try:
                structure = build_structure(phase)
                analyzer = SpacegroupAnalyzer(
                    structure,
                    symprec=args.symprec,
                    angle_tolerance=args.angle_tolerance,
                )
                sg_number = int(analyzer.get_space_group_number())
                sg_symbol = analyzer.get_space_group_symbol()
                crystal_system = analyzer.get_crystal_system()
                raw_pattern_hash, normalized_pattern_hash = pattern_hashes(record)
            except Exception as exc:
                failures[f"{type(exc).__name__}: {exc}"] += 1
                continue

            metadata = decode_json_strings(record.get("metadata"))
            metadata = metadata if isinstance(metadata, dict) else {}
            source = name.split("/", 1)[0]
            row = {
                "record": name,
                "source": source,
                "institution": metadata.get("institution"),
                "sg_number": sg_number,
                "sg_symbol": sg_symbol,
                "eg_number": sg_to_eg[sg_number],
                "crystal_system": crystal_system,
                "reduced_formula": structure.composition.element_composition.reduced_formula,
                "n_sites": len(structure),
                "cell_a": float(structure.lattice.a),
                "cell_b": float(structure.lattice.b),
                "cell_c": float(structure.lattice.c),
                "cell_alpha": float(structure.lattice.alpha),
                "cell_beta": float(structure.lattice.beta),
                "cell_gamma": float(structure.lattice.gamma),
                "cell_volume": float(structure.volume),
                "raw_pattern_hash": raw_pattern_hash,
                "normalized_pattern_hash": normalized_pattern_hash,
                "exact_structure_hash": exact_structure_hash(structure),
            }
            rows.append(row)
            structures.append(structure)

    sg_counts = Counter(row["sg_number"] for row in rows)
    eg_counts = Counter(row["eg_number"] for row in rows)
    system_counts = Counter(row["crystal_system"] for row in rows)
    source_counts = Counter(row["source"] for row in rows)
    institution_counts = Counter(row["institution"] for row in rows)
    formula_counts = Counter(row["reduced_formula"] for row in rows)

    raw_pattern_groups: dict[str, list[int]] = defaultdict(list)
    normalized_pattern_groups: dict[str, list[int]] = defaultdict(list)
    exact_structure_groups: dict[str, list[int]] = defaultdict(list)
    formula_groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        raw_pattern_groups[row["raw_pattern_hash"]].append(index)
        normalized_pattern_groups[row["normalized_pattern_hash"]].append(index)
        exact_structure_groups[row["exact_structure_hash"]].append(index)
        formula_groups[row["reduced_formula"]].append(index)

    matcher = StructureMatcher(
        ltol=args.structure_ltol,
        stol=args.structure_stol,
        angle_tol=args.structure_angle_tol,
        primitive_cell=True,
        scale=False,
        attempt_supercell=False,
        allow_subset=False,
        comparator=ElementComparator(),
    )
    matched_groups = structure_match_groups(structures, rows, matcher)
    comparisons = matched_groups.pop("_comparisons")[0]
    pair_matches = matched_groups.pop("_pair_matches")[0]

    result = {
        "scope": (
            "opXRD records with exactly one declared phase and a structure from "
            "which pymatgen computes a space group"
        ),
        "parameters": {
            "symprec": args.symprec,
            "angle_tolerance": args.angle_tolerance,
            "structure_matcher": {
                "ltol": args.structure_ltol,
                "stol": args.structure_stol,
                "angle_tol": args.structure_angle_tol,
                "primitive_cell": True,
                "scale": False,
                "comparator": "ElementComparator",
            },
        },
        "records": len(rows),
        "failures_among_single_phase_records": dict(failures.most_common()),
        "sources": dict(source_counts.most_common()),
        "institutions": {
            str(key): value for key, value in institution_counts.most_common()
        },
        "space_groups": distribution_summary(sg_counts, 230),
        "extinction_groups": distribution_summary(eg_counts, 99),
        "crystal_systems": [
            {
                "crystal_system": label,
                "count": count,
                "fraction": count / len(rows),
            }
            for label, count in system_counts.most_common()
        ],
        "chemistry": {
            "unique_reduced_formulas": len(formula_counts),
            "repeated_formula_groups": sum(
                count >= 2 for count in formula_counts.values()
            ),
            "records_in_repeated_formula_groups": sum(
                count for count in formula_counts.values() if count >= 2
            ),
            "top_formulas": [
                {"formula": formula, "count": count}
                for formula, count in formula_counts.most_common(25)
            ],
        },
        "duplicates": {
            "identical_raw_pattern": duplicate_summary(raw_pattern_groups, rows),
            "identical_normalized_pattern": duplicate_summary(
                normalized_pattern_groups, rows
            ),
            "identical_structure_payload": duplicate_summary(
                exact_structure_groups, rows
            ),
            "structure_matcher": {
                **duplicate_summary(matched_groups, rows),
                "candidate_pair_comparisons": comparisons,
                "matching_pairs_before_transitive_closure": pair_matches,
            },
        },
        "deduplicated_distributions": {
            "unique_raw_patterns_excluding_label_conflicts": {
                "space_groups": representative_distribution(
                    raw_pattern_groups,
                    rows,
                    "sg_number",
                    230,
                    exclude_label_conflicts=True,
                ),
                "extinction_groups": representative_distribution(
                    raw_pattern_groups,
                    rows,
                    "eg_number",
                    99,
                    exclude_label_conflicts=True,
                ),
            },
            "unique_exact_structure_payloads": {
                "space_groups": representative_distribution(
                    exact_structure_groups, rows, "sg_number", 230
                ),
                "extinction_groups": representative_distribution(
                    exact_structure_groups, rows, "eg_number", 99
                ),
            },
            "unique_structure_matcher_groups": {
                "space_groups": representative_distribution(
                    matched_groups,
                    rows,
                    "sg_number",
                    230,
                    exclude_label_conflicts=True,
                ),
                "extinction_groups": representative_distribution(
                    matched_groups,
                    rows,
                    "eg_number",
                    99,
                    exclude_label_conflicts=True,
                ),
            },
        },
    }

    match_group_by_index = {}
    match_group_size_by_index = {}
    for group_number, indices in enumerate(
        sorted(matched_groups.values(), key=lambda values: min(values))
    ):
        for index in indices:
            match_group_by_index[index] = group_number
            match_group_size_by_index[index] = len(indices)
    for index, row in enumerate(rows):
        raw_group = raw_pattern_groups[row["raw_pattern_hash"]]
        row["raw_pattern_group_size"] = len(raw_group)
        row["raw_pattern_label_conflict"] = (
            len({rows[item]["eg_number"] for item in raw_group}) > 1
        )
        row["structure_match_group"] = match_group_by_index[index]
        row["structure_match_group_size"] = match_group_size_by_index[index]

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")

    args.output_records_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_records_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
