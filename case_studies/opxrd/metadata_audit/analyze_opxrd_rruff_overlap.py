#!/usr/bin/env python3
"""Find conservative structural overlap between opXRD-908 and full RRUFF."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import gemmi
from pymatgen.analysis.structure_matcher import ElementComparator, StructureMatcher
from pymatgen.core import Composition, Element, Lattice, Structure

from compare_opxrd_full_rruff_distribution import space_group_symbol_to_number
from derive_opxrd_symmetry import build_structure, decode_json_strings, parse_phase


CELL_RE = re.compile(
    r"CELL PARAMETERS\s*:\s*"
    r"([-+.\dEe]+)\s+([-+.\dEe]+)\s+([-+.\dEe]+)\s+"
    r"([-+.\dEe]+)\s+([-+.\dEe]+)\s+([-+.\dEe]+)",
    re.IGNORECASE,
)
ATOM_HEADER_RE = re.compile(
    r"^\s*ATOM\s+X\s+Y\s+Z\s+OCCUPANCY", re.IGNORECASE | re.MULTILINE
)
ELEMENT_RE = re.compile(r"^([A-Z][a-z]?)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opxrd-archive", required=True, type=Path)
    parser.add_argument("--opxrd-records-csv", required=True, type=Path)
    parser.add_argument("--rruff-dif-zip", required=True, type=Path)
    parser.add_argument("--rruff-cukalpha-manifest", required=True, type=Path)
    parser.add_argument("--rruff-icsd-matches", type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--symprec", type=float, default=0.01)
    return parser.parse_args()


def cell_volume(cell: list[float]) -> float:
    a, b, c, alpha, beta, gamma = cell
    ar, br, gr = map(math.radians, (alpha, beta, gamma))
    term = 1 + 2 * math.cos(ar) * math.cos(br) * math.cos(gr)
    term -= math.cos(ar) ** 2 + math.cos(br) ** 2 + math.cos(gr) ** 2
    return a * b * c * math.sqrt(max(term, 0.0))


def permute_cell(cell: list[float], permutation: tuple[int, int, int]) -> list[float]:
    lengths = cell[:3]
    angles = cell[3:]
    angle_pairs = {0: (1, 2), 1: (0, 2), 2: (0, 1)}
    old_angle_for_pair = {
        tuple(sorted(pair)): angle_index
        for angle_index, pair in angle_pairs.items()
    }
    new_angles = []
    for axis in range(3):
        new_pair = angle_pairs[axis]
        old_pair = tuple(
            sorted((permutation[new_pair[0]], permutation[new_pair[1]]))
        )
        new_angles.append(angles[old_angle_for_pair[old_pair]])
    return [lengths[index] for index in permutation] + new_angles


def cell_distance(first: list[float], second: list[float]) -> dict[str, float]:
    first_volume = cell_volume(first)
    best: dict[str, float] | None = None
    for permutation in itertools.permutations((0, 1, 2)):
        candidate = permute_cell(second, permutation)
        length_rel = [
            abs(first[index] - candidate[index]) / max(abs(first[index]), 1e-9)
            for index in range(3)
        ]
        angle_abs = [
            abs(first[index + 3] - candidate[index + 3])
            for index in range(3)
        ]
        volume_rel = abs(first_volume - cell_volume(candidate)) / max(
            first_volume, 1e-9
        )
        metrics = {
            "length_rel_mean": sum(length_rel) / 3,
            "length_rel_max": max(length_rel),
            "angle_abs_mean": sum(angle_abs) / 3,
            "angle_abs_max": max(angle_abs),
            "volume_rel": volume_rel,
        }
        score = (
            metrics["length_rel_mean"]
            + metrics["angle_abs_mean"] / 180
            + metrics["volume_rel"]
        )
        if best is None or score < best["score"]:
            best = {**metrics, "score": score}
    assert best is not None
    return best


def strict_cell_match(metrics: dict[str, float]) -> bool:
    return (
        metrics["length_rel_mean"] <= 0.02
        and metrics["length_rel_max"] <= 0.04
        and metrics["angle_abs_max"] <= 2.0
        and metrics["volume_rel"] <= 0.05
    )


def loose_cell_match(metrics: dict[str, float]) -> bool:
    return (
        metrics["length_rel_mean"] <= 0.05
        and metrics["length_rel_max"] <= 0.10
        and metrics["angle_abs_max"] <= 5.0
        and metrics["volume_rel"] <= 0.15
    )


def read_opxrd_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["sg_number"] = int(row["sg_number"])
        row["cell"] = [
            float(row[f"cell_{key}"])
            for key in ("a", "b", "c", "alpha", "beta", "gamma")
        ]
    return rows


def read_rruff_rows(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text())
    parsed = []
    for row in rows:
        sg_number = space_group_symbol_to_number(row["space_group_symbol"])
        if sg_number is None:
            continue
        parsed.append(
            {
                **row,
                "sg_number": sg_number,
                "cell": [float(value) for value in row["cell"]],
            }
        )
    return parsed


def load_rruff_icsd_anchors(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text())
    anchors = {}
    for row in payload["results"]:
        if not row["top_matches"]:
            continue
        top = row["top_matches"][0]
        anchors[row["rid"]] = {
            "icsd_idnum": top["icsd_idnum"],
            "formula": top["icsd_sum_form"],
            "sg_exact": bool(top["sg_exact"]),
            "sg_relaxed": bool(top["sg_relaxed"]),
            "loose_cell": (
                float(top["length_rel_mean"]) <= 0.05
                and float(top["volume_rel"]) <= 0.15
            ),
        }
    return anchors


def formulas_match(first: Composition, second: Composition) -> bool:
    first_fraction = first.element_composition.fractional_composition
    second_fraction = second.element_composition.fractional_composition
    return first_fraction.almost_equals(
        second_fraction,
        rtol=1e-4,
        atol=1e-6,
    )


def load_opxrd_structures(
    archive_path: Path,
    record_names: set[str],
) -> tuple[dict[str, Structure], dict[str, str]]:
    structures = {}
    failures = {}
    with zipfile.ZipFile(archive_path) as archive:
        for name in sorted(record_names):
            try:
                record = json.loads(archive.read(name))
                label = decode_json_strings(record.get("label"))
                phases = label.get("phases") if isinstance(label, dict) else None
                if not phases or len(phases) != 1:
                    raise ValueError("record is not single-phase")
                structures[name] = build_structure(parse_phase(phases[0]))
            except Exception as exc:
                failures[name] = f"{type(exc).__name__}: {exc}"
    return structures, failures


def parse_element_label(label: str) -> str | None:
    match = ELEMENT_RE.match(label)
    if match is None:
        return None
    candidate = match.group(1)
    if Element.is_valid_symbol(candidate):
        return candidate
    if Element.is_valid_symbol(candidate[0]):
        return candidate[0]
    return None


def parse_rruff_asymmetric_sites(
    text: str,
) -> tuple[list[Any], list[list[float]]]:
    header = ATOM_HEADER_RE.search(text)
    if header is None:
        raise ValueError("atom table header not found")

    species: list[Any] = []
    coordinates: list[list[float]] = []
    for line in text[header.end() :].splitlines():
        if "X-RAY WAVELENGTH" in line.upper():
            break
        fields = line.split()
        if len(fields) < 6:
            continue
        element = parse_element_label(fields[0])
        if element is None:
            continue
        try:
            x, y, z, occupancy = map(float, fields[1:5])
        except ValueError:
            continue
        if occupancy <= 0:
            continue
        species.append(element if occupancy == 1.0 else {element: occupancy})
        coordinates.append([x, y, z])

    if not species:
        raise ValueError("no parseable atomic sites")
    return species, coordinates


def periodic_coordinate_key(coordinate: list[float], places: int = 6) -> tuple[float, ...]:
    return tuple(round(value % 1.0, places) % 1.0 for value in coordinate)


def expand_asymmetric_sites(
    space_group: gemmi.SpaceGroup,
    species: list[Any],
    coordinates: list[list[float]],
) -> tuple[list[Any], list[list[float]]]:
    expanded_species: list[Any] = []
    expanded_coordinates: list[list[float]] = []
    for site_species, coordinate in zip(species, coordinates):
        orbit: dict[tuple[float, ...], list[float]] = {}
        for operation in space_group.operations():
            transformed = [
                float(value) % 1.0
                for value in operation.apply_to_xyz(coordinate)
            ]
            orbit[periodic_coordinate_key(transformed)] = transformed
        expanded_species.extend([site_species] * len(orbit))
        expanded_coordinates.extend(orbit.values())
    return expanded_species, expanded_coordinates


def build_rruff_structure(text: str, sg_symbol: str) -> Structure:
    cell_match = CELL_RE.search(text)
    if cell_match is None:
        raise ValueError("cell parameters not found")
    cell = [float(value) for value in cell_match.groups()]
    species, coordinates = parse_rruff_asymmetric_sites(text)

    primary_group = gemmi.find_spacegroup_by_name(sg_symbol)
    if primary_group is None:
        raise ValueError(f"Gemmi cannot parse space-group setting {sg_symbol!r}")

    groups = [primary_group]
    primary_name = primary_group.xhm()
    if primary_name.endswith(":1"):
        alternate = gemmi.find_spacegroup_by_name(primary_name[:-2] + ":2")
        if alternate is not None:
            groups.append(alternate)

    expansions = [
        (*expand_asymmetric_sites(group, species, coordinates), group.xhm())
        for group in groups
    ]
    expanded_species, expanded_coordinates, _ = min(
        expansions, key=lambda expansion: len(expansion[0])
    )

    return Structure(
        Lattice.from_parameters(*cell),
        expanded_species,
        expanded_coordinates,
        coords_are_cartesian=False,
        to_unit_cell=True,
        validate_proximity=False,
    )


def load_rruff_structures(
    archive_path: Path,
    rows: dict[str, dict[str, Any]],
) -> tuple[dict[str, Structure], dict[str, str]]:
    structures = {}
    failures = {}
    with zipfile.ZipFile(archive_path) as archive:
        members = {Path(name).name: name for name in archive.namelist()}
        for rid, row in sorted(rows.items()):
            try:
                member = members[row["dif_file"]]
                text = archive.read(member).decode("utf-8", errors="ignore")
                structures[rid] = build_rruff_structure(
                    text, row["space_group_symbol"]
                )
            except Exception as exc:
                failures[rid] = f"{type(exc).__name__}: {exc}"
    return structures, failures


def element_set(structure: Structure) -> set[str]:
    return {element.symbol for element in structure.composition.elements}


def main() -> None:
    args = parse_args()
    opxrd_rows = read_opxrd_rows(args.opxrd_records_csv)
    rruff_rows = read_rruff_rows(args.rruff_cukalpha_manifest)
    rruff_icsd_anchors = load_rruff_icsd_anchors(args.rruff_icsd_matches)

    rruff_by_sg: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rruff_rows:
        rruff_by_sg[row["sg_number"]].append(row)

    candidate_pairs = []
    for opxrd in opxrd_rows:
        for rruff in rruff_by_sg[opxrd["sg_number"]]:
            metrics = cell_distance(opxrd["cell"], rruff["cell"])
            if not loose_cell_match(metrics):
                continue
            candidate_pairs.append(
                {
                    "opxrd_record": opxrd["record"],
                    "opxrd_source": opxrd["source"],
                    "opxrd_formula_csv": opxrd["reduced_formula"],
                    "rruff_rid": rruff["rid"],
                    "rruff_name": rruff["name"],
                    "sg_number": opxrd["sg_number"],
                    "strict_cell_match": strict_cell_match(metrics),
                    **metrics,
                }
            )

    opxrd_candidate_names = {
        pair["opxrd_record"] for pair in candidate_pairs
    }
    rruff_candidate_rows = {
        row["rid"]: row
        for row in rruff_rows
        if row["rid"] in {pair["rruff_rid"] for pair in candidate_pairs}
    }
    opxrd_structures, opxrd_failures = load_opxrd_structures(
        args.opxrd_archive, opxrd_candidate_names
    )
    rruff_structures, rruff_failures = load_rruff_structures(
        args.rruff_dif_zip, rruff_candidate_rows
    )

    strict_matcher = StructureMatcher(
        ltol=0.05,
        stol=0.30,
        angle_tol=3,
        primitive_cell=True,
        scale=False,
        attempt_supercell=False,
        comparator=ElementComparator(),
    )

    for pair in candidate_pairs:
        opxrd_structure = opxrd_structures.get(pair["opxrd_record"])
        rruff_structure = rruff_structures.get(pair["rruff_rid"])
        if opxrd_structure is None or rruff_structure is None:
            pair.update(
                {
                    "opxrd_formula": (
                        opxrd_structure.composition.reduced_formula
                        if opxrd_structure is not None
                        else None
                    ),
                    "rruff_formula": None,
                    "formula_exact": None,
                    "element_set_exact": None,
                    "structure_match": None,
                }
            )
        else:
            opxrd_formula = opxrd_structure.composition.reduced_formula
            rruff_formula = rruff_structure.composition.reduced_formula
            formula_exact = formulas_match(
                opxrd_structure.composition, rruff_structure.composition
            )
            elements_exact = element_set(opxrd_structure) == element_set(
                rruff_structure
            )
            structure_match = False
            if elements_exact and pair["strict_cell_match"]:
                try:
                    structure_match = bool(
                        strict_matcher.fit(opxrd_structure, rruff_structure)
                    )
                except Exception:
                    structure_match = False
            pair.update(
                {
                    "opxrd_formula": opxrd_formula,
                    "rruff_formula": rruff_formula,
                    "formula_exact": formula_exact,
                    "element_set_exact": elements_exact,
                    "structure_match": structure_match,
                }
            )

        anchor = rruff_icsd_anchors.get(pair["rruff_rid"])
        anchor_formula_exact = False
        if anchor is not None and opxrd_structure is not None:
            try:
                anchor_formula_exact = formulas_match(
                    opxrd_structure.composition,
                    Composition(anchor["formula"]),
                )
            except Exception:
                anchor_formula_exact = False
        pair.update(
            {
                "icsd_anchor_id": anchor["icsd_idnum"] if anchor else None,
                "icsd_anchor_formula": anchor["formula"] if anchor else None,
                "icsd_anchor_sg_exact": anchor["sg_exact"] if anchor else None,
                "icsd_anchor_loose_cell": anchor["loose_cell"] if anchor else None,
                "icsd_anchor_formula_exact": anchor_formula_exact,
            }
        )

    strict_pairs = [pair for pair in candidate_pairs if pair["strict_cell_match"]]
    formula_pairs = [pair for pair in strict_pairs if pair["formula_exact"]]
    element_pairs = [pair for pair in strict_pairs if pair["element_set_exact"]]
    structure_pairs = [pair for pair in strict_pairs if pair["structure_match"]]
    anchor_formula_pairs = [
        pair
        for pair in strict_pairs
        if pair["icsd_anchor_formula_exact"]
        and pair["icsd_anchor_sg_exact"]
        and pair["icsd_anchor_loose_cell"]
    ]
    composition_supported_pairs = [
        pair
        for pair in strict_pairs
        if pair["formula_exact"]
        or (
            pair["icsd_anchor_formula_exact"]
            and pair["icsd_anchor_sg_exact"]
            and pair["icsd_anchor_loose_cell"]
        )
    ]

    def unique_counts(pairs: list[dict[str, Any]]) -> dict[str, int]:
        return {
            "pairs": len(pairs),
            "opxrd_records": len({pair["opxrd_record"] for pair in pairs}),
            "rruff_rids": len({pair["rruff_rid"] for pair in pairs}),
            "rruff_minerals": len({pair["rruff_name"] for pair in pairs}),
        }

    summary = {
        "opxrd_records": len(opxrd_rows),
        "rruff_records": len(rruff_rows),
        "loose_same_sg_cell_candidates": unique_counts(candidate_pairs),
        "strict_same_sg_cell_candidates": unique_counts(strict_pairs),
        "strict_cell_exact_formula": unique_counts(formula_pairs),
        "strict_cell_exact_element_set": unique_counts(element_pairs),
        "strict_structure_match": unique_counts(structure_pairs),
        "strict_cell_exact_icsd_anchor_formula": unique_counts(
            anchor_formula_pairs
        ),
        "strict_cell_composition_supported_union": unique_counts(
            composition_supported_pairs
        ),
        "opxrd_structure_failures": len(opxrd_failures),
        "rruff_structure_failures": len(rruff_failures),
    }
    classifications = Counter(
        "structure_match"
        if pair["structure_match"]
        else "strict_formula"
        if pair["strict_cell_match"] and pair["formula_exact"]
        else "strict_elements"
        if pair["strict_cell_match"] and pair["element_set_exact"]
        else "strict_cell_only"
        if pair["strict_cell_match"]
        else "loose_cell_only"
        for pair in candidate_pairs
    )

    output = {
        "scope": (
            "Conservative candidate overlap between the 908 single-phase opXRD "
            "records with derived SG+lattice and the 2,777-record RRUFF Cu-Kalpha "
            "manifest. opXRD has no RRUFF RID field, so matches are structural "
            "inferences rather than identifier joins."
        ),
        "criteria": {
            "candidate_filter": "exact SG plus loose permutation-aware cell match",
            "strict_cell": {
                "length_rel_mean_max": 0.02,
                "length_rel_max_max": 0.04,
                "angle_abs_max_deg": 2.0,
                "volume_rel_max": 0.05,
            },
            "loose_cell": {
                "length_rel_mean_max": 0.05,
                "length_rel_max_max": 0.10,
                "angle_abs_max_deg": 5.0,
                "volume_rel_max": 0.15,
            },
            "structure_matcher": {
                "ltol": 0.05,
                "stol": 0.30,
                "angle_tol": 3,
                "primitive_cell": True,
                "scale": False,
                "comparator": "ElementComparator",
            },
            "icsd_anchor_support": (
                "Existing RRUFF-to-ICSD top anchor with exact SG, loose "
                "RRUFF-to-ICSD cell consistency, and exact elemental "
                "stoichiometry relative to opXRD"
            ),
        },
        "summary": summary,
        "classification_counts": dict(classifications),
        "opxrd_structure_failures": opxrd_failures,
        "rruff_structure_failures": rruff_failures,
        "candidate_pairs": sorted(
            candidate_pairs,
            key=lambda pair: (
                not bool(pair["structure_match"]),
                not bool(pair["formula_exact"]),
                not pair["strict_cell_match"],
                pair["score"],
                pair["opxrd_record"],
                pair["rruff_rid"],
            ),
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n")

    fieldnames = list(output["candidate_pairs"][0]) if candidate_pairs else []
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output["candidate_pairs"])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
