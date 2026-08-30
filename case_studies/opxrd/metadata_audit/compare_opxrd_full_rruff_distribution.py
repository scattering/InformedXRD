#!/usr/bin/env python3
"""Compare the opXRD-908 class distribution with full local RRUFF surfaces."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import gemmi
import numpy as np

from analyze_opxrd_single_phase_distribution import distribution_summary


SG_RE = re.compile(r"SPACE GROUP\s*[:=]\s*([^\n\r]+)", re.IGNORECASE)
WAVELENGTH_RE = re.compile(
    r"X-RAY WAVELENGTH\s*[:=]\s*([^\n\r]+)", re.IGNORECASE
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opxrd-records-csv", required=True, type=Path)
    parser.add_argument("--rruff-dif-zip", required=True, type=Path)
    parser.add_argument("--rruff-cukalpha-manifest", required=True, type=Path)
    parser.add_argument("--sg-eg-table", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser.parse_args()


def load_sg_to_eg(path: Path) -> dict[int, int]:
    with path.open(newline="") as handle:
        result = {
            int(row["Space Group"]): int(row["Extinction Group"])
            for row in csv.DictReader(handle)
        }
    if set(result) != set(range(1, 231)):
        raise ValueError("SG-to-EG table does not cover all 230 space groups")
    return result


def normalize_hm_symbol(symbol: str) -> list[str]:
    raw = symbol.strip().replace("\\par", "").strip()
    candidates = [raw]
    compact = raw.replace("_", "")
    compact = re.sub(r"^([A-ZR])(?=[0-9-])", r"\1 ", compact)
    candidates.extend((compact, compact.replace("/", " /")))
    return list(
        dict.fromkeys(" ".join(candidate.split()) for candidate in candidates)
    )


def space_group_symbol_to_number(symbol: str) -> int | None:
    manual_aliases = {
        "P2_1/b": 14,
        "F2/d": 15,
    }
    if symbol.strip() in manual_aliases:
        return manual_aliases[symbol.strip()]
    for candidate in normalize_hm_symbol(symbol):
        group = gemmi.find_spacegroup_by_name(candidate)
        if group is not None:
            return int(group.number)
    return None


def is_cu_like(wavelength: str | None) -> bool:
    if not wavelength:
        return False
    numbers = [
        float(value)
        for value in re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", wavelength)
    ]
    return any(1.53 <= value <= 1.55 for value in numbers)


def crystal_system(sg_number: int) -> str:
    boundaries = (
        (2, "triclinic"),
        (15, "monoclinic"),
        (74, "orthorhombic"),
        (142, "tetragonal"),
        (167, "trigonal"),
        (194, "hexagonal"),
        (230, "cubic"),
    )
    for maximum, label in boundaries:
        if sg_number <= maximum:
            return label
    raise ValueError(f"invalid space-group number: {sg_number}")


def load_opxrd(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["sg_number"] = int(row["sg_number"])
        row["eg_number"] = int(row["eg_number"])
    return rows


def conflict_aware_unique_patterns(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["raw_pattern_hash"]].append(row)

    selected = []
    for group in groups.values():
        if len({row["eg_number"] for row in group}) > 1:
            continue
        selected.append(group[0])
    return selected


def load_rruff_dif(
    path: Path,
    sg_to_eg: dict[int, int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    missing_sg = 0
    unparsed_symbols: Counter[str] = Counter()
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.endswith(".txt")]
        for name in names:
            text = archive.read(name).decode("utf-8", errors="ignore")
            sg_match = SG_RE.search(text)
            wavelength_match = WAVELENGTH_RE.search(text)
            if not sg_match:
                missing_sg += 1
                continue
            symbol = sg_match.group(1).strip()
            sg_number = space_group_symbol_to_number(symbol)
            if sg_number is None:
                unparsed_symbols[symbol] += 1
                continue
            wavelength = (
                wavelength_match.group(1).strip() if wavelength_match else None
            )
            rows.append(
                {
                    "record": name,
                    "name": Path(name).name.split("__", 1)[0],
                    "sg_symbol": symbol,
                    "sg_number": sg_number,
                    "eg_number": sg_to_eg[sg_number],
                    "wavelength": wavelength,
                    "cu_like": is_cu_like(wavelength),
                }
            )
    audit = {
        "archive_records": len(names),
        "missing_space_group": missing_sg,
        "unparsed_space_group_records": sum(unparsed_symbols.values()),
        "unparsed_space_group_symbols": dict(unparsed_symbols.most_common()),
        "mapped_records": len(rows),
    }
    return rows, audit


def load_rruff_manifest(
    path: Path,
    sg_to_eg: dict[int, int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_rows = json.loads(path.read_text())
    rows = []
    unparsed_symbols: Counter[str] = Counter()
    for row in source_rows:
        symbol = str(row["space_group_symbol"]).strip()
        sg_number = space_group_symbol_to_number(symbol)
        if sg_number is None:
            unparsed_symbols[symbol] += 1
            continue
        rows.append(
            {
                **row,
                "sg_number": sg_number,
                "eg_number": sg_to_eg[sg_number],
            }
        )
    return rows, {
        "manifest_records": len(source_rows),
        "mapped_records": len(rows),
        "unparsed_space_group_records": sum(unparsed_symbols.values()),
        "unparsed_space_group_symbols": dict(unparsed_symbols.most_common()),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sg_counts = Counter(row["sg_number"] for row in rows)
    eg_counts = Counter(row["eg_number"] for row in rows)
    system_counts = Counter(crystal_system(row["sg_number"]) for row in rows)
    return {
        "records": len(rows),
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
    }


def compare_distributions(
    first_rows: list[dict[str, Any]],
    second_rows: list[dict[str, Any]],
    label: str,
    universe: list[int],
) -> dict[str, Any]:
    first_counts = Counter(row[label] for row in first_rows)
    second_counts = Counter(row[label] for row in second_rows)
    first = np.asarray([first_counts[value] for value in universe], dtype=float)
    second = np.asarray([second_counts[value] for value in universe], dtype=float)
    first /= first.sum()
    second /= second.sum()
    midpoint = 0.5 * (first + second)

    def kl_divergence(probabilities: np.ndarray, reference: np.ndarray) -> float:
        mask = probabilities > 0
        return float(
            np.sum(
                probabilities[mask]
                * np.log2(probabilities[mask] / reference[mask])
            )
        )

    first_support = set(first_counts)
    second_support = set(second_counts)
    cosine = float(
        np.dot(first, second)
        / (math.sqrt(float(np.dot(first, first)))
           * math.sqrt(float(np.dot(second, second))))
    )
    return {
        "first_records": len(first_rows),
        "second_records": len(second_rows),
        "first_observed_classes": len(first_support),
        "second_observed_classes": len(second_support),
        "support_intersection": len(first_support & second_support),
        "first_support_covered_by_second": len(first_support & second_support)
        / len(first_support),
        "second_support_covered_by_first": len(first_support & second_support)
        / len(second_support),
        "first_only_classes": sorted(first_support - second_support),
        "second_only_classes": sorted(second_support - first_support),
        "total_variation_distance": float(0.5 * np.sum(np.abs(first - second))),
        "jensen_shannon_divergence_bits": float(
            0.5 * kl_divergence(first, midpoint)
            + 0.5 * kl_divergence(second, midpoint)
        ),
        "cosine_similarity": cosine,
    }


def main() -> None:
    args = parse_args()
    sg_to_eg = load_sg_to_eg(args.sg_eg_table)
    opxrd_all = load_opxrd(args.opxrd_records_csv)
    opxrd_clean = conflict_aware_unique_patterns(opxrd_all)
    rruff_all, rruff_dif_audit = load_rruff_dif(args.rruff_dif_zip, sg_to_eg)
    rruff_cu = [row for row in rruff_all if row["cu_like"]]
    rruff_manifest, rruff_manifest_audit = load_rruff_manifest(
        args.rruff_cukalpha_manifest, sg_to_eg
    )

    mineral_sg_rows = {}
    for row in rruff_manifest:
        key = (
            re.sub(r"[^a-z0-9]", "", str(row["name"]).lower()),
            row["sg_number"],
        )
        mineral_sg_rows.setdefault(key, row)
    rruff_unique_mineral_sg = list(mineral_sg_rows.values())

    surfaces = {
        "opxrd_908_as_shipped": opxrd_all,
        "opxrd_unique_patterns_without_label_conflicts": opxrd_clean,
        "rruff_all_mapped_dif": rruff_all,
        "rruff_cu_like_mapped_dif": rruff_cu,
        "rruff_cukalpha_manifest": rruff_manifest,
        "rruff_cukalpha_unique_mineral_sg": rruff_unique_mineral_sg,
    }
    comparisons = {}
    for opxrd_name in (
        "opxrd_908_as_shipped",
        "opxrd_unique_patterns_without_label_conflicts",
    ):
        for rruff_name in (
            "rruff_all_mapped_dif",
            "rruff_cu_like_mapped_dif",
            "rruff_cukalpha_manifest",
            "rruff_cukalpha_unique_mineral_sg",
        ):
            key = f"{opxrd_name}__vs__{rruff_name}"
            comparisons[key] = {
                "space_groups": compare_distributions(
                    surfaces[opxrd_name],
                    surfaces[rruff_name],
                    "sg_number",
                    list(range(1, 231)),
                ),
                "extinction_groups": compare_distributions(
                    surfaces[opxrd_name],
                    surfaces[rruff_name],
                    "eg_number",
                    list(range(1, 100)),
                ),
            }

    result = {
        "scope": (
            "Record-level SG/EG distributions for the opXRD single-phase "
            "derivable subset and full local RRUFF DIF/Cu-Kalpha surfaces"
        ),
        "rruff_dif_audit": rruff_dif_audit,
        "rruff_manifest_audit": rruff_manifest_audit,
        "surfaces": {
            name: summarize_rows(rows) for name, rows in surfaces.items()
        },
        "comparisons": comparisons,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
