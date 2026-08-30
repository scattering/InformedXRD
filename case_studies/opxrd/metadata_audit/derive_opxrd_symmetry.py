#!/usr/bin/env python3
"""Audit opXRD labels that can be derived from stored atomic structures."""

from __future__ import annotations

import argparse
import ast
import importlib.metadata
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--symprec", type=float, default=0.01)
    parser.add_argument("--angle-tolerance", type=float, default=5.0)
    parser.add_argument("--progress-every", type=int, default=10_000)
    return parser.parse_args()


def decode_json_strings(value: Any) -> Any:
    """Decode fields that opXRD serializes as one or more JSON strings."""
    while isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            break
        if decoded == value:
            break
        value = decoded
    return value


def parse_phase(value: Any) -> dict[str, Any]:
    phase = decode_json_strings(value)
    return phase if isinstance(phase, dict) else {}


def parse_lattice(value: Any) -> Lattice:
    lattice = decode_json_strings(value)
    if isinstance(lattice, str):
        lattice = ast.literal_eval(lattice)

    if isinstance(lattice, dict):
        keys = ("a", "b", "c", "alpha", "beta", "gamma")
        if not all(key in lattice for key in keys):
            raise ValueError(f"incomplete lattice dictionary: {sorted(lattice)}")
        parameters = [float(lattice[key]) for key in keys]
    elif isinstance(lattice, (list, tuple)) and len(lattice) == 6:
        parameters = [float(item) for item in lattice]
    else:
        raise ValueError(f"unsupported lattice value: {type(lattice).__name__}")

    return Lattice.from_parameters(*parameters)


def parse_basis(value: Any) -> tuple[list[Any], list[list[float]]]:
    basis = decode_json_strings(value)
    if not isinstance(basis, list) or not basis:
        raise ValueError("basis is absent or is not a nonempty list")

    species: list[Any] = []
    fractional_coordinates: list[list[float]] = []
    for raw_site in basis:
        site = decode_json_strings(raw_site)
        if not isinstance(site, dict):
            raise ValueError("basis entry is not a dictionary")

        symbol = site.get("symbol")
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("basis entry has no chemical symbol")
        occupancy = float(site.get("occupancy", 1.0))
        if occupancy <= 0:
            raise ValueError(f"nonpositive occupancy: {occupancy}")

        species.append(symbol if occupancy == 1.0 else {symbol: occupancy})
        fractional_coordinates.append(
            [float(site[axis]) for axis in ("x", "y", "z")]
        )

    return species, fractional_coordinates


def derive_space_group(
    phase: dict[str, Any],
    *,
    symprec: float,
    angle_tolerance: float,
) -> tuple[int, str]:
    structure = build_structure(phase)
    analyzer = SpacegroupAnalyzer(
        structure,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
    )
    return analyzer.get_space_group_number(), analyzer.get_space_group_symbol()


def build_structure(phase: dict[str, Any]) -> Structure:
    lattice = parse_lattice(phase.get("lattice"))
    species, fractional_coordinates = parse_basis(phase.get("basis"))
    return Structure(
        lattice,
        species,
        fractional_coordinates,
        coords_are_cartesian=False,
        to_unit_cell=True,
        validate_proximity=False,
    )


def has_value(value: Any) -> bool:
    value = decode_json_strings(value)
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict)):
        return bool(value)
    return True


def increment(counter: Counter[str], key: str, condition: bool) -> None:
    if condition:
        counter[key] += 1


def main() -> None:
    args = parse_args()
    counts: Counter[str] = Counter()
    derived_space_groups: Counter[int] = Counter()
    failure_types: Counter[str] = Counter()
    failure_examples: list[dict[str, str]] = []

    with zipfile.ZipFile(args.archive) as archive:
        names = [name for name in archive.namelist() if name.endswith(".json")]
        counts["records_total"] = len(names)

        for record_index, name in enumerate(names, start=1):
            with archive.open(name) as handle:
                record = json.load(handle)

            label = decode_json_strings(record.get("label"))
            label = label if isinstance(label, dict) else {}
            raw_phases = label.get("phases") or []
            phases = [parse_phase(raw_phase) for raw_phase in raw_phases]

            if not phases:
                counts["records_without_phases"] += 1
                continue

            counts["records_with_phases"] += 1
            is_single = len(phases) == 1
            prefix = "single_phase" if is_single else "multiphase"
            counts[f"{prefix}_records"] += 1

            stored_flags: list[bool] = []
            lattice_parameter_flags: list[bool] = []
            computed_flags: list[bool] = []
            assignable_flags: list[bool] = []

            for phase_index, phase in enumerate(phases):
                stored = has_value(phase.get("spacegroup"))
                try:
                    parse_lattice(phase.get("lattice"))
                    lattice_parameters = True
                except Exception:
                    lattice_parameters = False
                stored_flags.append(stored)
                lattice_parameter_flags.append(lattice_parameters)

                try:
                    sg_number, _ = derive_space_group(
                        phase,
                        symprec=args.symprec,
                        angle_tolerance=args.angle_tolerance,
                    )
                    computed = True
                    derived_space_groups[sg_number] += 1
                except Exception as exc:  # Keep auditing malformed public records.
                    computed = False
                    failure_key = f"{type(exc).__name__}: {exc}"
                    failure_types[failure_key] += 1
                    if len(failure_examples) < 25:
                        failure_examples.append(
                            {
                                "record": name,
                                "phase_index": str(phase_index),
                                "error": failure_key,
                            }
                        )

                computed_flags.append(computed)
                assignable_flags.append(stored or computed)

            any_stored = any(stored_flags)
            all_stored = all(stored_flags)
            any_computed = any(computed_flags)
            all_computed = all(computed_flags)
            any_assignable = any(assignable_flags)
            all_assignable = all(assignable_flags)
            any_lattice_parameters = any(lattice_parameter_flags)
            all_lattice_parameters = all(lattice_parameter_flags)
            all_lattice_parameters_and_assignable = all(
                lattice and assignable
                for lattice, assignable in zip(
                    lattice_parameter_flags, assignable_flags
                )
            )

            increment(counts, f"{prefix}_any_sg_stored", any_stored)
            increment(counts, f"{prefix}_all_sg_stored", all_stored)
            increment(counts, f"{prefix}_any_sg_computed", any_computed)
            increment(counts, f"{prefix}_all_sg_computed", all_computed)
            increment(counts, f"{prefix}_any_sg_assignable", any_assignable)
            increment(counts, f"{prefix}_all_sg_assignable", all_assignable)
            increment(
                counts,
                f"{prefix}_any_lattice_parameters",
                any_lattice_parameters,
            )
            increment(
                counts,
                f"{prefix}_all_lattice_parameters",
                all_lattice_parameters,
            )
            increment(
                counts,
                f"{prefix}_all_lattice_parameters_and_sg_assignable",
                all_lattice_parameters_and_assignable,
            )

            increment(counts, "records_any_sg_stored", any_stored)
            increment(counts, "records_all_sg_stored", all_stored)
            increment(counts, "records_any_sg_computed", any_computed)
            increment(counts, "records_all_sg_computed", all_computed)
            increment(counts, "records_any_sg_assignable", any_assignable)
            increment(counts, "records_all_sg_assignable", all_assignable)
            increment(
                counts,
                "records_any_lattice_parameters",
                any_lattice_parameters,
            )
            increment(
                counts,
                "records_all_lattice_parameters",
                all_lattice_parameters,
            )
            increment(
                counts,
                "records_all_lattice_parameters_and_sg_assignable",
                all_lattice_parameters_and_assignable,
            )

            if args.progress_every and record_index % args.progress_every == 0:
                print(
                    f"processed {record_index:,}/{len(names):,} records",
                    file=sys.stderr,
                    flush=True,
                )

    result = {
        "archive": str(args.archive.resolve()),
        "software": {
            package: importlib.metadata.version(package)
            for package in ("pymatgen", "spglib")
        },
        "parameters": {
            "symprec": args.symprec,
            "angle_tolerance": args.angle_tolerance,
        },
        "counts": dict(sorted(counts.items())),
        "derived_space_group_counts": {
            str(key): value for key, value in sorted(derived_space_groups.items())
        },
        "derivation_failures": {
            "counts": dict(failure_types.most_common()),
            "examples": failure_examples,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["counts"], indent=2))


if __name__ == "__main__":
    main()
