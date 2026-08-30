#!/usr/bin/env python3
"""Build RRUFF-compatible Cu-Kalpha evaluation views from single-phase opXRD.

The wavelength gate deliberately matches the RRUFF source convention:

* an explicit Cu Kalpha1/Kalpha2 pair near 1.540562/1.544390 Angstrom, or
* the conventional 2:1 weighted-average wavelength.

Monochromated Kalpha1-only and Kalpha2-only records are excluded. Structures
are projected into the frozen ICSD structural-community frame from
arXiv:2606.30967. Community membership is retained as benchmark metadata; it
is not used as an automatic validity filter.

The script emits:

* a complete candidate CSV, including duplicate and community metadata;
* a JSON summary with wavelength, range, grid, duplicate, and community counts;
* one compressed NPZ candidate view for each requested angular window; and
* one CSV manifest for each NPZ view.

Each evaluation view is interpolated onto a 0.01-degree grid and max-normalized.
By default, every retained native pattern strictly covers the complete output
grid. A nonzero endpoint-bin tolerance is an explicit diagnostic option that
uses NumPy's constant edge fill for no more than the requested number of native
bins; it is not used for the primary strict-support surface.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import csv
import hashlib
import importlib
import importlib.metadata
import json
import math
import sys
import warnings
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


CU_KALPHA1 = 1.540562
CU_KALPHA2 = 1.544390
CU_KALPHA_WEIGHTED_AVERAGE = (2.0 * CU_KALPHA1 + CU_KALPHA2) / 3.0
DEFAULT_WAVELENGTH_TOLERANCE = 5e-4
DEFAULT_WINDOWS = ("5:90", "5:80", "10:90", "10:80", "15:80")

_COMMUNITY_MODULE = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--sg-eg-table", required=True, type=Path)
    parser.add_argument("--icsd-features", type=Path)
    parser.add_argument("--community-assignments", type=Path)
    parser.add_argument("--community-thresholds", type=Path)
    parser.add_argument("--crystal-community-scripts", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--windows", nargs="+", default=list(DEFAULT_WINDOWS))
    parser.add_argument("--grid-step", type=float, default=0.01)
    parser.add_argument("--max-native-step", type=float, default=0.05)
    parser.add_argument(
        "--endpoint-native-bin-tolerance",
        type=float,
        default=0.0,
        help=(
            "Optional constant edge fill, expressed in native median bins. "
            "The default 0 requires strict measured support."
        ),
    )
    parser.add_argument(
        "--wavelength-tolerance",
        type=float,
        default=DEFAULT_WAVELENGTH_TOLERANCE,
    )
    parser.add_argument("--symprec", type=float, default=0.01)
    parser.add_argument("--angle-tolerance", type=float, default=5.0)
    parser.add_argument("--wl-iters", type=int, default=3)
    parser.add_argument("--max-sites", type=int, default=256)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-community",
        action="store_true",
        help="Build wavelength/range manifests without ICSD projection.",
    )
    return parser.parse_args()


def decode_json_strings(value: Any) -> Any:
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
            raise ValueError("incomplete lattice dictionary")
        parameters = [float(lattice[key]) for key in keys]
    elif isinstance(lattice, (list, tuple)) and len(lattice) == 6:
        parameters = [float(item) for item in lattice]
    else:
        raise ValueError("unsupported lattice value")
    return Lattice.from_parameters(*parameters)


def parse_basis(value: Any) -> tuple[list[Any], list[list[float]]]:
    basis = decode_json_strings(value)
    if not isinstance(basis, list) or not basis:
        raise ValueError("basis is absent or empty")

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
            raise ValueError("basis entry has nonpositive occupancy")
        species.append(symbol if occupancy == 1.0 else {symbol: occupancy})
        fractional_coordinates.append(
            [float(site[axis]) for axis in ("x", "y", "z")]
        )
    return species, fractional_coordinates


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


def numeric_vector(value: Any, name: str) -> np.ndarray:
    value = decode_json_strings(value)
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or not len(array):
        raise ValueError(f"{name} is not a nonempty one-dimensional vector")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains nonfinite values")
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def exact_structure_hash(structure: Structure) -> str:
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


def parse_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def near(value: float | None, target: float, tolerance: float) -> bool:
    return value is not None and abs(value - target) < tolerance


def classify_rruff_cukalpha(
    primary: float | None,
    secondary: float | None,
    tolerance: float,
) -> str | None:
    if primary is None:
        return None
    if secondary is not None:
        forward = near(primary, CU_KALPHA1, tolerance) and near(
            secondary, CU_KALPHA2, tolerance
        )
        reverse = near(primary, CU_KALPHA2, tolerance) and near(
            secondary, CU_KALPHA1, tolerance
        )
        return "kalpha1_kalpha2_pair" if forward or reverse else None
    if near(primary, CU_KALPHA_WEIGHTED_AVERAGE, tolerance):
        return "weighted_average"
    return None


def load_sg_to_eg(path: Path) -> dict[int, int]:
    result: dict[int, int] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            result[int(row["Space Group"])] = int(row["Extinction Group"])
    if set(result) != set(range(1, 231)):
        raise ValueError("SG-to-EG table does not cover all 230 space groups")
    return result


def parse_windows(values: list[str]) -> list[tuple[float, float]]:
    result = []
    for value in values:
        parts = value.split(":")
        if len(parts) != 2:
            raise ValueError(f"window must be MIN:MAX, received {value!r}")
        lower, upper = map(float, parts)
        if not lower < upper:
            raise ValueError(f"window lower bound must be below upper: {value!r}")
        result.append((lower, upper))
    return result


def window_slug(lower: float, upper: float) -> str:
    def encode(value: float) -> str:
        return f"{value:g}".replace(".", "p")

    return f"{encode(lower)}_{encode(upper)}"


def supports_window(
    row: dict[str, Any],
    lower: float,
    upper: float,
    endpoint_native_bin_tolerance: float,
) -> bool:
    tolerance = endpoint_native_bin_tolerance * float(row["native_step"])
    return (
        float(row["two_theta_min"]) <= lower + tolerance + 1e-9
        and float(row["two_theta_max"]) >= upper - tolerance - 1e-9
    )


def strictly_covers_window(
    row: dict[str, Any],
    lower: float,
    upper: float,
) -> bool:
    return (
        float(row["two_theta_min"]) <= lower + 1e-9
        and float(row["two_theta_max"]) >= upper - 1e-9
    )


def load_candidates(
    archive_path: Path,
    sg_to_eg: dict[int, int],
    *,
    wavelength_tolerance: float,
    symprec: float,
    angle_tolerance: float,
) -> tuple[list[dict[str, Any]], list[Structure], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    structures: list[Structure] = []
    counts: Counter[str] = Counter()
    failures: Counter[str] = Counter()

    with zipfile.ZipFile(archive_path) as archive:
        names = sorted(name for name in archive.namelist() if name.endswith(".json"))
        counts["archive_records"] = len(names)
        for name in names:
            with archive.open(name) as handle:
                record = json.load(handle)

            label = decode_json_strings(record.get("label"))
            if not isinstance(label, dict):
                counts["missing_label"] += 1
                continue
            phases = label.get("phases") or []
            if len(phases) != 1:
                counts["not_single_phase"] += 1
                continue
            counts["single_phase"] += 1

            phase = parse_phase(phases[0])
            try:
                structure = build_structure(phase)
                analyzer = SpacegroupAnalyzer(
                    structure,
                    symprec=symprec,
                    angle_tolerance=angle_tolerance,
                )
                sg_number = int(analyzer.get_space_group_number())
                sg_symbol = analyzer.get_space_group_symbol()
                crystal_system = analyzer.get_crystal_system()
            except Exception as exc:
                counts["structure_or_symmetry_failure"] += 1
                failures[f"structure_or_symmetry: {type(exc).__name__}: {exc}"] += 1
                continue
            counts["single_phase_symmetry_derivable"] += 1

            xray_info = decode_json_strings(label.get("xray_info")) or {}
            if not isinstance(xray_info, dict):
                xray_info = {}
            primary = parse_float(xray_info.get("primary_wavelength"))
            secondary = parse_float(xray_info.get("secondary_wavelength"))
            wavelength_mode = classify_rruff_cukalpha(
                primary,
                secondary,
                wavelength_tolerance,
            )
            if wavelength_mode is None:
                counts["wavelength_rejected"] += 1
                continue
            counts["wavelength_accepted"] += 1
            counts[f"wavelength_{wavelength_mode}"] += 1

            try:
                two_theta = numeric_vector(
                    record.get("two_theta_values"), "two_theta_values"
                )
                intensities = numeric_vector(record.get("intensities"), "intensities")
                if len(two_theta) != len(intensities):
                    raise ValueError("two-theta and intensity lengths differ")
                deltas = np.diff(two_theta)
                if not np.all(deltas > 0):
                    raise ValueError("two-theta values are not strictly increasing")
                native_step = float(np.median(deltas))
                if native_step <= 0:
                    raise ValueError("nonpositive median two-theta step")
            except Exception as exc:
                counts["pattern_axis_failure"] += 1
                failures[f"pattern_axis: {type(exc).__name__}: {exc}"] += 1
                continue
            counts["pattern_axis_valid"] += 1

            maximum = float(np.max(np.abs(intensities)))
            normalized = intensities / maximum if maximum > 0 else intensities
            metadata = decode_json_strings(record.get("metadata"))
            metadata = metadata if isinstance(metadata, dict) else {}
            raw_hash = hash_arrays(two_theta, intensities)
            normalized_hash = hash_arrays(two_theta, normalized, decimals=8)
            row = {
                "record": name,
                "source": name.split("/", 1)[0],
                "institution": metadata.get("institution"),
                "wavelength_mode": wavelength_mode,
                "primary_wavelength": primary,
                "secondary_wavelength": secondary,
                "sg_number": sg_number,
                "sg_symbol": sg_symbol,
                "eg_number": sg_to_eg[sg_number],
                "crystal_system": crystal_system,
                "reduced_formula": (
                    structure.composition.element_composition.reduced_formula
                ),
                "n_sites": len(structure),
                "two_theta_min": float(two_theta[0]),
                "two_theta_max": float(two_theta[-1]),
                "n_points": len(two_theta),
                "native_step": native_step,
                "native_step_min": float(np.min(deltas)),
                "native_step_max": float(np.max(deltas)),
                "raw_pattern_hash": raw_hash,
                "normalized_pattern_hash": normalized_hash,
                "exact_structure_hash": exact_structure_hash(structure),
                "duplicate_status": "unique",
                "duplicate_of": "",
                "selected_unique_pattern": True,
                "_two_theta": two_theta,
                "_intensities": intensities,
            }
            rows.append(row)
            structures.append(structure)

    raw_groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        raw_groups[str(row["raw_pattern_hash"])].append(index)

    duplicate_groups = 0
    redundant_records = 0
    conflict_groups = 0
    conflict_records = 0
    for indices in raw_groups.values():
        if len(indices) < 2:
            continue
        duplicate_groups += 1
        labels = {
            (int(rows[index]["sg_number"]), int(rows[index]["eg_number"]))
            for index in indices
        }
        if len(labels) > 1:
            conflict_groups += 1
            conflict_records += len(indices)
            for index in indices:
                rows[index]["duplicate_status"] = "label_conflict"
                rows[index]["selected_unique_pattern"] = False
            continue
        representative = min(indices, key=lambda index: str(rows[index]["record"]))
        rows[representative]["duplicate_status"] = "representative"
        for index in indices:
            if index == representative:
                continue
            rows[index]["duplicate_status"] = "redundant"
            rows[index]["duplicate_of"] = rows[representative]["record"]
            rows[index]["selected_unique_pattern"] = False
            redundant_records += 1

    summary = {
        "counts": dict(counts),
        "failure_types": dict(failures.most_common()),
        "duplicates": {
            "exact_raw_pattern_groups": duplicate_groups,
            "redundant_records_removed": redundant_records,
            "label_conflict_groups_excluded": conflict_groups,
            "records_in_label_conflict_groups": conflict_records,
            "selected_unique_patterns": sum(
                bool(row["selected_unique_pattern"]) for row in rows
            ),
        },
    }
    return rows, structures, summary


def initialize_community_worker(
    crystal_community_scripts: str,
    wl_iters: int,
    max_sites: int,
) -> None:
    global _COMMUNITY_MODULE
    scripts_path = str(Path(crystal_community_scripts).resolve())
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)
    warnings.filterwarnings(
        "ignore",
        message=r"No oxidation states specified on sites.*",
        category=UserWarning,
    )
    module = importlib.import_module("icsd_densify_worker")
    module.init_worker(
        None,
        None,
        None,
        wl_iters,
        max_sites,
        "matminer_ops",
    )
    _COMMUNITY_MODULE = module


def featurize_structure(item: tuple[int, Structure]) -> tuple[int, np.ndarray | None, str]:
    index, structure = item
    try:
        if _COMMUNITY_MODULE is None:
            raise RuntimeError("community worker was not initialized")
        if len(structure) > _COMMUNITY_MODULE.WORKER_MAX_SITES:
            raise ValueError(
                f"structure has {len(structure)} sites, "
                f"limit is {_COMMUNITY_MODULE.WORKER_MAX_SITES}"
            )
        embedding_structure = structure.copy()
        embedding_structure.remove_oxidation_states()
        embedding = _COMMUNITY_MODULE.build_structure_embedding(
            embedding_structure,
            wl_iters=_COMMUNITY_MODULE.WORKER_WL_ITERS,
            cnn=_COMMUNITY_MODULE.WORKER_CNN,
            site_fp=_COMMUNITY_MODULE.WORKER_SITE_FP,
            geom_dim=_COMMUNITY_MODULE.WORKER_GEOM_DIM,
            local_mode=_COMMUNITY_MODULE.WORKER_LOCAL_MODE,
        )
        return index, np.asarray(embedding, dtype=np.float64), ""
    except Exception as exc:
        return index, None, f"{type(exc).__name__}: {exc}"


def load_community_labels(path: Path) -> np.ndarray:
    labels = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                labels.append(int(float(row.get("community", "-1"))))
            except (TypeError, ValueError):
                labels.append(-1)
    return np.asarray(labels, dtype=np.int64)


def community_centroids(
    projected: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    community_ids = np.asarray(sorted(set(labels[labels >= 0].tolist())), dtype=np.int64)
    index = {int(community): position for position, community in enumerate(community_ids)}
    centroids = np.zeros((len(community_ids), projected.shape[1]), dtype=np.float64)
    counts = np.zeros(len(community_ids), dtype=np.int64)
    for vector, label in zip(projected, labels):
        if label < 0:
            continue
        position = index[int(label)]
        centroids[position] += vector
        counts[position] += 1
    centroids /= np.maximum(counts[:, None], 1)
    return community_ids, centroids


def project_communities(
    rows: list[dict[str, Any]],
    structures: list[Structure],
    *,
    icsd_features_path: Path,
    community_assignments_path: Path,
    community_thresholds_path: Path,
    crystal_community_scripts: Path,
    wl_iters: int,
    max_sites: int,
    n_jobs: int,
    seed: int,
) -> dict[str, Any]:
    warnings.filterwarnings(
        "ignore",
        message=r"CrystalNN: cannot locate an appropriate radius.*",
        category=UserWarning,
    )

    embeddings: list[np.ndarray | None] = [None] * len(rows)
    errors: dict[int, str] = {}
    items = list(enumerate(structures))
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=n_jobs,
        initializer=initialize_community_worker,
        initargs=(str(crystal_community_scripts), wl_iters, max_sites),
    ) as executor:
        for index, embedding, error in executor.map(
            featurize_structure,
            items,
            chunksize=4,
        ):
            embeddings[index] = embedding
            if error:
                errors[index] = error

    successful = [index for index, embedding in enumerate(embeddings) if embedding is not None]
    if not successful:
        raise RuntimeError("no opXRD structure could be featurized")

    icsd_features = np.load(icsd_features_path, mmap_mode="r")
    labels = load_community_labels(community_assignments_path)
    if len(icsd_features) != len(labels):
        raise ValueError(
            f"ICSD feature rows ({len(icsd_features)}) do not match "
            f"community labels ({len(labels)})"
        )

    scaler = StandardScaler()
    scaled = scaler.fit_transform(icsd_features)
    pca = PCA(n_components=32, random_state=seed)
    projected_icsd = pca.fit_transform(scaled)
    community_ids, centroids = community_centroids(projected_icsd, labels)

    output_embeddings = np.vstack([embeddings[index] for index in successful])
    if output_embeddings.shape[1] != icsd_features.shape[1]:
        raise ValueError(
            f"opXRD embedding width ({output_embeddings.shape[1]}) does not match "
            f"ICSD feature width ({icsd_features.shape[1]})"
        )
    projected_output = pca.transform(scaler.transform(output_embeddings))
    distances = np.linalg.norm(
        projected_output[:, None, :] - centroids[None, :, :],
        axis=2,
    )
    nearest_positions = np.argmin(distances, axis=1)
    nearest_communities = community_ids[nearest_positions]
    nearest_distances = distances[np.arange(len(successful)), nearest_positions]

    threshold_payload = json.loads(community_thresholds_path.read_text())
    thresholds = {
        int(key): float(value)
        for key, value in threshold_payload["per_community_p95_threshold"].items()
    }
    community_sizes = {
        int(key): int(value)
        for key, value in threshold_payload.get("per_community_size", {}).items()
    }
    if set(map(int, community_ids)) != set(thresholds):
        raise ValueError(
            "community ids in the frozen assignments do not match the "
            "per-community threshold artifact"
        )

    for index, community, distance, vector in zip(
        successful,
        nearest_communities,
        nearest_distances,
        projected_output,
    ):
        threshold = thresholds.get(int(community))
        rows[index]["assigned_community"] = int(community)
        rows[index]["nearest_centroid_distance"] = float(distance)
        rows[index]["community_p95_threshold"] = threshold
        rows[index]["community_size"] = community_sizes.get(int(community))
        rows[index]["outlier_like"] = threshold is None or float(distance) > threshold
        rows[index]["community_pca1"] = float(vector[0])
        rows[index]["community_pca2"] = float(vector[1])

    for index, error in errors.items():
        rows[index]["community_error"] = error
        rows[index]["outlier_like"] = None

    del projected_icsd
    del scaled
    del icsd_features

    assigned = [row for row in rows if row.get("assigned_community") is not None]
    return {
        "reference": {
            "icsd_features": str(icsd_features_path),
            "icsd_features_sha256": sha256_file(icsd_features_path),
            "community_assignments": str(community_assignments_path),
            "community_assignments_sha256": sha256_file(
                community_assignments_path
            ),
            "community_thresholds": str(community_thresholds_path),
            "community_thresholds_sha256": sha256_file(
                community_thresholds_path
            ),
            "crystal_community_scripts": str(crystal_community_scripts),
            "wl_iters": wl_iters,
            "oxidation_state_handling": (
                "strip opXRD oxidation decorations before applying the "
                "production neutral-element chemistry featurizer"
            ),
            "pca_components": 32,
            "threshold_mode": "per_community_p95",
        },
        "records_requested": len(rows),
        "records_featurized": len(assigned),
        "records_failed": len(errors),
        "max_sites": max_sites,
        "failure_examples": [
            {"record": rows[index]["record"], "error": error}
            for index, error in sorted(errors.items())[:50]
        ],
        "assigned_communities": len(
            {int(row["assigned_community"]) for row in assigned}
        ),
        "outlier_like_records": sum(bool(row["outlier_like"]) for row in assigned),
        "in_basin_records": sum(not bool(row["outlier_like"]) for row in assigned),
        "top_assigned_communities": [
            {"community": community, "count": count}
            for community, count in Counter(
                int(row["assigned_community"]) for row in assigned
            ).most_common(20)
        ],
    }


def public_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def write_candidate_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    public = [public_row(row) for row in rows]
    fields = sorted({key for row in public for key in row})
    preferred = [
        "record",
        "source",
        "institution",
        "wavelength_mode",
        "primary_wavelength",
        "secondary_wavelength",
        "sg_number",
        "sg_symbol",
        "eg_number",
        "crystal_system",
        "reduced_formula",
        "n_sites",
        "two_theta_min",
        "two_theta_max",
        "n_points",
        "native_step",
        "native_step_min",
        "native_step_max",
        "duplicate_status",
        "duplicate_of",
        "selected_unique_pattern",
        "assigned_community",
        "nearest_centroid_distance",
        "community_p95_threshold",
        "community_size",
        "outlier_like",
        "community_pca1",
        "community_pca2",
        "community_error",
        "raw_pattern_hash",
        "normalized_pattern_hash",
        "exact_structure_hash",
    ]
    fieldnames = [field for field in preferred if field in fields]
    fieldnames.extend(field for field in fields if field not in fieldnames)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(public)


def write_window_view(
    output_dir: Path,
    rows: list[dict[str, Any]],
    *,
    lower: float,
    upper: float,
    grid_step: float,
    max_native_step: float,
    endpoint_native_bin_tolerance: float,
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if bool(row["selected_unique_pattern"])
        and float(row["native_step"]) <= max_native_step + 1e-9
        and supports_window(
            row,
            lower,
            upper,
            endpoint_native_bin_tolerance,
        )
    ]
    selected.sort(key=lambda row: str(row["record"]))

    n_points = int(round((upper - lower) / grid_step)) + 1
    grid = lower + np.arange(n_points, dtype=np.float64) * grid_step
    grid[-1] = upper

    patterns = []
    for row in selected:
        values = np.interp(
            grid,
            np.asarray(row["_two_theta"], dtype=np.float64),
            np.asarray(row["_intensities"], dtype=np.float64),
        )
        maximum = float(np.max(np.abs(values)))
        patterns.append(values / maximum if maximum > 0 else values)
    intensity = (
        np.asarray(patterns, dtype=np.float32)
        if patterns
        else np.empty((0, n_points), dtype=np.float32)
    )

    slug = window_slug(lower, upper)
    npz_path = output_dir / f"opxrd_cukalpha_{slug}_maxnorm.npz"
    manifest_path = output_dir / f"opxrd_cukalpha_{slug}_manifest.csv"
    np.savez_compressed(
        npz_path,
        intensity=intensity,
        two_theta=grid.astype(np.float32),
        record=np.asarray([row["record"] for row in selected], dtype=str),
        sg=np.asarray([row["sg_number"] for row in selected], dtype=np.int16),
        eg=np.asarray([row["eg_number"] for row in selected], dtype=np.int16),
        community=np.asarray(
            [row.get("assigned_community", -1) for row in selected],
            dtype=np.int32,
        ),
        community_distance=np.asarray(
            [row.get("nearest_centroid_distance", np.nan) for row in selected],
            dtype=np.float32,
        ),
        community_outlier=np.asarray(
            [bool(row.get("outlier_like", False)) for row in selected],
            dtype=np.bool_,
        ),
        community_assigned=np.asarray(
            [row.get("assigned_community") is not None for row in selected],
            dtype=np.bool_,
        ),
        wavelength_mode=np.asarray(
            [row["wavelength_mode"] for row in selected],
            dtype=str,
        ),
    )

    manifest_fields = [
        "index",
        "record",
        "source",
        "wavelength_mode",
        "primary_wavelength",
        "secondary_wavelength",
        "sg_number",
        "sg_symbol",
        "eg_number",
        "crystal_system",
        "reduced_formula",
        "native_two_theta_min",
        "native_two_theta_max",
        "native_step",
        "strict_window_coverage",
        "endpoint_edge_fill_used",
        "assigned_community",
        "nearest_centroid_distance",
        "community_p95_threshold",
        "community_size",
        "outlier_like",
        "community_error",
        "raw_pattern_hash",
        "exact_structure_hash",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=manifest_fields, lineterminator="\n"
        )
        writer.writeheader()
        for index, row in enumerate(selected):
            writer.writerow(
                {
                    "index": index,
                    "record": row["record"],
                    "source": row["source"],
                    "wavelength_mode": row["wavelength_mode"],
                    "primary_wavelength": row["primary_wavelength"],
                    "secondary_wavelength": row["secondary_wavelength"],
                    "sg_number": row["sg_number"],
                    "sg_symbol": row["sg_symbol"],
                    "eg_number": row["eg_number"],
                    "crystal_system": row["crystal_system"],
                    "reduced_formula": row["reduced_formula"],
                    "native_two_theta_min": row["two_theta_min"],
                    "native_two_theta_max": row["two_theta_max"],
                    "native_step": row["native_step"],
                    "strict_window_coverage": strictly_covers_window(
                        row, lower, upper
                    ),
                    "endpoint_edge_fill_used": not strictly_covers_window(
                        row, lower, upper
                    ),
                    "assigned_community": row.get("assigned_community"),
                    "nearest_centroid_distance": row.get(
                        "nearest_centroid_distance"
                    ),
                    "community_p95_threshold": row.get(
                        "community_p95_threshold"
                    ),
                    "community_size": row.get("community_size"),
                    "outlier_like": row.get("outlier_like"),
                    "community_error": row.get("community_error"),
                    "raw_pattern_hash": row["raw_pattern_hash"],
                    "exact_structure_hash": row["exact_structure_hash"],
                }
            )

    return {
        "window": [lower, upper],
        "grid_step": grid_step,
        "grid_points": n_points,
        "max_native_step": max_native_step,
        "endpoint_native_bin_tolerance": endpoint_native_bin_tolerance,
        "records": len(selected),
        "strictly_covering_records": sum(
            strictly_covers_window(row, lower, upper) for row in selected
        ),
        "records_using_endpoint_edge_fill": sum(
            not strictly_covers_window(row, lower, upper) for row in selected
        ),
        "unique_space_groups": len({int(row["sg_number"]) for row in selected}),
        "unique_extinction_groups": len(
            {int(row["eg_number"]) for row in selected}
        ),
        "assigned_communities": len(
            {
                int(row["assigned_community"])
                for row in selected
                if row.get("assigned_community") is not None
            }
        ),
        "in_basin_records": sum(
            row.get("outlier_like") is False for row in selected
        ),
        "outlier_like_records": sum(
            row.get("outlier_like") is True for row in selected
        ),
        "records_without_community_assignment": sum(
            row.get("assigned_community") is None for row in selected
        ),
        "wavelength_modes": dict(
            Counter(str(row["wavelength_mode"]) for row in selected)
        ),
        "npz": str(npz_path),
        "manifest": str(manifest_path),
    }


def main() -> int:
    args = parse_args()
    windows = parse_windows(args.windows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    community_paths = (
        args.icsd_features,
        args.community_assignments,
        args.community_thresholds,
        args.crystal_community_scripts,
    )
    if not args.skip_community and any(path is None for path in community_paths):
        raise ValueError(
            "community projection requires --icsd-features, "
            "--community-assignments, --community-thresholds, and "
            "--crystal-community-scripts"
        )

    sg_to_eg = load_sg_to_eg(args.sg_eg_table)
    rows, structures, candidate_summary = load_candidates(
        args.archive,
        sg_to_eg,
        wavelength_tolerance=args.wavelength_tolerance,
        symprec=args.symprec,
        angle_tolerance=args.angle_tolerance,
    )

    community_summary = None
    if not args.skip_community:
        community_summary = project_communities(
            rows,
            structures,
            icsd_features_path=args.icsd_features,
            community_assignments_path=args.community_assignments,
            community_thresholds_path=args.community_thresholds,
            crystal_community_scripts=args.crystal_community_scripts,
            wl_iters=args.wl_iters,
            max_sites=args.max_sites,
            n_jobs=args.n_jobs,
            seed=args.seed,
        )

    candidate_csv = args.output_dir / "opxrd_cukalpha_candidates.csv"
    write_candidate_csv(candidate_csv, rows)

    views = [
        write_window_view(
            args.output_dir,
            rows,
            lower=lower,
            upper=upper,
            grid_step=args.grid_step,
            max_native_step=args.max_native_step,
            endpoint_native_bin_tolerance=args.endpoint_native_bin_tolerance,
        )
        for lower, upper in windows
    ]

    summary = {
        "scope": (
            "Single-phase opXRD records with symmetry derivable from the stored "
            "atomic structure and RRUFF-compatible Cu Kalpha radiation"
        ),
        "archive": str(args.archive),
        "archive_size_bytes": args.archive.stat().st_size,
        "archive_sha256": sha256_file(args.archive),
        "sg_eg_table": str(args.sg_eg_table),
        "sg_eg_table_sha256": sha256_file(args.sg_eg_table),
        "software_versions": {
            package: importlib.metadata.version(package)
            for package in (
                "numpy",
                "pymatgen",
                "scikit-learn",
                "matminer",
            )
        },
        "wavelength_rule": {
            "accepted": [
                "explicit Cu Kalpha1/Kalpha2 pair in either order",
                "2:1 weighted-average Cu Kalpha wavelength",
            ],
            "excluded": [
                "monochromated Cu Kalpha1-only",
                "monochromated Cu Kalpha2-only",
                "other wavelengths",
            ],
            "kalpha1_angstrom": CU_KALPHA1,
            "kalpha2_angstrom": CU_KALPHA2,
            "weighted_average_angstrom": CU_KALPHA_WEIGHTED_AVERAGE,
            "absolute_tolerance_angstrom": args.wavelength_tolerance,
        },
        "symmetry_parameters": {
            "symprec": args.symprec,
            "angle_tolerance": args.angle_tolerance,
        },
        "candidate_audit": candidate_summary,
        "community_projection": community_summary,
        "candidate_csv": str(candidate_csv),
        "views": views,
        "synthetic_data_implication": {
            "new_wavelength_simulation_required": False,
            "reason": (
                "Accepted records use the same explicit Cu Kalpha1/Kalpha2 "
                "or 2:1 weighted-average convention as RRUFF."
            ),
            "range_adaptation": (
                "A narrower opXRD view requires cropping the existing 5-90 "
                "degree synthetic patterns to the same window and retraining "
                "the compared models; it does not require resimulating the "
                "diffraction patterns."
            ),
        },
    }
    summary_path = args.output_dir / "opxrd_cukalpha_community_surface_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
