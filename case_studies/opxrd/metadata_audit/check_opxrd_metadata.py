#!/usr/bin/env python3
from __future__ import annotations

import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("<DATA_ROOT>/AI and Diffraction/OpXRD")
ZIP_PATH = ROOT / "opxrd.zip"
SUMMARY_JSON = ROOT / "opxrd_metadata_summary.json"
SUMMARY_MD = ROOT / "opxrd_metadata_summary.md"


def is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, tuple, dict)):
        return len(value) == 0
    return False


def phase_has_lattice(phase: dict) -> bool:
    lattice = phase.get("lattice")
    if isinstance(lattice, dict):
        vals = [v for v in lattice.values() if not is_missing(v)]
        return len(vals) > 0
    if isinstance(lattice, (list, tuple)):
        return any(not is_missing(v) for v in lattice)
    return not is_missing(lattice)


def phase_has_spacegroup(phase: dict) -> bool:
    return not is_missing(phase.get("spacegroup"))


def get_xray_info(label: dict) -> dict:
    value = label.get("xray_info") or {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
    return {}


def normalize_label(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
    return {}


def normalize_phase(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
    return {}


def near(value: float | None, target: float, tol: float = 5e-4) -> bool:
    if value is None:
        return False
    try:
        return abs(float(value) - target) <= tol
    except Exception:
        return False


def normalize_source(top_level: str) -> str:
    mapping = {
        "CNRS": "CNRS / PSL University",
        "CNRS /PSL University": "CNRS / PSL University",
        "PSL University": "CNRS / PSL University",
        "LBNL": "LBNL",
        "KIT": "KIT",
        "EMPA": "Empa",
        "Empa": "Empa",
        "USC": "University of Southern California",
        "University of Southern California": "University of Southern California",
    }
    return mapping.get(top_level, top_level)


def main() -> None:
    total_patterns = 0
    patterns_with_phases = 0
    patterns_with_lattice = 0
    patterns_with_spacegroup = 0
    patterns_with_both_pattern_level = 0
    phase_entries_with_both = 0

    cu_kalpha1 = 0
    cu_kalpha2 = 0
    both_cu_kalpha = 0
    only_cu_kalpha1 = 0
    tertiary_used = 0

    source_counts = Counter()
    source_lattice = Counter()
    source_spacegroup = Counter()
    source_both_pattern = Counter()

    combo_counts = Counter()

    with zipfile.ZipFile(ZIP_PATH) as zf:
        names = [n for n in zf.namelist() if n.endswith(".json")]
        total_patterns = len(names)
        for name in names:
            top_level = name.split("/", 1)[0]
            source = normalize_source(top_level)
            source_counts[source] += 1

            with zf.open(name) as fh:
                data = json.load(fh)

            label = normalize_label(data.get("label") or {})
            phases = label.get("phases") or []
            if phases:
                patterns_with_phases += 1

            any_lattice = False
            any_spacegroup = False

            for raw_phase in phases:
                phase = normalize_phase(raw_phase)
                has_lat = phase_has_lattice(phase)
                has_sg = phase_has_spacegroup(phase)
                any_lattice = any_lattice or has_lat
                any_spacegroup = any_spacegroup or has_sg
                if has_lat and has_sg:
                    phase_entries_with_both += 1

            if any_lattice:
                patterns_with_lattice += 1
                source_lattice[source] += 1
            if any_spacegroup:
                patterns_with_spacegroup += 1
                source_spacegroup[source] += 1
            if any_lattice and any_spacegroup:
                patterns_with_both_pattern_level += 1
                source_both_pattern[source] += 1

            xray = get_xray_info(label)
            p1 = xray.get("primary_wavelength")
            p2 = xray.get("secondary_wavelength")
            p3 = xray.get("tertiary_wavelength")
            has_k1 = near(p1, 1.54056) or near(p2, 1.54056)
            has_k2 = near(p1, 1.54439) or near(p2, 1.54439)
            if has_k1:
                cu_kalpha1 += 1
            if has_k2:
                cu_kalpha2 += 1
            if has_k1 and has_k2:
                both_cu_kalpha += 1
            if has_k1 and not has_k2:
                only_cu_kalpha1 += 1
            if not is_missing(p3):
                tertiary_used += 1
            combo_counts[(p1, p2)] += 1

    common_combos = [
        {
            "primary_wavelength": k[0],
            "secondary_wavelength": k[1],
            "count": v,
        }
        for k, v in combo_counts.most_common(10)
    ]

    summary = {
        "zip_path": str(ZIP_PATH),
        "total_patterns": total_patterns,
        "patterns_with_phases": patterns_with_phases,
        "patterns_with_lattice": patterns_with_lattice,
        "patterns_with_spacegroup": patterns_with_spacegroup,
        "patterns_with_both_pattern_level": patterns_with_both_pattern_level,
        "phase_entries_with_both": phase_entries_with_both,
        "wavelengths": {
            "cu_kalpha1_like": cu_kalpha1,
            "cu_kalpha2_like": cu_kalpha2,
            "both_cu_kalpha1_kalpha2": both_cu_kalpha,
            "only_cu_kalpha1_like": only_cu_kalpha1,
            "tertiary_wavelength_used": tertiary_used,
            "common_primary_secondary_combinations": common_combos,
        },
        "sources": {
            "total_patterns": dict(source_counts),
            "patterns_with_lattice": dict(source_lattice),
            "patterns_with_spacegroup": dict(source_spacegroup),
            "patterns_with_both_pattern_level": dict(source_both_pattern),
        },
    }

    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))

    md = []
    md.append("# OpXRD Metadata Check")
    md.append("")
    md.append(f"- Archive: `{ZIP_PATH}`")
    md.append(f"- Total pattern entries: `{total_patterns}`")
    md.append(f"- Patterns with any phase labels: `{patterns_with_phases}`")
    md.append(f"- Patterns with lattice parameters: `{patterns_with_lattice}`")
    md.append(f"- Patterns with space group: `{patterns_with_spacegroup}`")
    md.append(f"- Patterns with both lattice and space group somewhere in the same pattern: `{patterns_with_both_pattern_level}`")
    md.append(f"- Phase-level entries with both lattice and space group on the same phase: `{phase_entries_with_both}`")
    md.append("")
    md.append("## Wavelength Summary")
    md.append("")
    md.append(f"- Cu Kalpha1-like (`~1.54056 A`): `{cu_kalpha1}`")
    md.append(f"- Cu Kalpha2-like (`~1.54439 A`): `{cu_kalpha2}`")
    md.append(f"- Both Cu Kalpha1 and Kalpha2: `{both_cu_kalpha}`")
    md.append(f"- Only Cu Kalpha1-like: `{only_cu_kalpha1}`")
    md.append(f"- Tertiary wavelength used: `{tertiary_used}`")
    md.append("")
    md.append("## Source Counts")
    md.append("")
    for source in sorted(source_counts):
        md.append(
            f"- {source}: total `{source_counts[source]}`, lattice `{source_lattice[source]}`, "
            f"space group `{source_spacegroup[source]}`, both `{source_both_pattern[source]}`"
        )
    md.append("")
    md.append("## Common Wavelength Combinations")
    md.append("")
    for row in common_combos:
        md.append(
            f"- primary `{row['primary_wavelength']}`, secondary `{row['secondary_wavelength']}`: `{row['count']}`"
        )
    md.append("")
    md.append(f"Machine-readable summary: `{SUMMARY_JSON}`")
    SUMMARY_MD.write_text("\n".join(md) + "\n")

    print(f"Wrote {SUMMARY_JSON}")
    print(f"Wrote {SUMMARY_MD}")


if __name__ == "__main__":
    main()
