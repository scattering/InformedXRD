#!/usr/bin/env python3
from __future__ import annotations

import ast
import csv
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
import argparse

import h5py
import numpy as np
from pymatgen.symmetry.groups import SpaceGroup
from scipy.signal import find_peaks


CU_KA_WAVELENGTH = 1.5406
ALPHADIFFRACT_WAVELENGTH = 12.398419843320026 / 20.0
CELL_RE = re.compile(r"CELL PARAMETERS:\s+([^\n\r]+)")
SG_RE = re.compile(r"SPACE GROUP:\s*([^\n\r]+)")
WAVE_RE = re.compile(r"X-RAY WAVELENGTH:\s*([^\n\r]+)")
LOC_RE = re.compile(r"Locality:\s*(.*)")


def sample_key(filename: str) -> str:
    return filename.split("__Powder__")[0]


def parse_xy_text(text: str) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("##"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            xs.append(float(parts[0]))
            ys.append(float(parts[1]))
        except Exception:
            continue
    return np.asarray(xs), np.asarray(ys)


def two_theta_to_q(two_theta_deg: np.ndarray, wavelength: float) -> np.ndarray:
    theta_rad = np.deg2rad(two_theta_deg / 2.0)
    return (4.0 * np.pi * np.sin(theta_rad)) / wavelength


def normalize_to_0_100(y: np.ndarray) -> np.ndarray:
    y = np.clip(y, 0, None)
    lo = float(np.min(y))
    hi = float(np.max(y))
    d = max(hi - lo, 1e-10)
    return ((y - lo) / d) * 100.0


def cell_dist(a: list[float], b: list[float]) -> float:
    acc = 0.0
    n = 0
    for x, y in zip(a, b):
        scale = max(abs(x), abs(y), 1.0)
        acc += ((x - y) / scale) ** 2
        n += 1
    return math.sqrt(acc / n) if n else 0.0


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


class DSU:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.p[rb] = ra


def hm_symbol_to_number(symbol: str) -> int | None:
    raw = symbol.strip().replace("\\par", "").strip()
    candidates = [raw]
    s = raw.replace("_", "")
    s = re.sub(r"^([A-ZR])(?=[0-9-])", r"\1 ", s)
    candidates.append(s)
    s2 = raw.replace("_", "")
    s2 = re.sub(r"^([A-ZR])(?=[0-9-])", r"\1 ", s2)
    s2 = s2.replace("/", " /")
    candidates.append(s2)
    seen = []
    for cand in candidates:
        cand = " ".join(cand.split())
        if cand and cand not in seen:
            seen.append(cand)
    for cand in seen:
        try:
            return int(SpaceGroup(cand).int_number)
        except Exception:
            pass
    return None


def freq_baseline(y: np.ndarray) -> dict:
    c = Counter(map(int, y.tolist()))
    ranked = [cls for cls, _ in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))]
    top5 = ranked[:5]
    return {
        "n": int(len(y)),
        "n_classes": int(len(c)),
        "top1_class": int(ranked[0]),
        "top5_classes": [int(v) for v in top5],
        "top1": float(sum(1 for v in y if int(v) == ranked[0]) / len(y)),
        "top5": float(sum(1 for v in y if int(v) in set(top5)) / len(y)),
        "top_counts": [[int(cls), int(c[cls])] for cls in ranked[:10]],
    }


def load_sg_to_eg(mapping_csv: Path) -> np.ndarray:
    sg_to_eg = np.full(231, -1, dtype=np.int64)
    with open(mapping_csv, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            eg_index = int(row["Index"])
            sg_list = ast.literal_eval(row["Space Group Numbers"])
            for sg in sg_list:
                sg_to_eg[int(sg)] = eg_index
    return sg_to_eg


def build_proc612_labels(processed_zip: Path, dif_zip: Path, sg_to_eg: np.ndarray) -> np.ndarray:
    target_two_theta = np.linspace(5.0, 20.0, 8192, dtype=np.float64)
    target_q = two_theta_to_q(target_two_theta, ALPHADIFFRACT_WAVELENGTH)

    rows = []
    with zipfile.ZipFile(processed_zip) as zproc, zipfile.ZipFile(dif_zip) as zdif:
        proc = {
            sample_key(Path(n).name): n
            for n in zproc.namelist()
            if Path(n).name.endswith(".txt") and "Xray_Data_XY_Processed" in Path(n).name
        }
        dif = {
            sample_key(Path(n).name): n
            for n in zdif.namelist()
            if Path(n).name.endswith(".txt") and "DIF_File" in Path(n).name
        }
        keys = sorted(set(proc) & set(dif))
        for key in keys:
            dtxt = zdif.read(dif[key]).decode("utf-8", errors="ignore")
            m_cell = CELL_RE.search(dtxt)
            m_sg = SG_RE.search(dtxt)
            m_wave = WAVE_RE.search(dtxt)
            m_loc = LOC_RE.search(dtxt)
            vals = []
            if m_cell:
                for v in re.split(r"\s+", m_cell.group(1).strip())[:6]:
                    try:
                        vals.append(float(v))
                    except Exception:
                        break
            has_cell = len(vals) == 6
            has_sg = bool(m_sg and m_sg.group(1).strip())
            wave = None
            if m_wave:
                try:
                    wave = float(m_wave.group(1).split()[0])
                except Exception:
                    wave = None
            locality = m_loc.group(1).strip() if m_loc else ""
            xtxt = zproc.read(proc[key]).decode("utf-8", errors="ignore")
            xs, ys = parse_xy_text(xtxt)
            if xs.size < 10:
                continue
            src_q = two_theta_to_q(xs, CU_KA_WAVELENGTH)
            y_interp = np.interp(target_q, src_q, ys, left=0.0, right=0.0)
            y_norm = normalize_to_0_100(y_interp)
            peaks, _ = find_peaks(y_norm, prominence=1.0, distance=8)
            rows.append(
                {
                    "key": key,
                    "name": key.split("__")[0],
                    "cell": vals if has_cell else None,
                    "sg_symbol": m_sg.group(1).strip() if has_sg else None,
                    "has_cell": has_cell,
                    "has_sg": has_sg,
                    "wave": wave,
                    "is_synthetic_locality": "synthetic" in locality.lower(),
                    "peak_count": int(len(peaks)),
                    "pattern": y_norm.astype(np.float32),
                }
            )

    filtered = [
        row
        for row in rows
        if row["has_cell"]
        and row["has_sg"]
        and row["wave"] is not None
        and 1.53 <= row["wave"] <= 1.55
        and not row["is_synthetic_locality"]
        and row["peak_count"] >= 3
    ]
    by_group: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in filtered:
        by_group[(row["name"], row["sg_symbol"])].append(row)

    selected = []
    for _, group in by_group.items():
        dsu = DSU(len(group))
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                if cell_dist(group[i]["cell"], group[j]["cell"]) > 0.002:
                    continue
                if corr(group[i]["pattern"], group[j]["pattern"]) < 0.6:
                    continue
                dsu.union(i, j)
        comps: dict[int, list[dict]] = defaultdict(list)
        for i, row in enumerate(group):
            comps[dsu.find(i)].append(row)
        for comp in comps.values():
            rep = sorted(comp, key=lambda row: (-row["peak_count"], row["key"]))[0]
            selected.append(rep)

    selected = sorted(selected, key=lambda row: row["key"])
    y = []
    for row in selected:
        sg_num = hm_symbol_to_number(row["sg_symbol"])
        if sg_num is None or sg_to_eg[sg_num] < 0:
            continue
        y.append(int(sg_to_eg[sg_num]))
    return np.asarray(y, dtype=np.int64)


def parse_args():
    p = argparse.ArgumentParser(description="Compute label-frequency baselines for 473, proc209, and proc612 slices.")
    p.add_argument("--full-h5", required=True)
    p.add_argument("--subset209-json", required=True)
    p.add_argument("--processed-zip", required=True)
    p.add_argument("--dif-zip", required=True)
    p.add_argument("--mapping-csv", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    full_h5 = Path(args.full_h5)
    subset209_json = Path(args.subset209_json)
    processed_zip = Path(args.processed_zip)
    dif_zip = Path(args.dif_zip)
    mapping_csv = Path(args.mapping_csv)

    sg_to_eg = load_sg_to_eg(mapping_csv)

    with h5py.File(full_h5, "r") as handle:
        y473 = handle["y_test"][:].astype(np.int64)
        case_ids = [(x.decode("utf-8") if isinstance(x, bytes) else str(x)) for x in handle["case_ids"][:]]

    case_to_y = dict(zip(case_ids, y473))
    subset209 = json.loads(subset209_json.read_text())
    y209 = np.asarray([case_to_y[row["case_id"]] for row in subset209["rows"]], dtype=np.int64)
    y612 = build_proc612_labels(processed_zip, dif_zip, sg_to_eg)

    out = {
        "baseline_full473": freq_baseline(y473),
        "baseline_proc209": freq_baseline(y209),
        "baseline_proc612": freq_baseline(y612),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
