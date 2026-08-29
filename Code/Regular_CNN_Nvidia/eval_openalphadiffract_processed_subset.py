#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks


CU_KA_WAVELENGTH = 1.5406
ALPHADIFFRACT_WAVELENGTH = 12.398419843320026 / 20.0
CELL_RE = re.compile(r"CELL PARAMETERS:\s+([^\n\r]+)")
SG_RE = re.compile(r"SPACE GROUP:\s*([^\n\r]+)")
WAVE_RE = re.compile(r"X-RAY WAVELENGTH:\s*([^\n\r]+)")
LOC_RE = re.compile(r"Locality:\s*(.*)")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate OpenAlphaDiffract on a processed-RRUFF reconstructed subset.")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--processed-zip", required=True)
    p.add_argument("--dif-zip", required=True)
    p.add_argument("--mapping-csv", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--corr-threshold", type=float, default=0.6)
    p.add_argument("--cell-threshold", type=float, default=0.002)
    p.add_argument("--min-peaks", type=int, default=3)
    return p.parse_args()


def sample_key(filename: str) -> str:
    return filename.split("__Powder__")[0]


def load_sg_to_eg_map(mapping_csv: str) -> np.ndarray:
    sg_to_eg = np.full(231, -1, dtype=np.int64)
    with open(mapping_csv, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            eg_index = int(row["Index"])
            sg_list = ast.literal_eval(row["Space Group Numbers"])
            for sg in sg_list:
                sg_to_eg[int(sg)] = eg_index
    missing = np.where(sg_to_eg[1:] < 0)[0]
    if len(missing):
        raise ValueError(f"Missing SG->EG mapping for SGs: {(missing + 1).tolist()}")
    return sg_to_eg


def collapse_sg_probs_to_eg(sg_probs: np.ndarray, sg_to_eg: np.ndarray) -> np.ndarray:
    eg_count = int(sg_to_eg.max()) + 1
    eg_probs = np.zeros((sg_probs.shape[0], eg_count), dtype=np.float32)
    for sg in range(1, 231):
        eg_probs[:, sg_to_eg[sg]] += sg_probs[:, sg - 1]
    return eg_probs


def topk_accuracy(probs: np.ndarray, targets: np.ndarray, k: int) -> float:
    topk = np.argpartition(-probs, kth=min(k, probs.shape[1]) - 1, axis=1)[:, :k]
    hits = (topk == targets[:, None]).any(axis=1)
    return float(hits.mean())


def two_theta_to_q(two_theta_deg: np.ndarray, wavelength: float) -> np.ndarray:
    theta_rad = np.deg2rad(two_theta_deg / 2.0)
    return (4.0 * np.pi * np.sin(theta_rad)) / wavelength


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
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def normalize_to_0_100(y: np.ndarray) -> np.ndarray:
    y = np.clip(y, 0, None)
    if y.size == 0:
        return y
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
            import gemmi

            sg = gemmi.find_spacegroup_by_name(cand)
            if sg is not None:
                return int(sg.number)
        except Exception:
            pass
        try:
            from pymatgen.symmetry.groups import SpaceGroup

            return int(SpaceGroup(cand).int_number)
        except Exception:
            pass
    return None


def main():
    args = parse_args()

    sys.path.insert(0, str(Path(args.model_dir).resolve()))
    import torch
    from model import AlphaDiffract

    sg_to_eg = load_sg_to_eg_map(args.mapping_csv)
    target_two_theta = np.linspace(5.0, 20.0, 8192, dtype=np.float64)
    target_q = two_theta_to_q(target_two_theta, ALPHADIFFRACT_WAVELENGTH)

    with zipfile.ZipFile(args.processed_zip) as zproc, zipfile.ZipFile(args.dif_zip) as zdif:
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
        rows = []
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
                    "rid": key.split("__")[1],
                    "cell": vals if has_cell else None,
                    "sg_symbol": m_sg.group(1).strip() if has_sg else None,
                    "has_cell": has_cell,
                    "has_sg": has_sg,
                    "wave": wave,
                    "locality": locality,
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
        and row["peak_count"] >= args.min_peaks
    ]

    by_group: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in filtered:
        by_group[(row["name"], row["sg_symbol"])].append(row)

    selected = []
    clusters = []
    for key, group in by_group.items():
        dsu = DSU(len(group))
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                if cell_dist(group[i]["cell"], group[j]["cell"]) > args.cell_threshold:
                    continue
                if corr(group[i]["pattern"], group[j]["pattern"]) < args.corr_threshold:
                    continue
                dsu.union(i, j)
        comps: dict[int, list[dict]] = defaultdict(list)
        for i, row in enumerate(group):
            comps[dsu.find(i)].append(row)
        for comp in comps.values():
            rep = sorted(comp, key=lambda row: (-row["peak_count"], row["key"]))[0]
            selected.append(rep)
            clusters.append(
                {
                    "name": key[0],
                    "sg_symbol": key[1],
                    "cluster_size": len(comp),
                    "representative_key": rep["key"],
                    "members": [row["key"] for row in comp],
                }
            )

    selected = sorted(selected, key=lambda row: row["key"])
    y = []
    x = []
    kept_keys = []
    sg_numbers = []
    for row in selected:
        sg_num = hm_symbol_to_number(row["sg_symbol"])
        if sg_num is None or sg_to_eg[sg_num] < 0:
            continue
        y.append(int(sg_to_eg[sg_num]))
        x.append(row["pattern"])
        kept_keys.append(row["key"])
        sg_numbers.append(sg_num)

    x_arr = np.stack(x, axis=0).astype(np.float32)
    y_arr = np.asarray(y, dtype=np.int64)
    model = AlphaDiffract.from_pretrained(args.model_dir, device="cpu")

    sg_chunks = []
    with torch.no_grad():
        for start in range(0, x_arr.shape[0], 64):
            batch = torch.from_numpy(x_arr[start : start + 64])
            out = model(batch)
            sg_chunks.append(torch.softmax(out["sg_logits"], dim=-1).cpu().numpy().astype(np.float32))
    sg_probs = np.concatenate(sg_chunks, axis=0)
    eg_probs = collapse_sg_probs_to_eg(sg_probs, sg_to_eg)

    result = {
        "processed_zip": args.processed_zip,
        "dif_zip": args.dif_zip,
        "corr_threshold": args.corr_threshold,
        "cell_threshold": args.cell_threshold,
        "min_peaks": args.min_peaks,
        "n_rows_all_pairs": len(rows),
        "n_rows_filtered": len(filtered),
        "n_clusters_selected": len(selected),
        "n_cases_evaluable": int(x_arr.shape[0]),
        "top1_eg": topk_accuracy(eg_probs, y_arr, 1),
        "top5_eg": topk_accuracy(eg_probs, y_arr, 5),
        "mean_max_sg_prob": float(sg_probs.max(axis=1).mean()),
        "mean_max_eg_prob": float(eg_probs.max(axis=1).mean()),
        "sample_keys": kept_keys[:20],
        "sample_sg_numbers": sg_numbers[:20],
        "clusters": clusters[:50],
    }

    Path(args.output_json).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
