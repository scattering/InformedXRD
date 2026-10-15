#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import CFML_api
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test the low-level ILL CFML_api powder-pattern path.")
    parser.add_argument("--cif", required=True, help="Path to CIF file")
    parser.add_argument("--theta-min", type=float, default=0.0)
    parser.add_argument("--theta-max", type=float, default=120.0)
    parser.add_argument("--u-resolution", type=float, default=0.5)
    parser.add_argument("--po-r", type=float, default=1.0)
    parser.add_argument("--po-h", type=float, default=0.0)
    parser.add_argument("--po-k", type=float, default=0.0)
    parser.add_argument("--po-l", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cif_path = Path(args.cif).expanduser().resolve()

    cif = CFML_api.CIFFile(str(cif_path))
    cell = cif.cell
    space_group = cif.space_group
    atom_list = cif.atom_list
    job_info = cif.job_info

    job_info.range_2theta = (args.theta_min, args.theta_max)
    job_info.u_resolution = args.u_resolution
    job_info.po_r = args.po_r
    job_info.po_h = args.po_h
    job_info.po_k = args.po_k
    job_info.po_l = args.po_l

    reflection_list = CFML_api.ReflectionList(cell, space_group, True, job_info)
    reflection_list.compute_structure_factors(space_group, atom_list, job_info)

    pattern = CFML_api.DiffractionPattern(job_info, reflection_list, cell, cell.reciprocal_cell_vol)
    intensities = np.asarray(pattern.ycalc, dtype=np.float64)

    print("cif", cif_path)
    print("nref", reflection_list.nref)
    print("npts", pattern.npts)
    print("xmin", float(pattern.xmin))
    print("xmax", float(pattern.xmax))
    print("step", float(pattern.step))
    print("ymax", float(np.max(intensities)))
    print("y0_5", [float(v) for v in intensities[:5]])


if __name__ == "__main__":
    main()
