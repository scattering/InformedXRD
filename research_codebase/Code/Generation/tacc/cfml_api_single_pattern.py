#!/usr/bin/env python3
from __future__ import annotations

import argparse

import CFML_api
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single CFML_api powder-pattern calculation in an isolated process.")
    parser.add_argument("--cif", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--theta-min", type=float, required=True)
    parser.add_argument("--theta-max", type=float, required=True)
    parser.add_argument("--step", type=float, required=True)
    parser.add_argument("--u-resolution", type=float, required=True)
    parser.add_argument("--po-r", type=float, default=1.0)
    parser.add_argument("--po-h", type=float, default=0.0)
    parser.add_argument("--po-k", type=float, default=0.0)
    parser.add_argument("--po-l", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cif = CFML_api.CIFFile(args.cif)
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
    x = np.asarray(pattern.x, dtype=np.float32)
    y = np.asarray(pattern.ycalc, dtype=np.float32)

    if args.step > 0 and x.shape[0] > 1:
        expected_n = int(round((args.theta_max - args.theta_min) / args.step)) + 1
        target_x = np.linspace(args.theta_min, args.theta_max, expected_n, dtype=np.float32)
        if x.shape[0] != expected_n or not np.allclose(x, target_x, atol=1e-6):
            y = np.interp(target_x, x, y, left=0.0, right=0.0).astype(np.float32)
            x = target_x

    np.savez(args.out, x=x, y=y)


if __name__ == "__main__":
    main()
