#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import CFML_api
import numpy as np


def powder_pattern_from_cif(
    cif_path: str | Path,
    theta_min: float = 0.0,
    theta_max: float = 120.0,
    u_resolution: float = 0.5,
    po_r: float = 1.0,
    po_h: float = 0.0,
    po_k: float = 0.0,
    po_l: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    cif = CFML_api.CIFFile(str(Path(cif_path).expanduser().resolve()))
    cell = cif.cell
    space_group = cif.space_group
    atom_list = cif.atom_list
    job_info = cif.job_info

    job_info.range_2theta = (theta_min, theta_max)
    job_info.u_resolution = u_resolution
    job_info.po_r = po_r
    job_info.po_h = po_h
    job_info.po_k = po_k
    job_info.po_l = po_l

    reflection_list = CFML_api.ReflectionList(cell, space_group, True, job_info)
    reflection_list.compute_structure_factors(space_group, atom_list, job_info)

    pattern = CFML_api.DiffractionPattern(job_info, reflection_list, cell, cell.reciprocal_cell_vol)
    x = np.asarray(pattern.x, dtype=np.float32)
    y = np.asarray(pattern.ycalc, dtype=np.float32)
    return x, y
