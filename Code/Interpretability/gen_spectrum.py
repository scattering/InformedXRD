"""
Step 1: Generate synthetic XRD spectrum + HKL peaks from the DB using cctbx.
Run this in the cctbx env (no matplotlib, no torch needed).
Saves one .npz + one .json per sample index to OUTPUT_DIR.
"""

import os
import csv
import json
import sqlite3
import numpy as np
from math import sin, asin, radians, degrees

from cctbx import crystal

SAMPLE_INDICES = [16, 26]
DB_PATH    = os.path.join(os.path.dirname(__file__), "intrep_crystal_xrd.db")
CSV_PATH   = os.path.join(os.path.dirname(__file__), "interp_metadata_clean.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "gen_output")

TWO_THETA_MIN = 10.0
TWO_THETA_MAX = 90.0
SPEC_LENGTH   = 8501
WAVELENGTH    = 1.541838


# ── crystallography helpers ──────────────────────────────────────────────────
def twotheta_to_d(tt):
    return WAVELENGTH / (2 * sin(radians(tt) / 2))

def d_to_twotheta(d):
    return 2 * degrees(asin(WAVELENGTH / (2 * d)))

def calcgaussian(x, fwhm):
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return np.exp(-x**2 / (2 * sigma**2))

class Peak:
    def __init__(self, center, u, v, w, I):
        self.center = center
        self.u = u; self.v = v; self.w = w; self.I = I

    @property
    def H(self):
        return np.sqrt(max(self.u * np.tan(radians(self.center / 2))**2
                          + self.v * np.tan(radians(self.center / 2))
                          + self.w, 1e-6))

    def __call__(self, x):
        return self.I * calcgaussian(x - self.center, self.H)


def generate_pattern(spacegroup_number, unit_cell,
                     twotheta_min=5.0, twotheta_max=90.0, twothstep=0.01,
                     u=0.01, v=-0.01, w=0.02, peak_scale=1000):
    symm = crystal.symmetry(unit_cell=unit_cell, space_group=spacegroup_number)
    d_max = twotheta_to_d(twotheta_min)
    d_min = twotheta_to_d(twotheta_max)
    miller_set = symm.build_miller_set(anomalous_flag=False, d_min=d_min, d_max=d_max)
    uc = symm.unit_cell()

    twotheta = np.arange(twotheta_min, twotheta_max, twothstep)
    I = np.zeros(len(twotheta))
    for hkl in miller_set.indices():
        d = uc.d(hkl)
        tt_c = 2 * np.arcsin(WAVELENGTH / (2 * d)) * 180 / np.pi
        ps = np.random.uniform(5, peak_scale)
        I += Peak(tt_c, u, v, w, ps)(twotheta)

    lorentz = 1.0 / (np.sin(np.radians(twotheta)) * np.sin(np.radians(twotheta / 2)))
    I *= lorentz
    return twotheta, I


def get_hkl_peaks(spacegroup_number, unit_cell, twotheta_min, twotheta_max):
    symm = crystal.symmetry(unit_cell=unit_cell, space_group=spacegroup_number)
    d_max = twotheta_to_d(twotheta_min)
    d_min = twotheta_to_d(twotheta_max)
    miller_set = symm.build_miller_set(anomalous_flag=False, d_min=d_min, d_max=d_max)
    uc = symm.unit_cell()
    results = sorted(
        [(d_to_twotheta(uc.d(hkl)), hkl) for hkl in miller_set.indices()],
        key=lambda x: x[0]
    )
    twotheta_peaks = np.array([r[0] for r in results])
    hkls = [list(r[1]) for r in results]   # list of [h,k,l] — JSON-serialisable
    return twotheta_peaks, hkls


# ── main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load CSV index
    hdf5_to_structure = {}
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(row for row in f if not row.lstrip().startswith("#"))
        for row in reader:
            hdf5_to_structure[int(row["hdf5_index"])] = row["structure_id"]

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    for sample_index in SAMPLE_INDICES:
        print(f"\n── Sample {sample_index} ──")
        structure_id = hdf5_to_structure[sample_index]
        cur.execute("SELECT * FROM Structures WHERE structure_id = ?", (structure_id,))
        row = cur.fetchone()
        if row is None:
            print(f"  Not found in DB, skipping.")
            continue

        extinction_group  = row[3]
        crystal_system    = row[4]
        spacegroup_number = row[5]
        unit_cell = (row[7], row[8], row[9], row[10], row[11], row[12])
        print(f"  SG {spacegroup_number} | {crystal_system} | extinction group {extinction_group}")

        # Generate at fine resolution then resample to model range/length
        tt_fine, I_fine = generate_pattern(spacegroup_number, unit_cell)
        tt_model = np.linspace(TWO_THETA_MIN, TWO_THETA_MAX, SPEC_LENGTH)
        I_model  = np.interp(tt_model, tt_fine, I_fine)
        print(f"  Pattern: {SPEC_LENGTH} points over {TWO_THETA_MIN}–{TWO_THETA_MAX}°")

        tt_peaks, hkls = get_hkl_peaks(spacegroup_number, unit_cell, TWO_THETA_MIN, TWO_THETA_MAX)
        print(f"  HKL peaks: {len(tt_peaks)}")

        # Save arrays
        npz_path = os.path.join(OUTPUT_DIR, f"sample_{sample_index}.npz")
        np.savez(npz_path, twotheta=tt_model, intensity=I_model, twotheta_peaks=tt_peaks)

        # Save metadata + hkls
        meta = {
            "sample_index":     sample_index,
            "structure_id":     structure_id,
            "spacegroup_number": spacegroup_number,
            "extinction_group": extinction_group,
            "crystal_system":   crystal_system,
            "hkls":             hkls,
        }
        json_path = os.path.join(OUTPUT_DIR, f"sample_{sample_index}.json")
        with open(json_path, "w") as f:
            json.dump(meta, f)

        print(f"  Saved: {npz_path}")
        print(f"  Saved: {json_path}")

    conn.close()
    print("\nDone.")
