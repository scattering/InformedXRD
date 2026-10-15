# merged_diffraction_gen.py

import datetime
import numpy as np
from numpy import sin, radians, tan, sqrt
import h5py
import matplotlib.pyplot as plt
import spglib
import pyxtal
from cctbx import crystal
import argparse
from multiprocessing import Pool, cpu_count
import pickle
import random
import h5py
import numpy as np
import datetime

# --- Instrument Parameters ---
INSTRUMENT_PARAMS = {
    "wavelength": 1.54030,
    "twotheta_min": 10,
    "twotheta_max": 110,
    "step_size": 0.1, #CHNAGED THIS FROM 0.1 to 0.05
    "u": 0.025,
    "v": -0.025,
    "w": 0.025
}
#with open('SG_To_Ext_Map.pkl', 'rb') as file:
with open('Ext_to_SG_Map.pkl', 'rb') as file:
    mapping = pickle.load(file)

# --- CLI Args ---
parser = argparse.ArgumentParser()
parser.add_argument("--num_samples", type=int, default=2300)
parser.add_argument("--output", type=str, default="diffraction_patterns.h5")
parser.add_argument("--ncpus", type=int, default=cpu_count())
args = parser.parse_args()

OUTPUT_HDF5 = args.output
NUM_SAMPLES = args.num_samples
NUM_PROCESSES = args.ncpus


def twotheta_to_d(twotheta, wavelength=1.54):
    return wavelength / (2 * sin(radians(twotheta) / 2))


def calcgaussian(x, fwhm):
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return np.exp(-0.5 * ((x / sigma) ** 2))


class Peak:
    scaleFactor = 1
    def __init__(self, center, u, v, w, I, hkl=[None, None, None], shape="Gaussian", eta=0):
        self.center = center
        self.u, self.v, self.w = u, v, w
        self.I = I
        self.shape = shape
        self.eta = eta
        try:
            self.H = sqrt(u * (tan(radians(center / 2))**2) + v * tan(radians(center / 2)) + w)
            self.scale = self.I * Peak.scaleFactor
        except ValueError:
            self.H = 0
            self.scale = 0
        self.hkl = hkl

    def __call__(self, x):
        return self.scale * calcgaussian(x - self.center, self.H)


def generate_unit_cell(spacegroup_number):
    group = pyxtal.symmetry.Group(spacegroup_number)
    lattice_type = group.lattice_type
    a, b, c = np.random.uniform(4.0, 10.0, 3)
    alpha, beta, gamma = np.random.uniform(60.0, 120.0, 3)

    if lattice_type == "monoclinic":
        alpha = gamma = 90.0
        beta = np.random.uniform(90.5, 120.0)
    elif lattice_type == "orthorhombic":
        alpha = beta = gamma = 90.0
    elif lattice_type == "tetragonal":
        alpha = beta = gamma = 90.0
        b = a
    elif lattice_type == "cubic":
        alpha = beta = gamma = 90.0
        b = c = a
    elif lattice_type in ["hexagonal", "trigonal"]:
        alpha = beta = 90.0
        gamma = 120.0
        b = c = a

    return a, b, c, alpha, beta, gamma


def generate_diffraction_pattern(spacegroup_number, a, b, c, alpha, beta, gamma):
    symmetry = crystal.symmetry(unit_cell=(a, b, c, alpha, beta, gamma), space_group=spacegroup_number)
    d_min = twotheta_to_d(INSTRUMENT_PARAMS["twotheta_max"], INSTRUMENT_PARAMS["wavelength"])
    d_max = twotheta_to_d(INSTRUMENT_PARAMS["twotheta_min"], INSTRUMENT_PARAMS["wavelength"])

    miller_set = symmetry.build_miller_set(anomalous_flag=False, d_min=d_min, d_max=d_max)
    reflections = miller_set.indices()
    uc = symmetry.unit_cell()
    twotheta = np.arange(INSTRUMENT_PARAMS["twotheta_min"], INSTRUMENT_PARAMS["twotheta_max"], INSTRUMENT_PARAMS["step_size"])
    I = np.zeros_like(twotheta)

    for hkl in reflections:
        d = uc.d(hkl)
        theta = 2 * np.arcsin(INSTRUMENT_PARAMS["wavelength"] / (2 * d)) * 180 / np.pi
        ps = np.random.uniform(5, 1000)
        peak = Peak(theta, INSTRUMENT_PARAMS["u"], INSTRUMENT_PARAMS["v"], INSTRUMENT_PARAMS["w"], ps, hkl=hkl)
        I += peak(twotheta)

    lorentz = 1 / (np.sin(np.radians(twotheta)) * np.sin(np.radians(twotheta / 2)))
    I *= lorentz
    I = 1000 * I / I.max()
    return twotheta, I


def save_patterns_to_hdf5(patterns_dict, filename):
    total_patterns = sum(len(p) for p in patterns_dict.values())
    if total_patterns == 0:
        print("⚠️ No patterns to save.")
        return

    # Collect and pad patterns
    all_intensity, all_ext_groups = [], []
    for ext, patterns in sorted(patterns_dict.items()):
        for intensity in patterns:
            all_intensity.append(np.array(intensity, dtype=np.float32))
            all_ext_groups.append(ext)

    max_len = max(len(x) for x in all_intensity)
    n = len(all_intensity)
    inten_pad = np.zeros((n, max_len), dtype=np.float32)
    for i, ii in enumerate(all_intensity):
        inten_pad[i, :len(ii)] = ii
    ext_array = np.array(all_ext_groups, dtype=np.int32)

    # Random split
    idx = np.random.permutation(n)
    train, val, test = np.split(idx, [int(0.8 * n), int(0.9 * n)])

    # Save required data only
    with h5py.File(filename, 'w') as f:
        f.attrs['total_patterns'] = total_patterns
        f.attrs['creation_date'] = np.bytes_(datetime.datetime.now().isoformat())

        ml_group = f.create_group('ml_ready')
        ml_group.create_dataset('intensity', data=inten_pad, compression="gzip", compression_opts=4)
        ml_group.create_dataset('ext_groups', data=ext_array, compression="gzip", compression_opts=4)

        split_group = f.create_group('data_split')
        split_group.create_dataset('train_indices', data=train, compression="gzip", compression_opts=4)
        split_group.create_dataset('validation_indices', data=val, compression="gzip", compression_opts=4)
        split_group.create_dataset('test_indices', data=test, compression="gzip", compression_opts=4)

    print(f"\n💾 Saved {total_patterns} ML-ready patterns to {filename}")
    print(f"    Max points per pattern: {max_len}")
    print(f"    Ready for use with `take_indicies()`")

def generate_for_ext(args):
    ext_group, num_patterns = args
    results = []
    for _ in range(num_patterns):
        values = mapping[ext_group]
        if isinstance(values, (list, np.ndarray)):
            sg_number = values[0] if len(values) == 1 else random.choice(values)
        else:
            sg_number = values  # in case it's a single value, not in a list/array
        try:
            a, b, c, alpha, beta, gamma = generate_unit_cell(sg_number)
            twotheta, intensity = generate_diffraction_pattern(sg_number, a, b, c, alpha, beta, gamma)
            results.append((ext_group, intensity))
        except Exception as e:
            print(f"[Ext {ext_group}] Generation failed: {e}")
    return results


def main():
    num_per_ext = NUM_SAMPLES // 99
    args_list = [(ext, num_per_ext) for ext in range(1, 100)]
    print(num_per_ext)
    with Pool(processes=NUM_PROCESSES) as pool:
        print(f"Processing {NUM_SAMPLES} samples with {pool._processes} processes")
        all_results = pool.map(generate_for_ext, args_list)

    pattern_dict = {}
    for group in all_results:
        for ext_number, intensity in group:
            pattern_dict.setdefault(ext_number, []).append(intensity)

    save_patterns_to_hdf5(pattern_dict, OUTPUT_HDF5)


if __name__ == "__main__":
    main()
