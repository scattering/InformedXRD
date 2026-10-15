#XRD_patterens_crystal_gen

# -- Step 1: Script Configuration --
import os
import numpy as np
import random
import h5py
from tqdm import tqdm
from pyxtal import pyxtal
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from multiprocessing import Pool, cpu_count, Queue, Process
from pyxtal.symmetry import get_wyckoffs, choose_wyckoff
import warnings
import re
import spglib

# Configuration
NUM_SAMPLES = 500000        # Total samples to generate
MAX_INTENSITY = 100      # Intensity normalization
THETA_MIN = 20           # Minimum theta (degrees)
THETA_MAX = 90           # Maximum theta (degrees)
NUM_POINTS = 500         # Number of points in the XRD pattern
OUTPUT_HDF5 = "dynamic_xrd_patterns_improved.hdf5"
NUM_PROCESSES = max(1, cpu_count() - 1)

# Suppress warnings
warnings.filterwarnings("ignore")

# Queue for inter-process communication
queue = Queue()

# Valid space groups and elements
VALID_SPACE_GROUPS = list(range(1, 231))
ELEMENTS = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
    'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
    'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
    'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
    'Pa', 'U'
]

# -- Step 2: Structure Generation Functions --
def sample_valid_composition(space_group):
    """ Ensures atom counts are compatible with available Wyckoff positions. """
    wyckoff_data = get_wyckoffs(space_group, organized=True)

    valid_counts = []
    for positions in wyckoff_data:
        total_atoms = sum(len(pos) for pos in positions)  # Get valid atom counts
        if total_atoms > 1:
            valid_counts.append(total_atoms)

    if not valid_counts:
        return ['C'], [2]  # Default fallback

    num_atoms = random.choice(valid_counts)
    num_elements = np.random.randint(2, 4)
    elements = random.sample(ELEMENTS, num_elements)

    valid_ratios = [r for r in range(2, num_atoms + 1) if num_atoms % r == 0]
    ratios = [random.choice(valid_ratios) for _ in range(num_elements)]

    return elements, ratios

def validate_structure(structure, expected_space_group):
    """ Checks if generated structure matches the requested space group. """
    atoms = structure.to_ase()
    lattice = atoms.get_cell()
    positions = atoms.get_scaled_positions()
    numbers = atoms.get_atomic_numbers()

    cell = (lattice, positions, numbers)
    detected_group = spglib.get_spacegroup(cell, symprec=1e-2)

    match = re.search(r'\((\d+)\)', detected_group)
    if match:
        detected_space_group = int(match.group(1))
        return expected_space_group == detected_space_group
    return False

def get_valid_structure(space_group, max_attempts=5000):
    """ Generates a valid structure with correct Wyckoff positions and compositions. """
    attempts = 0
    while attempts < max_attempts:
        try:
            elements, ratios = sample_valid_composition(space_group)
            structure = pyxtal()
            structure.from_random(dim=3, group=space_group, species=elements, numIons=ratios)

            if validate_structure(structure, space_group):
                return structure
        except Exception as e:
            error_message = str(e)

        attempts += 1

    print(f"Failed to generate valid structure for SG {space_group} after {max_attempts} attempts: {error_message}")
    return None

# -- Step 3: XRD Pattern Generation --
def generate_xrd_pattern(structure):
    """Generate XRD pattern from a structure."""
    try:
        calc = XRDCalculator(wavelength="CuKa")
        pattern = calc.get_pattern(structure.to_pymatgen())

        fixed_two_theta = np.linspace(THETA_MIN, THETA_MAX, NUM_POINTS)
        fixed_intensity = np.interp(fixed_two_theta, pattern.x, pattern.y)

        fixed_intensity /= fixed_intensity.max() / MAX_INTENSITY
        return fixed_two_theta, fixed_intensity
    except Exception as e:
        print(f"Error generating XRD pattern: {e}")
        return None

# -- Step 4: HDF5 Saving Function --
def save_to_hdf5():
    """Save XRD patterns to HDF5 file."""
    with h5py.File(OUTPUT_HDF5, "a") as f:
        while True:
            item = queue.get()
            if item == "DONE":
                break
            sg, two_theta, intensity = item
            group_name = f"SG_{sg}"
            if group_name not in f:
                group = f.create_group(group_name)
                group.create_dataset("2theta", data=two_theta)
                group.create_dataset("intensity", data=[intensity], maxshape=(None, NUM_POINTS), chunks=True)
            else:
                group = f[group_name]
                group["intensity"].resize((group["intensity"].shape[0] + 1), axis=0)
                group["intensity"][-1, :] = intensity

# -- Step 5: Parallel XRD Generation Task --
def process_task(space_group):
    """Generate an XRD pattern for a given space group."""
    structure = get_valid_structure(space_group)
    if structure:
        result = generate_xrd_pattern(structure)
        if result:
            two_theta, intensity = result
            queue.put((space_group, two_theta, intensity))

# -- Step 6: Main Execution --
def main():
    print("\nStarting Dynamic XRD Pattern Generation...\n")
    writer_process = Process(target=save_to_hdf5)
    writer_process.start()

    space_groups = random.choices(VALID_SPACE_GROUPS, k=NUM_SAMPLES)

    with tqdm(total=NUM_SAMPLES, desc="Generating XRD Patterns", dynamic_ncols=True) as pbar:
        with Pool(processes=NUM_PROCESSES) as pool:
            for sg in space_groups:
                pool.apply_async(process_task, args=(sg,))
                pbar.update(1)
            pool.close()
            pool.join()

    queue.put("DONE")
    writer_process.join()
    print("\nXRD pattern generation complete!")

if __name__ == "__main__":
    main()
