#crysfml_pattern_gen

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
from pycrysfml import cfml_py_utilities
import json
import traceback
from ase import atoms

os.environ["FULLPROF"]="/storage/ejb12/FullProf"  # Install Fullprof in directory

# Configuration
NUM_SAMPLES = 123000       # Total samples to generate
MAX_INTENSITY = 100      # Intensity normalization

INSTRUMENT_PARAMS = {
    "wavelength": 1.54030,
    "twotheta_min": 3,           # Minimum theta (degrees)
    "twotheta_max": 155,           # Maximum theta (degrees)
    "step_size": 0.05,           # Step size between 2theta points (degrees)
    "u": 0.025,           # Resolution parameters
    "v": -0.025,
    "w": 0.025
}
NUM_POINTS = (INSTRUMENT_PARAMS["twotheta_max"]-INSTRUMENT_PARAMS["twotheta_min"])/INSTRUMENT_PARAMS["step_size"] + 1

OUTPUT_HDF5 = "123k_pyCrysfml.hdf5"
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

def generate_p1_structure():
    """ Generates a random P1 (SG 1) structure with random elements
and atom count. """
    structure = pyxtal()

    num_elements = random.randint(1, 3)  # Choose 1-3 different elements
    selected_elements = random.sample(ELEMENTS, num_elements)  # Pick random elements

    num_atoms = random.randint(2, 10)  # Random atom count (ensuring diverse structures)
    atom_distribution = [num_atoms // num_elements] * num_elements  # Distribute evenly

    # Ensure total atoms match
    remainder = num_atoms % num_elements
    for i in range(remainder):
        atom_distribution[i] += 1

    # Generate the random P1 structure
    structure.from_random(dim=3, group=1, species=selected_elements,
numIons=atom_distribution)
    return structure

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
    error_message = "Unknown error"
    while attempts < max_attempts:
        try:
            if space_group == 1:
                structure = generate_p1_structure()
            else:
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
def extract_structural_info(structure):
    """Extracts unit cell from generated crystal structure"""
    atoms: Atoms = structure.to_ase()
    cell_lengths = atoms.get_cell_lengths_and_angles()

    a, b, c, alpha, beta, gamma = cell_lengths

    space_group = structure.group.number # Get space group number
    positions = atoms.get_scaled_positions()  # Get fractional coordinates
    symbols = atoms.get_chemical_symbols() # Get elements
    atom_list = [
        {"symbol": sym, "position": pos}
        for sym, pos in zip(symbols, positions)
    ]
    return space_group, a, b, c, alpha, beta, gamma, atom_list

def create_pattern_json(space_group, a, b, c, alpha, beta, gamma, atom_list):
    """ Creates JSON input for pyCrysfml generation"""
    sg = spglib.get_spacegroup_type(space_group)['international_short']
    data = {
             "phases": [
                    {
                        "placeholder": {
                            "_space_group_name_H-M_alt": sg,
                            "_cell_length_a": float(a),
                            "_cell_length_b": float(b),
                            "_cell_length_c": float(c),
                            "_cell_angle_alpha": float(alpha),
                            "_cell_angle_beta": float(beta),
                            "_cell_angle_gamma": float(gamma),
                            
                            
                            "_atom_site": [
                                {
                                    "_label": atom["symbol"],
                                    "_type_symbol": atom["symbol"],
                                    "_fract_x": float(atom["position"][0]),
                                    "_fract_y": float(atom["position"][1]),
                                    "_fract_z": float(atom["position"][2]),
                                    "_occupancy": 1.0,
                                    "_adp_type": "iso",
                                    "_B_iso_or_equiv": 0
                                }
                                for atom in atom_list
                            ]
                        }
                    }
            ],
            "experiments": [
                {
                "NPD": {
                    #"_diffrn_source": "nuclear reactor", uncomment for neutron source
                    #_diffrn_radiation_probe": "neutron" uncomment for neutron
                    "_diffrn_radiation_probe": "xray", # comment for neutron
                    "_diffrn_radiation_wavelength": INSTRUMENT_PARAMS["wavelength"],
                    "_pd_instr_resolution_u": INSTRUMENT_PARAMS["u"],
                    "_pd_instr_resolution_v": INSTRUMENT_PARAMS["v"],
                    "_pd_instr_resolution_w": INSTRUMENT_PARAMS["w"],
                    "_pd_instr_resolution_x": 0.0015,
                    "_pd_instr_resolution_y": 0,
                    "_pd_instr_reflex_asymmetry_p1": 0,
                    "_pd_instr_reflex_asymmetry_p2": 0,
                    "_pd_instr_reflex_asymmetry_p3": 0,
                    "_pd_instr_reflex_asymmetry_p4": 0,
                    "_pd_meas_2theta_offset": 0,
                    "_pd_meas_2theta_range_min": INSTRUMENT_PARAMS["twotheta_min"],
                    "_pd_meas_2theta_range_max": INSTRUMENT_PARAMS["twotheta_max"],
                    "_pd_meas_2theta_range_inc": INSTRUMENT_PARAMS["step_size"],
                    "_phase": [
                    {
                        "_label": "placeholder",
                        "_scale": 1
                    }
                    ],
                    "_pd_background": [
                    {
                        "_2theta": INSTRUMENT_PARAMS["twotheta_min"],
                        "_intensity": 0
                    },
                    {
                        "_2theta": INSTRUMENT_PARAMS["twotheta_max"],
                        "_intensity": 0
                    }
                    ]
                }
                }
            ]
            }
    return data

def generate_diff_pattern(structure):
    """ Uses pyCrysfml to generate diffraction pattern"""
    space_group, a, b, c, alpha, beta, gamma, atoms = extract_structural_info(structure)
    pattern_json = create_pattern_json(space_group, a, b, c, alpha, beta, gamma, atoms)
    try:
        x,y = cfml_py_utilities.cw_powder_pattern_from_dict(pattern_json)
        return x,y
    except Exception as e:
        traceback.print_exc()
        print(f"Error: {e}")
        return None

# -- Step 4: HDF5 Saving Function --
def get_xtal_system(space_group):
    if space_group in range(1,3): # Groups 1-3 are triclinic
        return "triclinic"
    if space_group in range(3,16): # Groups 3-15 are monoclinic
        return "monoclinic"
    if space_group in range(16, 75): # Groups 16-74 are orthorhombic
        return "orthorhombic"
    if space_group in range(75, 143): # Groups 75-142 are tetragonal
        return "tetragonal"
    if space_group in range(143, 168): # Groups 143-167 are trigonal
        return "trigonal"
    if space_group in range(168, 195): # Groups 168-194 are hexagonal
        return "hexagonal"
    if space_group in range(195, 231):# Groups 195-230 are cubic
        return "cubic"
    else:
        return "unknown"
        
def save_to_hdf5():
    """Save diffraction patterns to HDF5 file."""
    two_theta_written = False  # Flag to ensure 2theta is written only once
    with h5py.File(OUTPUT_HDF5, "a") as f:
        while True:
            item = queue.get()
            if item == "DONE":
                break
            sg, two_theta, intensity = item
            group_name = f"SG_{sg}"

            # Check if group exists
            if group_name not in f:
                try:
                    group = f.create_group(group_name)
                except ValueError:
                    # If it was just created by another thread in-between, fallback
                    group = f[group_name]
            else:
                group = f[group_name]
                
            # Save crystal system as attribute
            if "crystal_system" not in group.attrs:
                group.attrs["crystal_system"] = get_xtal_system(sg)
                
            # Write 2theta only once at the root level
            if not two_theta_written:
                if "2theta" not in f:
                    f.create_dataset("2theta", data=two_theta)
                two_theta_written = True
                
            # Add intensity
            if "intensity" not in group:
                group.create_dataset("intensity", data=[intensity], maxshape=(None, NUM_POINTS), chunks=True)
            else:
                group["intensity"].resize((group["intensity"].shape[0] + 1), axis=0)
                group["intensity"][-1, :] = intensity

# -- Step 5: Parallel XRD Generation Task --
def process_task(space_group):
    """Generate an XRD pattern for a given space group."""
    structure = get_valid_structure(space_group)
    if structure:
        result = generate_diff_pattern(structure)
        extract_structural_info(structure)
        if result:
            two_theta, intensity = result
            queue.put((space_group, two_theta, intensity))
    
# -- Step 6: Main Execution --
def main():
    print("\nStarting Dynamic Pattern Generation...\n")
    writer_process = Process(target=save_to_hdf5)
    writer_process.start()

    space_groups = random.choices(VALID_SPACE_GROUPS, k=NUM_SAMPLES)

    completed = 0

    def on_success(_):
        nonlocal completed
        completed += 1
        pbar.update(1)

    def on_error(e):
        print(f"[ERROR] Task failed: {e}")

    with tqdm(total=NUM_SAMPLES, desc="Generating XRD Patterns", dynamic_ncols=True) as pbar:
        with Pool(processes=NUM_PROCESSES) as pool:
            for sg in space_groups:
                pool.apply_async(process_task, args=(sg,), callback=on_success, error_callback=on_error)
            pool.close()
            pool.join()

    queue.put("DONE")
    writer_process.join()
    print("\nXRD pattern generation complete!")

    
if __name__ == "__main__":
    main()
