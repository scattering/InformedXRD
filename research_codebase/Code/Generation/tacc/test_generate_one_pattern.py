import random

import numpy as np
import spglib
from pycrysfml import cfml_py_utilities
from pyxtal import pyxtal
from pyxtal.symmetry import get_wyckoffs


ELEMENTS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si",
    "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni",
    "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb",
    "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho",
    "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",
]


def sample_valid_composition(space_group):
    wyckoff_data = get_wyckoffs(space_group, organized=True)
    valid_counts = []
    for positions in wyckoff_data:
        total_atoms = sum(len(pos) for pos in positions)
        if total_atoms > 1:
            valid_counts.append(total_atoms)

    if not valid_counts:
        return ["C"], [2]

    num_atoms = random.choice(valid_counts)
    num_elements = np.random.randint(2, 4)
    elements = random.sample(ELEMENTS, num_elements)
    valid_ratios = [r for r in range(2, num_atoms + 1) if num_atoms % r == 0]
    ratios = [random.choice(valid_ratios) for _ in range(num_elements)]
    return elements, ratios


def validate_structure(structure, expected_space_group):
    atoms = structure.to_ase()
    lattice = atoms.get_cell()
    positions = atoms.get_scaled_positions()
    numbers = atoms.get_atomic_numbers()
    detected_group = spglib.get_spacegroup((lattice, positions, numbers), symprec=1e-2)
    return detected_group is not None and f"({expected_space_group})" in detected_group


def get_valid_structure(space_group, max_attempts=100):
    for _ in range(max_attempts):
        try:
            structure = pyxtal()
            elements, ratios = sample_valid_composition(space_group)
            structure.from_random(dim=3, group=space_group, species=elements, numIons=ratios)
            if validate_structure(structure, space_group):
                return structure
        except Exception:
            continue
    raise RuntimeError(f"Failed to generate valid structure for SG {space_group}")


def extract_structural_info(structure):
    atoms = structure.to_ase()
    a, b, c, alpha, beta, gamma = atoms.get_cell_lengths_and_angles()
    positions = atoms.get_scaled_positions()
    symbols = atoms.get_chemical_symbols()
    atom_list = [{"symbol": sym, "position": pos} for sym, pos in zip(symbols, positions)]
    return structure.group.number, a, b, c, alpha, beta, gamma, atom_list


def create_pattern_json(space_group, a, b, c, alpha, beta, gamma, atom_list):
    return {
        "phases": [
            {
                "placeholder": {
                    "_space_group_name_H-M_alt": spglib.get_spacegroup_type(space_group)["international_short"],
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
                            "_B_iso_or_equiv": 0,
                        }
                        for atom in atom_list
                    ],
                }
            }
        ],
        "experiments": [
            {
                "NPD": {
                    "_diffrn_radiation_probe": "xray",
                    "_diffrn_radiation_wavelength": 1.54030,
                    "_pd_instr_resolution_u": 0.025,
                    "_pd_instr_resolution_v": -0.025,
                    "_pd_instr_resolution_w": 0.025,
                    "_pd_instr_resolution_x": 0.0015,
                    "_pd_instr_resolution_y": 0,
                    "_pd_instr_reflex_asymmetry_p1": 0,
                    "_pd_instr_reflex_asymmetry_p2": 0,
                    "_pd_instr_reflex_asymmetry_p3": 0,
                    "_pd_instr_reflex_asymmetry_p4": 0,
                    "_pd_meas_2theta_offset": 0,
                    "_pd_meas_2theta_range_min": 10.0,
                    "_pd_meas_2theta_range_max": 110.0,
                    "_pd_meas_2theta_range_inc": 0.1,
                    "_phase": [{"_label": "placeholder", "_scale": 1}],
                    "_pd_background": [
                        {"_2theta": 10.0, "_intensity": 0},
                        {"_2theta": 110.0, "_intensity": 0},
                    ],
                }
            }
        ],
    }


def main():
    random.seed(123)
    np.random.seed(123)

    sg = 225
    structure = get_valid_structure(sg)
    sg, a, b, c, alpha, beta, gamma, atom_list = extract_structural_info(structure)
    pattern_json = create_pattern_json(sg, a, b, c, alpha, beta, gamma, atom_list)
    two_theta, intensity = cfml_py_utilities.cw_powder_pattern_from_dict(pattern_json)

    intensity = np.asarray(intensity, dtype=np.float32)
    two_theta = np.asarray(two_theta, dtype=np.float32)

    print("space_group", sg)
    print("num_atoms", len(atom_list))
    print("num_points", intensity.shape[0])
    print("nan_count", int(np.isnan(intensity).sum()))
    print("max_intensity", float(np.nanmax(intensity)))
    print("first_points", two_theta[:5].tolist())
    print("first_intensity", intensity[:5].tolist())


if __name__ == "__main__":
    main()
