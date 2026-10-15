# Import statements
import pyxtal
from pyxtal import pyxtal
import numpy as np
import random
import re
import json
#import pycrysfml
#from pycrysfml import cfml_utilities
import spglib

# Elements 1 - 92
all_elements = [
    'H', 'He', 
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn','Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn','Sb', 'Te', 'I', 'Xe', 
    'Cs', 'Ba', 
    'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb','Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg','Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
    'Fr', 'Ra',
    'Ac', 'Th', 'Pa', 'U'
]

# Make a guess for structure
def structure_guess():
    num = np.random.randint(1,4)
    elements = random.sample(all_elements, num)
    ratio = np.random.randint(0, 8, size=num)
    #space_group = np.random.randint(1,230) # Uncomment for random space group
    return elements, ratio

# Tries until a guess works. Returns structure in ase format.
def new_structure(space_group):
    structure = pyxtal()

    while True:
        try:
            # Generate random structure guess
            elements, ratio = structure_guess()
            structure.from_random(3, space_group, elements, ratio)

            # Validate space group
            if check_space_group(structure, space_group):
                return structure # Success exit             

        except Exception as e:
            #print(f"Attempt {attempts} failed: {e}") # Uncomment for error statments
            continue  # Retry on failure

# Check space group
def check_space_group(structure, spacegroup1):
    atoms = structure.to_ase()
    lattice = atoms.get_cell()
    positions = atoms.get_scaled_positions()  # Fractional coordinates
    numbers = atoms.get_atomic_numbers()
    cell = (lattice, positions, numbers)
    spacegroup2 = spglib.get_spacegroup(cell, symprec=1e-5)
    spacegroup2 = int(re.search(r'\((\d+)\)', spacegroup2).group(1))
    print(spacegroup1,spacegroup2)
    return spacegroup1 == spacegroup2

# Create a class for easy access to properties
class RandomStructure():
    def __init__(self, sg):
        self.structure = new_structure(sg)
        self._atoms = None
        self._cell_params = None
        self._space_group = None
        self._name = None
        self._json = None
        
    @property
    def cell_params(self):
        if self._cell_params is None:
            self._cell_params = self.atoms.cell.cellpar() 
        return self._cell_params
    
    def get_cell_params(self):
        return self.atoms.cell.cellpar()        
    
    @property
    def space_group(self):
        if self._space_group is None:
            self._space_group = self.get_space_group()
        return self._space_group
    
    def get_space_group(self):
        spacegroup_string = str(self.structure.group)
        pattern = r"\(([^)]+)\)"
        match = re.search(pattern, spacegroup_string)
        if match:
            return match.group(1)
        else:
            raise ValueError("Space group number not found in the provided string.")
    
    @property
    def atoms(self):
        if self._atoms is None:
            self._atoms = self.get_atoms()
        return self._atoms

    def get_atoms(self):
        return self.structure.to_ase()
    
    
    @property
    def name(self):
        if self._name is None:
            self._name = self.get_name()
        return self._name
    
    def get_name(self):
        return self.structure.formula
    
    @property
    def json(self):
        if self._json is None:
            self._json = self.get_json()
        return self._json
    
    def get_json(self):
        a, b, c, alpha, beta, gamma = self.cell_params
        data = {
            "phases": [
                {
                    self.name: {
                        "_space_group_name_H-M_alt": self.space_group,
                        "_cell_length_a": a,
                        "_cell_length_b": b,
                        "_cell_length_c": c,
                        "_cell_angle_alpha": alpha,
                        "_cell_angle_beta": beta,
                        "_cell_angle_gamma": gamma,
                        "_atom_site": [
                            {
                                "_label": atom.symbol,
                                "_type_symbol": atom.symbol,
                                "_fract_x": pos[0],
                                "_fract_y": pos[1],
                                "_fract_z": pos[2],
                                "_occupancy": 1.0,
                                "_adp_type": "iso",
                                "_B_iso_or_equiv": 0
                            }
                            for atom, pos in zip(self.atoms, self.atoms.get_scaled_positions())
                        ]
                    }
                }
          ],
          "experiments": [
            {
              "NPD": {
                "_diffrn_source": "nuclear reactor",
                "_diffrn_radiation_probe": "neutron",
                "_diffrn_radiation_wavelength": 1.54030,
                "_pd_instr_resolution_u": 0.347,
                "_pd_instr_resolution_v": -0.278,
                "_pd_instr_resolution_w": 0.166,
                "_pd_instr_resolution_x": 0.0015,
                "_pd_instr_resolution_y": 0,
                "_pd_instr_reflex_asymmetry_p1": 0,
                "_pd_instr_reflex_asymmetry_p2": 0,
                "_pd_instr_reflex_asymmetry_p3": 0,
                "_pd_instr_reflex_asymmetry_p4": 0,
                "_pd_meas_2theta_offset": 0,
                "_pd_meas_2theta_range_min": 3,
                "_pd_meas_2theta_range_max": 155,
                "_pd_meas_2theta_range_inc": 0.05,
                "_phase": [
                  {
                    "_label": self.name,
                    "_scale": 1
                  }
                ],
                "_pd_background": [
                  {
                    "_2theta": 3,
                    "_intensity": 0
                  },
                  {
                    "_2theta": 155,
                    "_intensity": 0
                  }
                ]
              }
            }
          ]
        }
        return json.dumps(data)


if __name__ == "__main__":
    print(new_structure(196))
