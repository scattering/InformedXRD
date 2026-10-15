import numpy as np
import random
import re
import json
import pyxtal
import spglib
from pyxtal import pyxtal
from pyxtal.symmetry import get_wyckoffs, choose_wyckoff

VALID_SPACE_GROUPS = list(range(2, 231))  # Space Groups 1-230
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
] # Try different elements

def get_valid_structure(space_group, max_attempts=1000):
    """ Generates a valid structure with correct Wyckoff positions and compositions. """
    attempts = 0
    while attempts < max_attempts:
        try:
            elements, ratios = sample_valid_composition(space_group)
            structure = pyxtal()
            structure.from_random(dim=3, group=space_group, species=elements, numIons=ratios)

            if validate_structure(structure, space_group):
                return structure  # Success!
        except Exception as e:
            pass  # Ignore failures and retry

        attempts += 1

    raise ValueError(f"❌ Failed to generate valid structure for SG {space_group} after {max_attempts} attempts.")

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

    num_atoms = random.choice(valid_counts)  # Pick a valid count

    # Select elements (ensure at least 2)
    num_elements = np.random.randint(2, 4)  # Use 2-4 elements for stability
    elements = random.sample(ELEMENTS, num_elements)

    # Ensure `numIons` sums to a valid Wyckoff-compatible number
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

def generate_all_structures():
    """ Generate valid structures for all space groups while ensuring uniform distribution. """
    valid_structures = []
    failed_sgs = {}

    for sg in VALID_SPACE_GROUPS:
        try:
            structure = get_valid_structure(sg)
            valid_structures.append((sg, structure))
            print(f"✅ Successfully generated structure for Space Group {sg}")
        except ValueError as e:
            failed_sgs[sg] = str(e)
            print(f"🚨 {e}")

    # Print summary
    print("\n🟢 **Final Summary**")
    print(f"Total valid structures: {len(valid_structures)}")
    print(f"Total failed space groups: {len(failed_sgs)}")
    print(f"Failed space groups: {list(failed_sgs.keys())}")

    return valid_structures, failed_sgs

##Generating for group 1
def generate_p1_structure():
    """ Generates a random P1 (SG 1) structure with random elements and atom count. """
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
    structure.from_random(dim=3, group=1, species=selected_elements, numIons=atom_distribution)
    return structure

def save_structures_to_file(structures_dict, filename):
    """
    Save the generated structures to a JSON file.
    
    Parameters:
    -----------
    structures_dict : dict
        Dictionary with space groups as keys and lists of structures as values
    filename : str
        Output filename
    """
    output_data = {}
    
    for sg, structures in structures_dict.items():
        if not structures:
            continue
            
        output_data[str(sg)] = []
        
        for structure in structures:
            # Convert structure to dictionary representation
            structure_data = {
                "space_group": sg,
                "formula": structure.formula,
                "lattice": structure.lattice.tolist(),
                "elements": structure.species,
                "wyckoff_sites": []
            }
            
            # Add Wyckoff positions
            for site in structure.atom_sites:
                wyckoff_data = {
                    "element": site[0],
                    "position": site[1].tolist(),
                    "wyckoff_letter": site[2],
                    "multiplicity": site[3]
                }
                structure_data["wyckoff_sites"].append(wyckoff_data)
                
            output_data[str(sg)].append(structure_data)
    
    # Save to file
    with open(filename, 'w') as f:
        json.dump(output_data, f, indent=2)

def is_duplicate_structure(structure, existing_structures, tolerance=0.01):
    """
    Check if a structure is a duplicate of any existing structure.
    
    Parameters:
    -----------
    structure : pyxtal structure
        The structure to check
    existing_structures : list
        List of existing pyxtal structures to compare against
    tolerance : float, optional
        Tolerance for lattice parameter comparison (default: 0.01)
        
    Returns:
    --------
    bool
        True if the structure is a duplicate, False otherwise
    """
    if not existing_structures:
        return False
        
    # Convert to ASE atoms object for comparison
    atoms1 = structure.to_ase()
    
    for existing in existing_structures:
        # Skip if formulas are different
        if structure.formula != existing.formula:
            continue
            
        # Convert existing structure to ASE atoms
        atoms2 = existing.to_ase()
        
        # Compare lattice parameters
        cell1 = atoms1.get_cell()
        cell2 = atoms2.get_cell()
        
        # Check if lattice parameters are similar
        if np.allclose(cell1, cell2, atol=tolerance):
            # Compare atomic positions
            pos1 = atoms1.get_scaled_positions()
            pos2 = atoms2.get_scaled_positions()
            
            # Sort positions by element type for comparison
            elements1 = atoms1.get_chemical_symbols()
            elements2 = atoms2.get_chemical_symbols()
            
            # Create sorted position arrays based on element types
            sorted_pos1 = [pos for _, pos in sorted(zip(elements1, pos1))]
            sorted_pos2 = [pos for _, pos in sorted(zip(elements2, pos2))]
            
            # Check if positions are similar (accounting for periodic boundary conditions)
            if are_positions_similar(sorted_pos1, sorted_pos2, tolerance):
                return True
                
    return False

def are_positions_similar(pos1, pos2, tolerance):
    """
    Check if two sets of positions are similar, accounting for periodic boundary conditions.
    
    Parameters:
    -----------
    pos1, pos2 : list of arrays
        Lists of position arrays to compare
    tolerance : float
        Tolerance for position comparison
        
    Returns:
    --------
    bool
        True if positions are similar, False otherwise
    """
    if len(pos1) != len(pos2):
        return False
        
    for p1, p2 in zip(pos1, pos2):
        # Account for periodic boundary conditions
        diff = np.abs(p1 - p2)
        diff = np.minimum(diff, 1.0 - diff)  # Account for periodicity
        
        if np.any(diff > tolerance):
            return False
            
    return True

def generate_n_samples_per_space_group(n, save_to_file=True, output_file="crystal_structures.json", check_duplicates=True):
    """
    Generate n samples for each space group from 1 to 230.
    
    Parameters:
    -----------
    n : int
        Number of samples to generate for each space group
    save_to_file : bool, optional
        Whether to save the generated structures to a file (default: True)
    output_file : str, optional
        Filename to save the structures (default: "crystal_structures.json")
    check_duplicates : bool, optional
        Whether to check for and avoid duplicate structures (default: True)
        
    Returns:
    --------
    dict
        Dictionary with space groups as keys and lists of structures as values
    """
    all_structures = {}
    all_failed = {}
    duplicate_counts = {}
    
    # Process all space groups from 1 to 230
    for sg in range(1, 231):
        print(f"\n🔍 Generating {n} samples for Space Group {sg}")
        structures = []
        failed_attempts = 0
        duplicates_found = 0
        
        # Special case for space group 1 (P1)
        if sg == 1:
            i = 0
            attempts = 0
            max_attempts = n * 10  # Limit total attempts to avoid infinite loops
            
            while i < n and attempts < max_attempts:
                attempts += 1
                try:
                    structure = generate_p1_structure()
                    if structure and structure.valid:
                        # Check for duplicates if enabled
                        if check_duplicates and is_duplicate_structure(structure, structures):
                            duplicates_found += 1
                            print(f"  🔄 Duplicate found for SG 1 (P1), regenerating...")
                            continue
                            
                        structures.append(structure)
                        i += 1
                        print(f"  ✅ Sample {i}/{n} for SG 1 (P1) generated successfully")
                    else:
                        failed_attempts += 1
                        print(f"  ❌ Generation attempt for SG 1 (P1) failed")
                except Exception as e:
                    failed_attempts += 1
                    print(f"  ❌ Generation attempt for SG 1 (P1) failed: {str(e)}")
        else:
            # For space groups 2-230
            i = 0
            attempts = 0
            max_attempts = n * 10  # Limit total attempts to avoid infinite loops
            
            while i < n and attempts < max_attempts:
                attempts += 1
                try:
                    structure = get_valid_structure(sg)
                    
                    # Check for duplicates if enabled
                    if check_duplicates and is_duplicate_structure(structure, structures):
                        duplicates_found += 1
                        print(f"  🔄 Duplicate found for SG {sg}, regenerating...")
                        continue
                        
                    structures.append(structure)
                    i += 1
                    print(f"  ✅ Sample {i}/{n} for SG {sg} generated successfully")
                except ValueError as e:
                    failed_attempts += 1
                    print(f"  ❌ Generation attempt for SG {sg} failed: {str(e)}")
                except Exception as e:
                    failed_attempts += 1
                    print(f"  ❌ Generation attempt for SG {sg} failed with unexpected error: {str(e)}")
        
        # Store results
        all_structures[sg] = structures
        if failed_attempts > 0:
            all_failed[sg] = failed_attempts
        if duplicates_found > 0:
            duplicate_counts[sg] = duplicates_found
    
    # Print summary
    print("\n🟢 **Final Generation Summary**")
    total_structures = sum(len(structures) for structures in all_structures.values())
    total_failed = sum(all_failed.values()) if all_failed else 0
    total_duplicates = sum(duplicate_counts.values()) if duplicate_counts else 0
    print(f"Total structures generated: {total_structures} out of {230 * n} attempted")
    print(f"Total failed generations: {total_failed}")
    print(f"Total duplicates detected and regenerated: {total_duplicates}")
    
    # Calculate success rate per space group
    print("\n📊 Success Rate by Space Group:")
    for sg, structures in all_structures.items():
        success_rate = (len(structures) / n) * 100
        print(f"SG {sg}: {success_rate:.1f}% ({len(structures)}/{n})")
        
    # Report on duplicates
    if duplicate_counts:
        print("\n🔍 Duplicate Detection Summary:")
        for sg, count in duplicate_counts.items():
            print(f"SG {sg}: {count} duplicates detected and regenerated")
    
    # Save to file if requested
    if save_to_file and total_structures > 0:
        save_structures_to_file(all_structures, output_file)
        print(f"\n💾 Saved {total_structures} structures to {output_file}")
    
    return all_structures

# Example usage:
# To generate 5 samples for each space group:
# structures = generate_n_samples_per_space_group(5)

# Run the original code for backward compatibility
if __name__ == "__main__":
    # Ask user for number of samples
    try:
        n_samples = int(input("Enter number of samples to generate per space group (default: 1): ") or "1")
        if n_samples <= 0:
            print("Number of samples must be positive, using default value of 1")
            n_samples = 1
    except ValueError:
        print("Invalid input, using default value of 1")
        n_samples = 1
    
    # Generate the samples
    structures = generate_n_samples_per_space_group(n_samples, output_file="crystal_structures.json", check_duplicates=True)

def save_structures_for_transformer(structures_dict, filename):
    """
    Save structures in a format optimized for transformer model training.
    """
    samples = []
    
    for sg, structures in structures_dict.items():
        for structure in structures:
            # Get lattice parameters in conventional form
            atoms = structure.to_ase()
            cell = atoms.get_cell()
            a, b, c, alpha, beta, gamma = cell.cellpar()
            
            # Create sample dictionary
            sample = {
                "space_group": sg,
                "space_group_symbol": structure.group.symbol,
                "lattice_parameters": {
                    "a": float(a),
                    "b": float(b),
                    "c": float(c),
                    "alpha": float(alpha),
                    "beta": float(beta),
                    "gamma": float(gamma)
                },
                "atoms": [],
                "formula": structure.formula,
                "num_atoms": len(atoms)
            }
            
            # Add atoms with positions and Wyckoff sites
            for site in structure.atom_sites:
                atom = {
                    "element": site[0],
                    "position": site[1].tolist(),
                    "wyckoff": f"{site[3]}{site[2]}"  # multiplicity + letter
                }
                sample["atoms"].append(atom)
            
            samples.append(sample)
    
    # Save to file
    with open(filename, 'w') as f:
        json.dump({"samples": samples}, f, indent=2)
    
    return len(samples)

def save_structures_to_hdf5(structures_dict, filename="crystal_structures.h5"):
    """
    Save crystal structures to HDF5 format optimized for ML model input.
    
    Parameters:
    -----------
    structures_dict : dict
        Dictionary with space groups as keys and lists of structures as values
    filename : str, optional
        Output HDF5 filename (default: "crystal_structures.h5")
        
    Returns:
    --------
    int
        Number of structures saved

    
        Example usage:
        import h5py
    import numpy as np

    # Load data
    with h5py.File('crystal_structures.h5', 'r') as f:
        # Get train indices
        train_idx = f['data_split/train_indices'][:]
        
        # Load ML-ready data
        lattice_params = f['ml_ready/lattice_parameters'][train_idx]
        atom_elements = f['ml_ready/atom_elements'][train_idx]
        atom_positions = f['ml_ready/atom_positions'][train_idx]
        atom_masks = f['ml_ready/atom_masks'][train_idx]
        space_groups = f['ml_ready/space_groups'][train_idx]
        
        # Create batches and train model
        # ...
    """
    try:
        import h5py
        import numpy as np
    except ImportError:
        print("Error: h5py package is required for HDF5 output. Install with 'pip install h5py'")
        return 0
    
    # Count total structures for pre-allocation
    total_structures = sum(len(structures) for structures in structures_dict.values())
    
    # Find maximum number of atoms across all structures for padding
    max_atoms = 0
    for sg, structures in structures_dict.items():
        for structure in structures:
            atoms = structure.to_ase()
            max_atoms = max(max_atoms, len(atoms))
    
    # Create HDF5 file
    with h5py.File(filename, 'w') as f:
        # Store metadata
        f.attrs['total_structures'] = total_structures
        f.attrs['max_atoms'] = max_atoms
        f.attrs['creation_date'] = np.string_(datetime.datetime.now().isoformat())
        
        # Create dataset groups
        structures_group = f.create_group('structures')
        
        # Create element mapping for consistent encoding
        elements_list = list(set(element for sg, structures in structures_dict.items() 
                                for structure in structures 
                                for element in structure.species))
        element_to_idx = {element: i for i, element in enumerate(sorted(elements_list))}
        
        # Store element mapping
        element_group = f.create_group('metadata/elements')
        for element, idx in element_to_idx.items():
            element_group.attrs[element] = idx
        
        # Process all structures
        structure_idx = 0
        for sg, structures in structures_dict.items():
            # Create space group group
            sg_group = structures_group.create_group(f'sg_{sg}')
            
            for i, structure in enumerate(structures):
                # Convert to ASE atoms for consistent processing
                atoms = structure.to_ase()
                cell = atoms.get_cell()
                positions = atoms.get_scaled_positions()
                elements = atoms.get_chemical_symbols()
                
                # Create structure group
                struct_group = sg_group.create_group(f'structure_{i}')
                
                # Store basic information
                struct_group.attrs['space_group'] = sg
                struct_group.attrs['formula'] = structure.formula
                struct_group.attrs['num_atoms'] = len(atoms)
                struct_group.attrs['space_group_symbol'] = structure.group.symbol
                
                # Store lattice parameters
                a, b, c, alpha, beta, gamma = cell.cellpar()
                lattice_params = np.array([a, b, c, alpha, beta, gamma], dtype=np.float32)
                struct_group.create_dataset('lattice_parameters', data=lattice_params)
                
                # Store full lattice matrix
                struct_group.create_dataset('lattice_matrix', data=cell.array.astype(np.float32))
                
                # Store atom data with padding
                atom_elements = np.zeros(max_atoms, dtype=np.int32)
                atom_positions = np.zeros((max_atoms, 3), dtype=np.float32)
                atom_mask = np.zeros(max_atoms, dtype=np.int8)  # Mask for valid atoms
                
                # Fill atom data
                for j, (element, position) in enumerate(zip(elements, positions)):
                    atom_elements[j] = element_to_idx[element]
                    atom_positions[j] = position
                    atom_mask[j] = 1
                
                # Store atom datasets
                struct_group.create_dataset('atom_elements', data=atom_elements)
                struct_group.create_dataset('atom_positions', data=atom_positions)
                struct_group.create_dataset('atom_mask', data=atom_mask)
                
                # Store Wyckoff positions
                wyckoff_data = []
                for site in structure.atom_sites:
                    wyckoff_data.append({
                        'element': site[0],
                        'position': site[1].tolist(),
                        'letter': site[2],
                        'multiplicity': site[3]
                    })
                
                # Store Wyckoff data as JSON string
                wyckoff_json = json.dumps(wyckoff_data)
                struct_group.create_dataset('wyckoff_data', data=np.string_(wyckoff_json))
                
                structure_idx += 1
        
        # Create convenient dataset arrays for ML training
        # These are pre-processed, padded arrays ready for model input
        
        # Allocate arrays
        all_lattice_params = np.zeros((total_structures, 6), dtype=np.float32)
        all_elements = np.zeros((total_structures, max_atoms), dtype=np.int32)
        all_positions = np.zeros((total_structures, max_atoms, 3), dtype=np.float32)
        all_masks = np.zeros((total_structures, max_atoms), dtype=np.int8)
        all_space_groups = np.zeros(total_structures, dtype=np.int32)
        
        # Fill arrays
        idx = 0
        for sg, structures in structures_dict.items():
            for structure in structures:
                atoms = structure.to_ase()
                cell = atoms.get_cell()
                positions = atoms.get_scaled_positions()
                elements = atoms.get_chemical_symbols()
                
                # Lattice parameters
                all_lattice_params[idx] = cell.cellpar()
                
                # Space group
                all_space_groups[idx] = sg
                
                # Atoms data
                for j, (element, position) in enumerate(zip(elements, positions)):
                    all_elements[idx, j] = element_to_idx[element]
                    all_positions[idx, j] = position
                    all_masks[idx, j] = 1
                
                idx += 1
        
        # Create ML-ready datasets
        ml_group = f.create_group('ml_ready')
        ml_group.create_dataset('lattice_parameters', data=all_lattice_params)
        ml_group.create_dataset('atom_elements', data=all_elements)
        ml_group.create_dataset('atom_positions', data=all_positions)
        ml_group.create_dataset('atom_masks', data=all_masks)
        ml_group.create_dataset('space_groups', data=all_space_groups)
        
        # Create train/val/test split indices (80/10/10)
        indices = np.random.permutation(total_structures)
        train_idx = indices[:int(0.8 * total_structures)]
        val_idx = indices[int(0.8 * total_structures):int(0.9 * total_structures)]
        test_idx = indices[int(0.9 * total_structures):]
        
        split_group = f.create_group('data_split')
        split_group.create_dataset('train_indices', data=train_idx)
        split_group.create_dataset('validation_indices', data=val_idx)
        split_group.create_dataset('test_indices', data=test_idx)
    
    print(f"\n💾 Saved {total_structures} structures to HDF5 file: {filename}")
    print(f"    Max atoms per structure: {max_atoms}")
    print(f"    Data is ready for ML model input in the 'ml_ready' group")
    
    return total_structures

# Update the main function to also save HDF5 format
if __name__ == "__main__":
    # Import datetime here to avoid issues if this module is imported elsewhere
    import datetime
    
    # Ask user for number of samples
    try:
        n_samples = int(input("Enter number of samples to generate per space group (default: 1): ") or "1")
        if n_samples <= 0:
            print("Number of samples must be positive, using default value of 1")
            n_samples = 1
    except ValueError:
        print("Invalid input, using default value of 1")
        n_samples = 1
    
    # Generate the samples
    structures = generate_n_samples_per_space_group(n_samples, output_file="crystal_structures.json", check_duplicates=True)
    
    # Save in transformer-friendly format
    save_structures_for_transformer(structures, "transformer_structures.json")
    
    # Save in HDF5 format for ML
    save_structures_to_hdf5(structures, "crystal_structures.h5")
