import datetime

import numpy as np
from numpy import sin, radians, tan, sqrt
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import spglib
import pyxtal
from cctbx import crystal

INSTRUMENT_PARAMS = {
    "wavelength": 1.54030,
    "twotheta_min": 10,
    "twotheta_max": 110,
    "step_size": 0.1,
    "u": 0.025,
    "v": -0.025,
    "w": 0.025
}
OUTPUT_HDF5 = ""
NUM_SAMPLES = 230000

def twotheta_to_d(twotheta, wavelength=1.54):
    """
    Convert 2-theta to d-spacing.
    """
    return wavelength / (2 * sin(radians(twotheta) / 2))

def calcgaussian(x,fwhm):
    """"
    Calculate the cagliotti peakshape for a given center and width.
    The width is the full width at half maximum (FWHM).
    The center is the position of the peak.
    """
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return np.exp(-0.5 * ((x / sigma) ** 2))

class Peak(object):
    scaleFactor = 1
    def __init__(self, center, u, v, w, I, hkl=[None, None, None], shape="Gaussian", eta=0):
        self.center = center    # 2*theta position
        #note that the keyword argument is not great for hkl.  
        self.u = u
        self.v = v
        self.w = w
        self.I = I
        self.shape = shape
        self.eta = eta
        try:
            self.H = np.sqrt(u*(tan(radians(center/2))**2)
                          + v*tan(radians(center/2)) + w)
            self.scale = self.I * Peak.scaleFactor
        except ValueError:
            self.H = 0
            self.scale = 0
        self.hkl = hkl

    # __call__: returns the value of the Peak at some 2*theta positions
    def __call__(self, x):
        if self.shape.lower() == 'gaussian':
            return self.scale * calcgaussian(x-self.center, self.H)
            #return [self.scale * calcgaussian(value-self.center, self.H) for value in x]
        elif self.shape.lower() == 'pseudovoigt':
            return [self.scale * funcs.calcpseudovoigt(value-self.center, self.H, float(self.eta)) for value in x]
        elif self.shape.lower() == 'lorentzian':
            return [self.scale * funcs.calclorentzian(value-self.center, self.H) for value in x]
        else:
            print(f"Unsupported Peak-shape: {self.shape}\n")
            quit()
    def add(self, v, x):
        # only add to nearby 2*theta positions
        idx = (x>self.center-self.H*3) & (x<self.center+self.H*3)
        try:
            v[idx] += self.__call__(np.array(x)[idx])
        except:
            pass


def generate_diffraction_pattern_helper(spacegroup_number, unit_cell, wavelength, twotheta_min, twotheta_max, twothstep, u, v, w, peak_scale=1000):
    """
    Generate a diffraction pattern for a given space group and unit cell.

    Parameters:
        spacegroup_number (int): Space group number.
        unit_cell (tuple): Unit cell parameters (a, b, c, alpha, beta, gamma).
        wavelength (float): Wavelength of the X-ray beam.
        twotheta_min (float): Minimum 2-theta value.
        twotheta_max (float): Maximum 2-theta value.
        twothstep (float): Step size for 2-theta.
        u, v, w (float): Peak shape parameters.
        peak_scale (float): Scale factor for peak intensity.

    Returns:
        tuple: (twotheta, I) where `twotheta` is an array of 2-theta values and `I` is the corresponding intensity array.
    """
    # Create symmetry object
    symmetry = crystal.symmetry(unit_cell=unit_cell, space_group=spacegroup_number)

    # Calculate d-spacing range
    d_max = twotheta_to_d(twotheta_min, wavelength=wavelength)
    d_min = twotheta_to_d(twotheta_max, wavelength=wavelength)

    # Build Miller set and get allowed reflections
    miller_set = symmetry.build_miller_set(anomalous_flag=False, d_min=d_min, d_max=d_max)
    allowed_reflections = miller_set.indices()
    uc = symmetry.unit_cell()

    # Initialize 2-theta and intensity arrays
    twotheta = np.arange(twotheta_min, twotheta_max, twothstep)
    I = np.zeros(len(twotheta))

    # Calculate intensity for each allowed reflection
    for hkl in allowed_reflections:
        d_spacing = uc.d(hkl)
        twotheta_center = 2 * np.arcsin(wavelength / (2 * d_spacing)) * 180 / np.pi
        ps=np.random.uniform(5, peak_scale)
        peak = Peak(twotheta_center, u, v, w, ps, hkl=hkl, shape="Gaussian", eta=0)
        I += peak(twotheta)

    # Apply Lorentz correction
    lorentz = 1 / (np.sin(np.radians(twotheta)) * np.sin(np.radians(twotheta / 2)))
    I *= lorentz

    # Normalize intensity
    I = 1000 * I / I.max()

    return twotheta, I


def plot_diffraction_pattern(twotheta, I):
    """
    Plot the diffraction pattern.

    Parameters:
        twotheta (array): Array of 2-theta values.
        I (array): Array of intensity values.
    """
    plt.errorbar(twotheta, I, yerr=np.sqrt(I), fmt='o', label='Data')
    plt.xlabel('2θ (degrees)')
    plt.ylabel('Intensity')
    plt.title('Diffraction Pattern')
    plt.legend()
    plt.show()


def generate_unit_cell(spacegroup_number):
    """
    Generates a random unit cell given the spacegroup. 

    Parameter:
        spacegroup_number (int): Space group number.
    """
    group = pyxtal.symmetry.Group(spacegroup_number)
    lattice_type = group.lattice_type

    min_a, max_a = 4.0, 10.0 # Could adjust these based of ICSD or CSD distribution
    min_alpha, max_alpha = 60.0, 120.0

    a, b, c = np.random.uniform(min_a, max_a, 3)
    alpha, beta, gamma = np.random.uniform(min_alpha, max_alpha, 3)

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

def generate_diffraction_pattern(spacegroup_number, a, b , c, alpha, beta, gamma):
    twotheta, I = generate_diffraction_pattern_helper(
        spacegroup_number,
        unit_cell = (a, b, c, alpha, beta, gamma),
        wavelength = INSTRUMENT_PARAMS["wavelength"],
        twotheta_min = INSTRUMENT_PARAMS["twotheta_min"],
        twotheta_max = INSTRUMENT_PARAMS["twotheta_max"],
        twothstep = INSTRUMENT_PARAMS["step_size"],
        u = INSTRUMENT_PARAMS["u"],
        v = INSTRUMENT_PARAMS["v"],
        w = INSTRUMENT_PARAMS["w"]
        #u = 0.347,
        #v = -0.278,
        #w = 0.166
    )
    return twotheta, I

def save_patterns_to_hdf5(patterns_dict, filename="diffraction_patterns.h5"):
    """
    Save powder diffraction patterns to an HDF5 file, grouped by space group.
    
    Parameters
    ----------
    patterns_dict : dict
        Dictionary with space group numbers as keys, and list of (twotheta, intensity) tuples as values.
        
    filename : str
        Output filename for HDF5 file.
        
    Returns
    -------
    int
        Total number of patterns saved.
    """
    try:
        import h5py
        import numpy as np
    except ImportError:
        print("Error: h5py and numpy are required. Install them with 'pip install h5py numpy'.")
        return 0
    
    # Count total patterns
    total_patterns = sum(len(patterns) for patterns in patterns_dict.values())
    
    with h5py.File(filename, 'w') as f:
        # Metadata
        f.attrs['total_patterns'] = total_patterns
        f.attrs['creation_date'] = np.bytes_(datetime.datetime.now().isoformat())
        
        # Group for all patterns
        patterns_group = f.create_group('patterns')
        
        pattern_idx = 0
        
        for sg, patterns in patterns_dict.items():
            # Create group for each space group
            sg_group = patterns_group.create_group(f'sg_{sg}')
            
            for i, (twotheta, intensity) in enumerate(patterns):
                # Create subgroup for each pattern
                pattern_group = sg_group.create_group(f'pattern_{i}')
                
                # Save the arrays
                pattern_group.create_dataset('twotheta', data=np.array(twotheta, dtype=np.float32))
                pattern_group.create_dataset('intensity', data=np.array(intensity, dtype=np.float32))
                
                # Save space group as attribute too
                pattern_group.attrs['space_group'] = sg
                
                pattern_idx += 1
        
        # Create ML-friendly datasets too (optional)
        all_twotheta = []
        all_intensity = []
        all_space_groups = []
        
        for sg, patterns in patterns_dict.items():
            for (twotheta, intensity) in patterns:
                all_twotheta.append(np.array(twotheta, dtype=np.float32))
                all_intensity.append(np.array(intensity, dtype=np.float32))
                all_space_groups.append(sg)
        
        # Find maximum length (to pad shorter arrays)
        max_len = max(len(x) for x in all_twotheta)
        
        # Allocate
        all_twotheta_padded = np.zeros((total_patterns, max_len), dtype=np.float32)
        all_intensity_padded = np.zeros((total_patterns, max_len), dtype=np.float32)
        all_masks = np.zeros((total_patterns, max_len), dtype=np.int8)  # Mask to know where data is valid
        all_space_groups = np.array(all_space_groups, dtype=np.int32)
        
        # Fill
        for idx, (theta, inten) in enumerate(zip(all_twotheta, all_intensity)):
            n = len(theta)
            all_twotheta_padded[idx, :n] = theta
            all_intensity_padded[idx, :n] = inten
            all_masks[idx, :n] = 1
        
        # Save ML-ready datasets
        ml_group = f.create_group('ml_ready')
        ml_group.create_dataset('twotheta', data=all_twotheta_padded)
        ml_group.create_dataset('intensity', data=all_intensity_padded)
        ml_group.create_dataset('masks', data=all_masks)
        ml_group.create_dataset('space_groups', data=all_space_groups)
        
        # Create a random split (80% train, 10% val, 10% test)
        indices = np.random.permutation(total_patterns)
        train_idx = indices[:int(0.8 * total_patterns)]
        val_idx = indices[int(0.8 * total_patterns):int(0.9 * total_patterns)]
        test_idx = indices[int(0.9 * total_patterns):]
        
        split_group = f.create_group('data_split')
        split_group.create_dataset('train_indices', data=train_idx)
        split_group.create_dataset('validation_indices', data=val_idx)
        split_group.create_dataset('test_indices', data=test_idx)
    
    print(f"\n💾 Saved {total_patterns} diffraction patterns to HDF5 file: {filename}")
    print(f"    Max points per pattern: {max_len}")
    print(f"    Data is ready for ML model input in the 'ml_ready' group")
    
    return total_patterns
    
def main():
    patterns_dict = {}
    
    num_samples_per_sg = NUM_SAMPLES//230
    for sg_number in range(1, 231):
        try:
            for _ in range(num_samples_per_sg):
                a, b, c, alpha, beta, gamma = generate_unit_cell(sg_number)
                twotheta, I = generate_diffraction_pattern(sg_number, a, b, c, alpha, beta, gamma)
                print("Generated pattern ", sg_number)
                if sg_number not in patterns_dict:
                    patterns_dict[sg_number] = []
                patterns_dict[sg_number].append((twotheta, I))

        except Exception as e:
            print(f"Error generating space group {sg_number}: {e}")


    # Save all to HDF5
    save_patterns_to_hdf5(patterns_dict, filename=OUTPUT_HDF5)

def test_file():
    """Check patterns in HDF5 file"""
    import h5py

    # Open the HDF5 file
    with h5py.File(OUTPUT_HDF5, 'r') as f:
        # Print the top-level groups
        print("Top-level keys:", list(f.keys()))
        
        # Print the attributes
        print("Attributes:", dict(f.attrs))

        # Dive into the 'patterns' group
        patterns_group = f['patterns']
        print("\nSpace groups available:", list(patterns_group.keys()))
        
        # For example, look into space group 62
        sg62 = patterns_group['sg_62']
        print("\nPatterns inside sg_62:", list(sg62.keys()))
        
        # Load one diffraction pattern
        pattern0 = sg62['pattern_0']
        twotheta = pattern0['twotheta'][:]
        intensity = pattern0['intensity'][:]

        # Plot it
        plt.figure(figsize=(8, 5))
        plt.plot(twotheta, intensity)
        plt.xlabel(r"2$\theta$ (degrees)")
        plt.ylabel("Intensity")
        plt.title("Powder Diffraction Pattern (sg_62, pattern_0)")
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    main()
    #test_file()
