import os
import traceback
import pandas
import h5py
import torch
from multiprocessing import Pool, cpu_count
from pycrysfml import cfml_py_utilities

os.environ["FULLPROF"] = "/storage/ejb12/FullProf"

def create_json(cif_path):
    """ Creates JSON input for pyCrysfml generation"""
    data = {
            "cif":cif_path,
            "experiments": [
                {
                    "NPD": {
                        #"_diffrn_source": "nuclear reactor", uncomment for neutron source
                        #_diffrn_radiation_probe": "neutron" uncomment for neutron
                        "_diffrn_radiation_probe": "xray", # comment for neutron
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
                        "_pd_meas_2theta_range_min": 3,
                        "_pd_meas_2theta_range_max": 155,
                        "_pd_meas_2theta_range_inc": 0.05,
                        "_phase": [{
                            "_label": "placeholder",
                            "_scale": 1}],
                        "_pd_background": [{
                            "_2theta": 0,
                            "_intensity": 0
                        },{
                            "_2theta": 155,
                            "_intensity": 0
                        }]
                }}
            ]}
    return data

def generate_pattern(cif_file):
    """Processes a .cif file and returns the pattern array."""
    try:
        data = create_json(cif_file)
        _, y = cfml_py_utilities.cw_powder_pattern_from_dict(data)
        return y
    except Exception as e:
        traceback.print_exc()
        return None


def create_h5py(x, y, output_path):
    """Save results as HDF5."""
    x_tensor = torch.stack([torch.tensor(data, dtype=torch.float32) for data in x], dim=0)
    y_tensor = torch.tensor(y, dtype=torch.long)
    num_samples = len(y)

    test_size = max(1, int(0.05 * num_samples))
    train_size = num_samples - test_size
    indices = torch.randperm(num_samples)
    train_idx, test_idx = indices[:train_size], indices[train_size:]

    with h5py.File(output_path, "w") as f:
        f.create_dataset("X_train", data=x_tensor[train_idx].numpy(), compression="gzip")
        f.create_dataset("y_train", data=y_tensor[train_idx].numpy(), compression="gzip")
        f.create_dataset("X_val", data=x_tensor[test_idx].numpy(), compression="gzip")
        f.create_dataset("y_val", data=y_tensor[test_idx].numpy(), compression="gzip")

    print(f"HDF5 dataset saved at {output_path}")


def generate_pattern_with_spg(args):
    """Try to generate a pattern and return (pattern, space group) or (None, None) if error."""
    path, spg = args
    try:
        pattern = generate_pattern(path)
        if pattern is None:
            return None, None
        return pattern, spg
    except Exception:
        traceback.print_exc()
        return None, None



if __name__ == '__main__':
    ICSDcif_directory = '/storage/ejb12/Work/ICSD/ICSD cif clean/'
    pathtocsv = 'De_Dupped_ICSD_and_CSD.csv'
    df = pandas.read_csv(pathtocsv)

    df['Local Path'] = df['Local Path'].str.replace('diffraction.', '', regex=False)
    df.loc[df['Database'] == 'ICSD', 'Local Path'] = df.loc[df['Database'] == 'ICSD', 'Local Path'].astype(str) + '.cif'
    df = df[df['Database'] == 'ICSD']

    cif_files_with_spg = list(
        zip((ICSDcif_directory + df["Local Path"].astype(str)).to_numpy(),
            df["Space Group"].to_numpy())
    )
    cif_files_with_spg = cif_files_with_spg
    print(f"Using {cpu_count()} processors...")
    with Pool(cpu_count()) as pool:
        results = pool.map(generate_pattern_with_spg, cif_files_with_spg)

    # Filter out failed results
    results = [res for res in results if res[0] is not None and res[1] is not None]
    ys, sgs = zip(*results)
    print("Finished Running Files")

    # Save HDF5
    create_h5py(ys, sgs, output_path='ICSD.h5')
