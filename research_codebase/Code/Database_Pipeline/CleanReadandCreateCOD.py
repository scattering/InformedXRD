import os
import zipfile
import torch
import numpy as np
import pandas as pd
import tempfile
import traceback
import torch.multiprocessing as mp
from pathlib import Path
from pycrysfml import cfml_utilities
import gc
import multiprocessing

#Held up Files
#Processing: cif/7/15/08/7150821.cif
os.environ["FULLPROF"] = "/storage/ejb12/FullProf"
torch.set_num_threads(1)  

df = pd.read_csv('spaceGroups.csv')
mapping = torch.load('mapping - Copy.pt')['Bravais Lattice']
reverseMapping = {value: key for key, value in mapping.items()}

def extract_name(filename):
    return Path(filename).stem

def normalize(y):
    rang = np.max(y) - np.min(y)
    return (y - np.min(y)) / rang if rang != 0 else np.zeros_like(y)
def generate_json(input_path):
    identifier = extract_name(input_path)
    return {
        "cif": input_path,
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
                    "_phase": [{"_label": identifier, "_scale": 1}],
                    "_pd_background": [
                        {"_2theta": 3, "_intensity": 0},
                        {"_2theta": 155, "_intensity": 0}
                    ]
                }
            }
        ]
    }
def get_space_group(cif_content):
    try:
        # Ensure cif_content is a string
        if isinstance(cif_content, bytes):
            cif_content = cif_content.decode('ISO-8859-1')

        lines = cif_content.splitlines()

        for line in lines:
            if "Unknown" in line:
                return None  
            elif '_space_group_IT_number' in line:
                parts = line.split()
                if len(parts) < 2:
                    continue
                space_group = parts[1]  # Assuming the second item is the space group number
                match = df.loc[df['Point Group'] == space_group, 'Bravais Lattice']

                if match.empty:
                    print("DEBUG: No match found in DataFrame for space group.")
                    return None
                bravais_lattice = match.values[0]
                mapped_value = reverseMapping.get(bravais_lattice, None)

                return mapped_value  # Return the mapped space group

        print("DEBUG: No valid space group found in CIF file.")
        return None  

    except Exception as e:
        print(f"ERROR in get_space_group: {e}")
        return None

def process_cif_file(zip_path, cif_filename):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_obj:
            if cif_filename == 'cif/7/15/08/7150821.cif':
                return None
            print(f"Processing: {cif_filename}")
            with zip_obj.open(cif_filename) as file:
                cif_content = file.read()

            space_group = get_space_group(cif_content)
            if space_group is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".cif") as tmp_file:
                    tmp_file.write(cif_content)
                    tmp_file_path = tmp_file.name  # Get the temp file path
                data = generate_json(tmp_file_path)  # Pass the temporary file path
                x, y = cfml_utilities.powder_pattern_from_json(data)

                if y is not None:
                    y = normalize(y)
                    return y.tolist(), space_group, cif_filename  
                else:
                    print(f"ERROR: No diffraction data for {cif_filename}")
        return None
    except Exception as e:
        print(f"Failed processing {cif_filename}: {e}")
        return None

    
def create_pt(x, y, paths):
    x_tensor = torch.stack([torch.tensor(data) for data in x], dim=0)
    y_tensor = torch.tensor(y)
    num_samples = len(y)

    test_size = max(1, int(0.05 * num_samples))  # Ensure at least 1 test sample
    train_size = num_samples - test_size

    indices = torch.randperm(num_samples)
    train_indices, test_indices = indices[:train_size], indices[train_size:]

    torch.save({'X': x_tensor[test_indices], 'Y': y_tensor[test_indices]}, f'400000COD_TestTensor.pt')
    torch.save({'X': x_tensor[train_indices], 'Y': y_tensor[train_indices]}, f'400000COD_TrainTensor.pt')

    df = pd.DataFrame({'y': y_tensor[test_indices].tolist(), 'path': [paths[i] for i in test_indices]})
    df.to_csv('COD_Bravais_and_Path.csv', index=False)

def process_cifs_parallel(zip_path, num_workers):
    with zipfile.ZipFile(zip_path, 'r') as zip_obj:
        cifs = [filename for filename in zip_obj.namelist() if filename.endswith('.cif')][:200000]

    x_data, y_data, paths = [], [], []
    
    with mp.Pool(num_workers) as pool:
        results = []
        for cif in cifs:
            result = pool.apply_async(process_cif_file, (zip_path, cif))
            results.append((cif, result))

        for cif, result in results:
            try:
                res = result.get(timeout=10)  # Enforce a 5-second timeout
                if res is not None:
                    x_data.append(res[0])
                    y_data.append(res[1])
                    paths.append(res[2])
            except multiprocessing.TimeoutError:
                print(f"Timeout: {cif} took too long to process.")
            except Exception as e:
                print(f"Error processing {cif}: {e}")

    del results
    gc.collect()
    print(f"Processed {len(cifs)} files. Memory cleanup done.")

    create_pt(x_data, y_data, paths)

    with open('FullyDone.txt', 'w') as f:
        f.write("Done")

def main():
    try:
        num_workers = min(64, mp.cpu_count())  
        print(f"Using {num_workers} workers for multiprocessing.")
        process_cifs_parallel("cod-cifs-mysql.zip", num_workers)
    except Exception as e:
        with open("error_log.txt", "a") as f:
            f.write(f"Error: {str(e)}\n")
            f.write(traceback.format_exc() + "\n")

if __name__ == "__main__":
    mp.set_start_method("fork" if os.name != "nt" else "spawn", force=True)
    main()
