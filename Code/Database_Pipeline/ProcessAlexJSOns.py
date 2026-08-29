import json
import os
from pymatgen.core import Structure
from pycrysfml import cfml_py_utilities
import traceback

# Import your XRD generation method from your existing script
from your_xrd_module import generate_diff_pattern  # Adjust this import accordingly

globpath = 'JSON/PBE/'
dir_list = os.listdir(globpath)
paths = [os.path.join(globpath, p) for p in dir_list]

# Just pick the first available file and entry for testing
selected_file = paths[0]
with open(selected_file, 'r') as file:
    json_data = json.load(file)
    entry = json_data['entries'][0]
    try:
        structure = Structure.from_dict(entry['data']['structure'])
        result = generate_diff_pattern(structure)
        if result:
            two_theta, intensity = result
            print("Successfully generated XRD pattern!")
            print("2theta:", two_theta[:10])
            print("Intensity:", intensity[:10])
    except Exception as e:
        traceback.print_exc()
        print("Error generating XRD pattern:", e)