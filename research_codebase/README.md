# Neural Network Systematics for Crystal Structure Classification

> **About this directory (added for the camera-ready release).** This is the
> full research codebase of the project (`scattering/ai-diffraction`),
> included verbatim from the snapshot prepared for the NeurIPS 2026 AI for
> Science workshop submission. The paper-facing, curated artifacts live one
> level up; see the top-level [README](../README.md) and
> [PROVENANCE](../PROVENANCE.md). The placeholder names described in the note
> below are retained as deliberate redactions of user names, institutions and
> allocation identifiers; nothing in the paper depends on them.

Anonymized code release supporting a double-blind submission. Here we apply the
methods of AI to the analysis of diffraction data.

> **Note on anonymization.** Author names, usernames, email addresses,
> institutional affiliations, and HPC allocation identifiers throughout this
> repository have been replaced with neutral placeholders (`UserA`–`UserF`,
> `OrgA`/`OrgB`, `ALLOC1`–`ALLOC3`). Paths that formerly pointed at personal
> home or scratch directories now reference placeholder locations and must be
> repointed before any script will run. Internal project notes, referee
> correspondence, and release checklists have been removed.

## TACC Stampede 3 - SSH Instructions - Intel - Interactive Mode


1. Login to TACC

```
ssh myusername@stampede3.tacc.utexas.edu
```

2. Change to $SCRATCH directory

```
cds
```

3. Load module: 

```
ml use /scratch/projects/compilers/modulefiles/
```

4. Install Intel OneAPI

```
source /scratch/projects/compilers/intel25.0/setvars.sh --force
```

5. Activate environment

```
python -m venv xxx
source xxx/bin/activate
```

6. install intel ipex-llm[xpu]:

```
python -m pip install torch==2.5.1+cxx11.abi torchvision==0.20.1+cxx11.abi torchaudio==2.5.1+cxx11.abi intel-extension-for-pytorch==2.5.10+xpu oneccl_bind_pt==2.5.0+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```

7. Test by activating idev/interactive mode (PVC queue has Intel GPUs): 

```
idev -p pvc -N 2 -n 2 -m 15
source xxx/bin/activate
export OCL_ICD_VENDORS=/etc/OpenCL/vendors
export CCL_ROOT=${CONDA_PREFIX}
python -c "import torch; import intel_extension_for_pytorch as ipex; print(torch.__version__); print(ipex.__version__); [print(f'[{i}]: {torch.xpu.get_device_properties(i)}') for i in range(torch.xpu.device_count())];"
```
If you encounter an error, change the ${CONDA_PREFIX} into Python virtual env path:
```
echo $VIRTUAL_ENV
```

A list of GPUs will display:
```
[0]: _XpuDeviceProperties(name='Intel(R) Data Center GPU Max 1550', platform_name='Intel(R) oneAPI Unified Runtime over Level-Zero', type='gpu', driver_version='1.3.27642', total_memory=65536MB, max_compute_units=512, gpu_eu_count=512, gpu_subslice_count=64, max_work_group_size=1024, max_num_sub_groups=64, sub_group_sizes=[16 32], has_fp16=1, has_fp64=1, has_atomic64=1)
[1]: _XpuDeviceProperties(name='Intel(R) Data Center GPU Max 1550', platform_name='Intel(R) oneAPI Unified Runtime over Level-Zero', type='gpu', driver_version='1.3.27642', total_memory=65536MB, max_compute_units=512, gpu_eu_count=512, gpu_subslice_count=64, max_work_group_size=1024, max_num_sub_groups=64, sub_group_sizes=[16 32], has_fp16=1, has_fp64=1, has_atomic64=1)
...
```


## TACC Stampede 3 - SSH Instructions - NVIDIA - Interactive Mode


1. Login to TACC

```
ssh myusername@stampede3.tacc.utexas.edu
```

2. Change to $SCRATCH directory

```
cds
```

3. Load module: 

```
ml use /scratch/projects/compilers/modulefiles/
```

4. Activate environment

```
python -m venv xxx
source xxx/bin/activate
```

5. Install torch prerequisites (Refer to respective requirements.txt)

```
pip install torch
pip install numpy
pip install importlib-metadata
pip install wandb
pip install h5py
pip install tqdm
pip install torchinfo
pip install unicodedata2
...
```

6. Test by activating idev/interactive mode (H100 queue has NVIDIA GPUs): 

```
idev -p h100 -N 1 -n 1 -m 15
source xxx/bin/activate
# Test PyTorch + CUDA device visibility and properties
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA device count: {torch.cuda.device_count()}')

for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'[GPU {i}] {props.name}, {props.total_memory / (1024**3):.2f} GB, Compute Capability: {props.major}.{props.minor}')
"
```

A list of GPUs will display:
```
PyTorch version: 2.7.1+cu126
CUDA available: True
CUDA device count: 4
[GPU 0] NVIDIA H100, 93.10 GB, Compute Capability: 9.0
[GPU 1] NVIDIA H100, 93.10 GB, Compute Capability: 9.0
[GPU 2] NVIDIA H100, 93.10 GB, Compute Capability: 9.0
[GPU 3] NVIDIA H100, 93.10 GB, Compute Capability: 9.0
...
```

## TACC Stampede 3 - Notebook Instructions


1. Change to $SCRATCH directory

```
cds
```

2. Activate environment

```
source xxx/bin/activate
```

3. Install the ipykernel library into the virtual environment

```
python3 -m pip install jupyterlab
```

4. Use the ipykernel library to install the virtual environment for use with Jupyter

```
python -m ipykernel install --user --name=xxx
```

5. Locate your kernel spec for xxx

```
jupyter kernelspec list
```

6. Open the kernel.json file inside the appropriate directory for the new kernel and modify the kernel launch script to ensure that the setvars.sh script is sourced before the Jupyter kernel starts.

```
{ "argv": [ 
"<YOUR KERNEL’S PYTHON LOCATION>", "-m", "ipykernel_launcher", "-f", "{connection_file}" ], 
"display_name": "oneapi", "language": "python",
 "metadata": { "debugger": true } }
```
TO 
```
{
  "argv": [
    "/bin/bash",
    "-c",
    "source /scratch/projects/compilers/intel25.0/setvars.sh --force && exec <YOUR KERNEL’S PYTHON LOCATION> -m ipykernel_launcher -f {connection_file}"
  ],
  "display_name": "xxx",
  "language": "python"
}
```

Example:
```
{
  "argv": [
    "/bin/bash",
    "-c",
    "source /scratch/projects/compilers/intel25.0/setvars.sh --force && exec /scratch/ALLOC1/UserA/oneapi/bin/python -m ipykernel_launcher -f {connection_file}"
  ],
  "display_name": "oneapi",
  "language": "python",
  "metadata": {
    "debugger": true
  }
}
```

## TACC Stampede 3 - Data Generation

1. Follow instructions below to set up CCTBX pre-requisites in Conda:
https://cci.lbl.gov/docs/cctbx/installation/



## TACC Stampede 3 - File Transfers

```
scp /local/path/ <username>@<machine>.tacc.utexas.edu:/path/to/tacc 


```

## Setting up shared dirs:
getfacl path -> shows permission for that pathsince 
$STOCKYARD is the highest "prviate file", i needed to set read access to stockyard (if there are other "private" super direcotries,  you must set super access there too")
shared_dir -> shared directory
setfacl -m u:testuser:rwx shared_dir   <- gives read/write/executre to shared_dir folder(recursivly applies rwx to all files in shared_dir)
setfacl -R -m u:UserA:rwx shared_dir/
setfacl -R -m u:UserB:rwX shared_dir/(sets default condions for files in shared_dir
setfacl -d -m u:UserA:rwx shared_dir/
setfacl -d -m u:UserB:rwx shared_dir/

## TACC Stampede 3 - SLURM

### Batch File

1. Create a .sh file with the below:

```
#!/bin/bash
#SBATCH -J supervised_trainer (job name)
#SBATCH -o supervised_trainer.o%j (output file)
#SBATCH -e supervised_trainer.e%j (error file)
#SBATCH -p rtx (partion)
#SBATCH -N 1 (# nodes)
#SBATCH -n 1 (# tasks)
#SBATCH -t 00:05:00
#SBATCH --mail-user=userf@example.org
#SBATCH --mail-type=all

(This is where you write what you want to execute)
(Ex: <python path> <file.py>)
```

### Managing Jobs

1. Queue job:
```
sbatch <name>.sh: Puts job into queue
```

2. Check queue:
```
squeue: Checks on entire queue
```

3. Check specific job:
```
squeue -u <username>: Checks on jobs in queue
```

4. Cancel specific job:
```
scancel <job id>
```

### PIP Install

```
module load python3/3.9.2 # Feel free to choose another version
python3 -m venv env
source env/bin/activate
pip3 install --upgrade pip
pip install <name>
```

### Install Modules or activate relevant Conda or venv environment

```
module load python3/3.9.2
python3 -m venv env
source env/bin/activate
pip3 install --upgrade pip
pip install urllib3==2.2.2
pip3 install torch==2.3.1 torchvision torchaudio urllib3==2.2.2 --extra-index-url https://download.pytorch.org.whl/cu121
pip list
```

### Multi-GPU

```
sbatch test.sh --gpus=3
```

## TACC Stampede 3 - Modules

1. TACC uses lmod for module management. Below are default modules in the TACC Stampede 3 environment:
```
Currently Loaded Modules:
  1) intel/24.0   2) impi/21.11   3) autotools/1.4   4) cmake/4.1.1   5) xalt/3.1.1   6) python/3.12.11   7) TACC
```
2. Purge modules using the ```module purge``` command and load the below modules using ```module load <module/version>```:
```
Currently Loaded Modules:
  1) ucc/1.3.0   2) ucx/1.18.1   3) cuda/12.4 (g)   4) openmpi/5.0.8   5) nvidia/24.5   6) python/3.9.18
```
3. Use ```module default save``` to save the default modules.