# ===============================
# Environment setup
# ===============================
VENV_PATH="$SCRATCH/nvidia/bin/activate"
PROJECT="inference"
CONFIG_PATH="./inference_config.json"

# Activate virtual environment
source $VENV_PATH

# Load required modules (adjust for your cluster)
ml use /scratch/projects/compilers/modulefiles/

# ===============================
# Run inference
# ===============================

python test_inference.py \
    --project "$PROJECT" \
    --config "$CONFIG_PATH" \
