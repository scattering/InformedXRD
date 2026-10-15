# ===============================
# Environment setup
# ===============================
VENV_PATH="$SCRATCH/nvidia/bin/activate"
PROJECT="09-07"

# Arguments (override when calling sbatch if needed)
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/scratch/ALLOC1/UserA/models/xrd_model_29spyogr.pth"}
DATA_PATH=${DATA_PATH:-"/scratch/ALLOC1/UserA/data/ext_792k_bal_ref_highres.h5"}
RUN_ID=${RUN_ID:-"wgszqw9f"}
TWO_THETA_MIN=${TWO_THETA_MIN:-10.0}
TWO_THETA_MAX=${TWO_THETA_MAX:-110.0}

# Model args
SPEC_LENGTH=1000
NUM_CLASSES=99
EMBED_DIM=256
DEPTH=8
NUM_HEADS=4
MLP_UNITS=256
FF_DIM=128
TWO_THETA_MIN=10.0
TWO_THETA_MAX=90.0

echo "Running inference with:"
echo "  CHECKPOINT: $CHECKPOINT_PATH"
echo "  DATA:       $DATA_PATH"
echo "  RUN_ID:     $RUN_ID"
echo "  2θ Range:   $TWO_THETA_MIN – $TWO_THETA_MAX"

# Activate virtual environment
source $VENV_PATH

# Load required modules (adjust for your cluster)
ml use /scratch/projects/compilers/modulefiles/

# ===============================
# Run inference
# ===============================

python inference.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --data_path "$DATA_PATH" \
    --run_id "$RUN_ID" \
    --project "$PROJECT" \
    --two_theta_min "$TWO_THETA_MIN" \
    --two_theta_max "$TWO_THETA_MAX" \
    --spec_length "$SPEC_LENGTH" \
    --mlp_units "$MLP_UNITS" \
    --num_classes "$NUM_CLASSES" \
    --embed_dim "$EMBED_DIM" \
    --depth "$DEPTH" \
    --num_heads "$NUM_HEADS" \
    --ff_dim "$FF_DIM"