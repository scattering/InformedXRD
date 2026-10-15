# ===============================
# Environment setup
# ===============================
VENV_PATH="$SCRATCH/nvidia-scratch/bin/activate"   # path to your venv
PROJECT="project"

# Arguments (override when calling sbatch if needed)
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/scratch/10486/UserB/Models/xrd_model_l0yy9wv9.pth"}
DATA_PATH=${DATA_PATH:-"/scratch/10486/UserB/Data/10Million_pyxtal_RRUFFF-type_repacked.hdf5"}
DATA_PATH_INTERP=${DATA_PATH_INTERP:-"/scratch/10486/UserB/Data/RRUFF_interp_data.hdf5"}
# RUN_ID=${RUN_ID:-"kvo9yx4s"}

# Model args
SPEC_LENGTH=8501
NUM_CLASSES=99
PATCH_SIZE=25
EMBED_DIM=256
DEPTH=12
NUM_HEADS=8
MLP_RATIO=4.0
TWO_THETA_MIN=5.0
TWO_THETA_MAX=90.0
DROP_RATIO=0.0
MLP_HIDDEN_DIM=512
START_COL=1
END_COL=8501

# Confusion Matrix
CONF_MATRIX=True

# Check NaN
CHECK_NAN=False

# new HKL plotting parameters
HKL_DB_PATH=${HKL_DB_PATH:-"/scratch/10486/UserB/Data/intrep_crystal_xrd.db"}
HKL_HDF5_PATH=${HKL_HDF5_PATH:-"$DATA_PATH"}   # reuse same HDF5
HKL_SAMPLE_INDEX=(25 85 275)
MAX_HKL_INDEX=5


echo "Running inference with:"
echo "  CHECKPOINT: $CHECKPOINT_PATH"
echo "  DATA:       $DATA_PATH"
echo "  RUN_ID:     $RUN_ID"
echo "  2θ Range:   $TWO_THETA_MIN – $TWO_THETA_MAX"
echo "  HKL DB:      $HKL_DB_PATH"
echo "  HKL Sample:  ${HKL_SAMPLE_INDEX[@]}"

# Activate virtual environment
source $VENV_PATH

# ===============================
# Run inference
# ===============================

python inference.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --data_path "$DATA_PATH" \
    --data_path_interp "$DATA_PATH_INTERP" \
    --run_id "$RUN_ID" \
    --project "$PROJECT" \
    --two_theta_min "$TWO_THETA_MIN" \
    --two_theta_max "$TWO_THETA_MAX" \
    --spec_length "$SPEC_LENGTH" \
    --num_classes "$NUM_CLASSES" \
    --patch_size "$PATCH_SIZE" \
    --embed_dim "$EMBED_DIM" \
    --depth "$DEPTH" \
    --num_heads "$NUM_HEADS" \
    --mlp_ratio "$MLP_RATIO" \
    --drop_ratio "$DROP_RATIO" \
    --mlp_hidden_dim "$MLP_HIDDEN_DIM" \
    --start_col "$START_COL" \
    --end_col "$END_COL" \
    $( [ "$CONF_MATRIX" = True ] && echo "--conf_matrix" ) \
    $( [ "$CHECK_NAN" = True ] && echo "--check_nan" ) \
    --hkl_db_path "$HKL_DB_PATH" \
    --hkl_hdf5_path "$HKL_HDF5_PATH" \
    --hkl_sample_index "${HKL_SAMPLE_INDEX[@]}" \
    --max_hkl_index "$MAX_HKL_INDEX"
