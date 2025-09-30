#!/bin/bash

#SBATCH --job-name=diffusion_training
#SBATCH --account=rjaiswal1           # Change to your account
#SBATCH --partition=gpu                   # Change to your GPU partition
#SBATCH --gres=gpu:h100:1                 # 1 H100 GPU
#SBATCH --cpus-per-task=32
#SBATCH --mem=128GB
#SBATCH --time=72:00:00                   # Max 72 hours
#SBATCH --output=logs/training_%j.out
#SBATCH --error=logs/training_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Set paths
WORK_DIR="/GenAI/Diffusion-Model"    # Change to your repo path
DATA_PATH="/GenAI/datasets/"           # Change to your dataset path

cd $WORK_DIR

# Activate your conda environment
# conda activate your_env_name
# OR
source GenAI/newenv/bin/activate

# Check GPU
echo "Checking GPU availability..."
nvidia-smi

# ============================================
# OPTION 1: Standard Training Pipeline (uncomment to use)
# ============================================

# echo ""
# echo "=========================================="
# echo "Starting Standard Training Pipeline"
# echo "=========================================="

# # Step 1: Train UNet++ Segmentation (2-3 hours on A100)
# echo "Step 1/2: Training UNet++ Segmentation"
# python src/Unet/train.py $DATA_PATH

# # Step 2: Train Diffusion Model (12-18 hours on A100)
# echo "Step 2/2: Training Diffusion Model"
# python src/mcvd/main.py \
#   --config src/mcvd/configs/next_frame.yml \
#   --data_path $DATA_PATH \
#   --exp experiments/slurm_${SLURM_JOB_ID} \
#   --ni

# ============================================
# OPTION 2: Inference Only (uncomment to use)
# ============================================

# echo "Running inference on hidden dataset..."
# python src/mcvd/test_diffusion_hidden.py \
#   $DATA_PATH \
#   experiments/training/logs/checkpoint_XXXXX.pt \
#   predictions/

# echo "Applying segmentation..."
# python src/run_segmentation.py \
#   unetplusplus_best.pt \
#   predictions/

# ============================================
# OPTION 3: H100 Training (7-10 hours total)
# ============================================

echo ""
echo "=========================================="
echo "Starting H100 Training Pipeline"
echo "Expected time: 7-10 hours"
echo "Expected final score: 38-40% Jaccard"
echo "=========================================="

bash train_h100.sh $DATA_PATH

echo ""
echo "=========================================="
echo "Job completed!"
echo "End Time: $(date)"
echo "=========================================="
