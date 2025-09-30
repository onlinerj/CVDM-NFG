#!/bin/bash
# Training script optimized for NVIDIA H100 GPU
# Expected total time: ~7-10 hours for complete pipeline

set -e  # Exit on error

echo "================================================================"
echo "H100 Optimized Training Pipeline"
echo "================================================================"
echo ""

# Check if data path is provided
if [ -z "$1" ]; then
    echo "Usage: bash train_h100.sh /path/to/dataset"
    echo ""
    echo "Expected dataset structure:"
    echo "  dataset/"
    echo "    ├── train/video_0, video_1, ..., video_999"
    echo "    ├── val/video_1000, video_1001, ..., video_1999"
    echo "    └── hidden/video_15000, ..., video_16999"
    exit 1
fi

DATA_PATH=$1
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Configuration:"
echo "  Data path: $DATA_PATH"
echo "  Timestamp: $TIMESTAMP"
echo "  GPU: H100 (80GB)"
echo ""

# Create output directories
mkdir -p experiments/h100_${TIMESTAMP}
mkdir -p experiments/h100_${TIMESTAMP}/logs
mkdir -p experiments/h100_${TIMESTAMP}/checkpoints

# Enable H100 optimizations
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

echo "================================================================"
echo "Step 1/2: Training UNet++ (Segmentation)"
echo "================================================================"
echo "Expected time: 1-1.5 hours"
echo "Expected result: 98-99% IoU"
echo ""

python src/Unet/train_h100.py $DATA_PATH 2>&1 | tee experiments/h100_${TIMESTAMP}/logs/unet_training.log

if [ -f "unetplusplus_h100.pt" ]; then
    mv unetplusplus_h100.pt experiments/h100_${TIMESTAMP}/checkpoints/
    echo "✓ UNet training complete!"
    echo ""
else
    echo "✗ UNet training failed!"
    exit 1
fi

echo "================================================================"
echo "Step 2/2: Training Diffusion Model"
echo "================================================================"
echo "Expected time: 6-9 hours"
echo "Expected result: 37-40% Jaccard"
echo ""

python src/mcvd/main.py \
    --config src/mcvd/configs/next_frame_h100.yml \
    --data_path $DATA_PATH \
    --exp experiments/h100_${TIMESTAMP}/diffusion \
    --ni \
    2>&1 | tee experiments/h100_${TIMESTAMP}/logs/diffusion_training.log

echo ""
echo "================================================================"
echo "Training Complete!"
echo "================================================================"
echo ""
echo "Results saved in: experiments/h100_${TIMESTAMP}/"
echo ""
echo "To run inference on test set:"
echo "  python src/mcvd/test_diffusion_hidden.py \\"
echo "    $DATA_PATH \\"
echo "    experiments/h100_${TIMESTAMP}/diffusion/logs/checkpoint_XXXXX.pt \\"
echo "    experiments/h100_${TIMESTAMP}/predictions"
echo ""
echo "Then apply segmentation:"
echo "  python src/run_segmentation.py \\"
echo "    experiments/h100_${TIMESTAMP}/checkpoints/unetplusplus_h100.pt \\"
echo "    experiments/h100_${TIMESTAMP}/predictions/"
echo ""
echo "Total training time: ~7-10 hours"
echo "Expected final score: 37-40% Jaccard"
echo "================================================================"

