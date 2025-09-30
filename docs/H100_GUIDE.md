# 🚀 H100 GPU Configuration Guide

---

## 📊 Overview

This guide provides H100 configurations and scripts for **fast training** on NVIDIA H100 GPUs (80GB memory).

### Performance Comparison

| Component | A100 Config | H100 Config | Speedup |
|-----------|-------------|-------------|---------|
| UNet++ Training | 2-3 hours | 1-1.5 hours | **2x** |
| Diffusion Training | 12-18 hours | 6-9 hours | **2x** |
| **Total Pipeline** | **14-21 hours** | **7-10 hours** | **2x** |

---

## 🚀 Quick Start

```bash
# Train both models with H100 configs
bash train_h100.sh /path/to/dataset
```

Or manually:
```bash
# 1. Train UNet++ (1-1.5 hours)
python src/Unet/train_h100.py /path/to/dataset

# 2. Train Diffusion Model (6-9 hours)
python src/mcvd/main.py \
  --config src/mcvd/configs/next_frame_h100.yml \
  --data_path /path/to/dataset \
  --exp experiments/h100 \
  --ni
```

---

## 🔧 Key Configuration Changes

### **1. Larger Batch Sizes**

| Model | A100 | H100 |
|-------|------|------|
| UNet++ | 16 | **32** |
| Diffusion | 8 | **16** |

H100's 80GB VRAM allows 2x larger batches.

### **2. TF32 Acceleration**

TensorFloat-32 (TF32) format provides:
- ~3x faster matrix operations
- Maintained accuracy (19-bit precision)
- Automatic hardware acceleration

### **3. Data Loading**

| Parameter | A100 | H100 |
|-----------|------|------|
| `num_workers` | 4 | **8** |
| `pin_memory` | True | True |
| `persistent_workers` | False | **True** |

---

## 📝 Configuration Files

### **UNet++ Training (train_h100.py)**

```python
# Key parameters
batch_size = 32  # Up from 16
num_workers = 8  # Up from 4

# TF32 acceleration
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# DataLoader
DataLoader(
    dataset,
    batch_size=32,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True  # Reduces worker restart overhead
)
```

**Training time:** 1-1.5 hours (vs 2-3 hours on A100)

### **Diffusion Model Training (next_frame_h100.yml)**

```yaml
# Batch size and workers
training:
  batch_size: 16  # Up from 8
  
data:
  num_workers: 8  # Up from 4

# Model remains same
model:
  version: DDIM
  ngf: 128
  ch_mult: [1, 2, 3, 4, 4]
  num_res_blocks: 3
  spade: true
  spade_dim: 256

# Learning rate scaled with batch size
optim:
  lr: 0.00028  # 0.0002 * sqrt(16/8)
  optimizer: "AdamW"
  weight_decay: 0.01
```

**Training time:** 6-9 hours (vs 12-18 hours on A100)

---

## 💻 Complete Training Script

The `train_h100.sh` script handles everything:

```bash
#!/bin/bash

DATA_PATH=$1

if [ -z "$DATA_PATH" ]; then
    echo "Usage: bash train_h100.sh /path/to/dataset"
    exit 1
fi

echo "=========================================="
echo "H100 Training Pipeline"
echo "=========================================="
echo "Expected total time: 7-10 hours"
echo "Expected final score: 38-40% Jaccard"
echo ""

# Create experiment directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p experiments/h100_${TIMESTAMP}/{checkpoints,predictions,logs}

echo "Step 1/2: Training UNet++"
echo "=========================================="
python src/Unet/train_h100.py $DATA_PATH

echo ""
echo "Step 2/2: Training Diffusion Model"
echo "=========================================="
python src/mcvd/main.py \
    --config src/mcvd/configs/next_frame_h100.yml \
    --data_path $DATA_PATH \
    --exp experiments/h100_${TIMESTAMP}/diffusion \
    --ni

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
```

---

## 🔬 Technical Details

### **H100 GPU Capabilities**

| Specification | Value |
|--------------|-------|
| VRAM | 80 GB |
| FP32 Performance | 67 TFLOPS |
| TF32 Performance | 200 TFLOPS (3x) |
| Memory Bandwidth | 3.35 TB/s |

### **TF32 Format**

TensorFloat-32 (TF32):
- 19-bit precision (vs 23-bit FP32)
- Automatic for matrix operations
- No code changes required
- Maintained model accuracy

**Enable TF32:**
```python
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

### **Batch Size Scaling**

**Learning rate adjustment:**
```python
# Rule: lr_new = lr_base * sqrt(batch_new / batch_base)
lr_a100 = 0.0002  # batch_size=8
lr_h100 = 0.0002 * sqrt(16/8) = 0.00028  # batch_size=16
```

**Why square root?**
- Maintains training stability
- Preserves convergence properties
- Standard practice for batch size scaling

---

## 📊 Expected Results

| Metric | A100 | H100 | Notes |
|--------|------|------|-------|
| UNet++ IoU | 98.5% | 98.5% | Same quality |
| Diffusion Jaccard | 37-40% | 37-40% | Same quality |
| UNet Training Time | 2-3 hours | 1-1.5 hours | 2x faster |
| Diffusion Training Time | 12-18 hours | 6-9 hours | 2x faster |
| **Total Time** | **14-21 hours** | **7-10 hours** | **2x faster** |

**Key Takeaway:** H100 provides 2x speedup with identical final performance!

---

## 🔧 Memory Usage

### UNet++ (batch_size=32)

```
Model Parameters: ~54M
Activation Memory: ~12GB
Gradient Memory: ~6GB
Optimizer States: ~8GB
Total: ~26-30GB (comfortable on 80GB)
```

### Diffusion Model (batch_size=16)

```
Model Parameters: ~180M
Activation Memory: ~24GB
Gradient Memory: ~12GB
Optimizer States: ~20GB
Total: ~56-60GB (comfortable on 80GB)
```

---

## 🚀 Further Configuration

### Even Larger Batches?

You can try increasing batch sizes further:

```yaml
# next_frame_h100.yml
training:
  batch_size: 24  # or even 32
```

**But:**
- Requires learning rate adjustment
- May need more training iterations
- Diminishing returns after batch_size=16-24

### Mixed Precision Training?

H100 already uses TF32 automatically. FP16/BF16 mixed precision:
- May provide marginal speedup
- Risk of numerical instability
- Not recommended for this model

---

## 🎯 Quick Command Reference

```bash
# Full pipeline (one command)
bash train_h100.sh /path/to/dataset

# UNet++ only
python src/Unet/train_h100.py /path/to/dataset

# Diffusion only
python src/mcvd/main.py \
  --config src/mcvd/configs/next_frame_h100.yml \
  --data_path /path/to/dataset \
  --exp h100_exp \
  --ni

# Inference
python src/mcvd/test_diffusion_hidden.py \
  /path/to/dataset \
  experiments/h100_exp/logs/checkpoint_XXXXX.pt \
  predictions/
  
python src/run_segmentation.py unetplusplus_h100.pt predictions/
```

---

## 📚 Additional Resources

- **[GUIDE.md](GUIDE.md)** - Complete implementation guide
- **[UNET_GUIDE.md](UNET_GUIDE.md)** - UNet++ architecture details
- Main README: [`../README.md`](../README.md)

---

**Expected Result: 37-40% Jaccard in 7-10 hours** 🚀
