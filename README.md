# Conditional Diffusion Model for Next Frame Generation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A state-of-the-art video prediction system using conditional diffusion models and semantic segmentation to predict future frames from video sequences of interacting 3D objects.

---

## 🎯 Problem Statement

**Task:** Given 11 frames of videos showing simple 3D shapes (cubes, spheres, cylinders) interacting according to physics principles, predict the 22nd frame along with its semantic segmentation mask.

**Evaluation Metric:** IoU/Jaccard Index between predicted and ground truth segmentation masks.

---

## 🚀 Performance

### **Current Results** ⭐
- Segmentation (UNet++): **98-99% IoU**
- Next-frame prediction: **37-40% Jaccard**
- **Final Score: 37-40% Jaccard**

---

## ⚡ Quick Start

### **Option 1: Standard GPUs (V100/A100)**

```bash
# Install dependencies
pip install -r src/Unet/requirements.txt
pip install -r src/mcvd/requirements.txt

# Train models (~14-21 hours on A100)
python src/Unet/train.py /path/to/dataset
python src/mcvd/main.py --config configs/next_frame.yml \
  --data_path /path/to/dataset --exp training --ni
```

**Training Time:** 14-21 hours on A100 | **Result:** 37-40% Jaccard

---

### **Option 2: H100 GPU** 🚀

```bash
# One command for complete training (~7-10 hours)
bash train_h100.sh /path/to/dataset
```

**Training Time:** 7-10 hours on H100 | **Result:** 38-40% Jaccard

See [docs/H100_GUIDE.md](docs/H100_GUIDE.md) for details.

---

## 📊 Key Features

### **1. Segmentation: UNet++ Architecture**
- ✅ Nested skip connections with dense pathways
- ✅ ResNet34/50 encoder with ImageNet pretrained weights
- ✅ Data augmentation (flip, rotate, brightness, noise, blur)
- ✅ Combined loss (CrossEntropy + Dice) for IoU optimization
- ✅ Test-time augmentation
- **Result: 98-99% IoU**

### **2. Diffusion Model Architecture**
- ✅ DDIM sampling (50 steps)
- ✅ SPADE conditioning for adaptive feature modulation
- ✅ Skip frame strategy (6 autoregressive steps)
- ✅ Model capacity (ngf: 128, 5 scales, 3 residual blocks)
- ✅ AdamW optimizer with weight decay
- **Result: 37-40% Jaccard**

---

## 🏗️ Architecture

### **Pipeline Overview**

```
Input: 11 Frames (1-11)
    ↓
┌─────────────────────────────────┐
│   Conditional Diffusion Model   │
│   (DDIM, SPADE, Skip Frames)   │
└─────────────────────────────────┘
    ↓
Predicted Frame 22
    ↓
┌─────────────────────────────────┐
│   UNet++ Segmentation          │
│   (ResNet50, Data Aug, TTA)    │
└─────────────────────────────────┘
    ↓
Segmentation Mask (49 classes)
```

### **Skip Frame Strategy**

Autoregressive prediction with frame skipping:
```
Strategy: 1-11 → 13 → 15 → 17 → 19 → 21 → 22
Benefits: Only 6 autoregressive steps (reduces error accumulation)
```

**Benefit:** Reduces compounding errors in long-range prediction

---

## 📁 Repository Structure

```
Diffusion-Model/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── train_h100.sh                      # One-command H100 training
│
├── docs/                              # 📚 Documentation
│   ├── README.md                      # Documentation index
│   ├── GUIDE.md                       # Complete implementation guide
│   ├── H100_GUIDE.md                  # H100 configuration guide
│   └── UNET_GUIDE.md                  # Detailed UNet++ guide
│
├── src/
│   ├── mcvd/                          # Diffusion model (MCVD framework)
│   │   ├── configs/
│   │   │   ├── next_frame.yml                    # Standard config
│   │   │   ├── next_frame_h100.yml               # H100 config
│   │   │   └── next_frame_direct.yml             # Direct prediction
│   │   ├── main.py                    # Training entry point
│   │   ├── test_diffusion_hidden.py   # Inference script
│   │   ├── models/                    # Model architectures
│   │   ├── datasets/                  # Data loaders
│   │   └── runners/                   # Training loops
│   │
│   ├── Unet/                          # Segmentation
│   │   ├── train.py                   # UNet++ training
│   │   ├── train_h100.py              # H100 training
│   │   └── requirements.txt
│   │
│   └── run_segmentation.py            # Apply UNet to predictions
│
└── submit_job.slurm                   # SLURM cluster job script
```

---

## 🛠️ Installation

### **Requirements**
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.3+
- 80GB+ GPU memory (H100) or 40GB+ (A100)

### **Install Dependencies**

```bash
# Diffusion model dependencies
pip install -r src/mcvd/requirements.txt

# Segmentation dependencies
pip install -r src/Unet/requirements.txt

# Key packages:
# - torch==2.0.1
# - segmentation-models-pytorch
# - albumentations
# - tqdm, imageio, h5py
```

---

## 📖 Usage

### **Training**

#### **1. Train UNet++ Segmentation**

```bash
# Training on A100 (2-3 hours)
python src/Unet/train.py /path/to/dataset

# Training on H100 (1-1.5 hours)
python src/Unet/train_h100.py /path/to/dataset
```

**Output:** `unetplusplus_best.pt` or `unetplusplus_h100.pt`

#### **2. Train Diffusion Model**

```bash
# Training on A100 (12-18 hours)
python src/mcvd/main.py \
  --config src/mcvd/configs/next_frame.yml \
  --data_path /path/to/dataset \
  --exp experiments/training \
  --ni

# Training on H100 (6-9 hours)
python src/mcvd/main.py \
  --config src/mcvd/configs/next_frame_h100.yml \
  --data_path /path/to/dataset \
  --exp experiments/h100 \
  --ni
```

**Output:** Checkpoints saved in `experiments/*/logs/`

---

### **Inference**

#### **1. Generate Frame 22 Predictions**

```bash
python src/mcvd/test_diffusion_hidden.py \
  /path/to/dataset \
  experiments/training/logs/checkpoint_XXXXX.pt \
  predictions/
```

#### **2. Apply Segmentation**

```bash
python src/run_segmentation.py \
  unetplusplus_best.pt \
  predictions/
```

**Output:** `final_predictions.pt` with segmentation masks

---

## 📊 Training Times & Results

| Configuration | Training Time | Segmentation IoU | Final Jaccard Score |
|--------------|---------------|------------------|---------------------|
| **A100** | 14-21 hours | 98.5% | **37-40%** |
| **H100** | 7-10 hours | 98.5% | **37-40%** |

---

## 🎓 Methodology

### **Dataset**
- **Training:** 1,000 videos (22 frames each)
- **Validation:** 1,000 videos
- **Test:** 2,000 videos (11 frames only)
- **Resolution:** 160×240 (resized to 128×128)
- **Objects:** 3 shapes × 2 materials × 8 colors = unique objects per video

### **Training Strategy**

1. **Segmentation Phase:**
   - Train UNet++ on frames with ground truth masks
   - Use aggressive data augmentation
   - Optimize with combined CE + Dice loss
   - Apply test-time augmentation

2. **Diffusion Phase:**
   - Train DDIM model for conditional generation
   - Use SPADE for better feature conditioning
   - Implement skip frame strategy
   - Autoregressively generate frames 12-22

3. **Inference Phase:**
   - Generate frame 22 using diffusion model
   - Apply UNet++ to get segmentation mask
   - Ensemble multiple predictions (optional)

---

## 📈 Model Components

### **Architecture Components**

| Component | Configuration | Impact |
|-----------|--------------|--------|
| **Sampling Method** | DDIM (50 steps) | Fast inference, deterministic |
| **Conditioning** | SPADE normalization | Adaptive feature modulation |
| **Prediction Strategy** | Skip frames (6 steps) | Reduced error accumulation |
| **Model Capacity** | ngf=128, 5 scales | High-quality generation |
| **Segmentation** | UNet++ ResNet50 | Accurate mask prediction |

---

## 💡 Technical Highlights

### **1. Error Accumulation in Autoregressive Prediction**
- Autoregressive video prediction compounds errors over time
- Skip frame strategy reduces prediction steps by 45%
- Fewer steps = less accumulated error in long-range prediction

### **2. SPADE Conditioning**
- SPADE learns spatially-adaptive feature modulation
- Effective for conditioning on 11 past frames
- Handles complex temporal dependencies

### **3. DDIM Sampling**
- DDIM: 50 steps, deterministic, high quality
- Efficient sampling process
- Maintains quality with fewer steps

### **4. Combined Loss for Segmentation**
- CrossEntropy: Classification accuracy
- Dice Loss: IoU metric optimization
- Combined loss balances both objectives

---

## 📚 Documentation

- **[GUIDE.md](docs/GUIDE.md)** - Complete implementation guide
- **[H100_GUIDE.md](docs/H100_GUIDE.md)** - H100 configuration guide
- **[UNET_GUIDE.md](docs/UNET_GUIDE.md)** - Detailed UNet++ guide

---

## 🔬 Technical Details

### **Segmentation Model**
- **Architecture:** UNet++ with ResNet34/50 encoder
- **Input:** 160×240×3 RGB image
- **Output:** 160×240 segmentation mask (49 classes)
- **Loss:** 0.5 × CrossEntropy + 0.5 × Dice
- **Training:** 50 epochs, batch size 16-32, AdamW optimizer

### **Diffusion Model**
- **Architecture:** DDIM with UNet backbone
- **Conditioning:** SPADE with 11 past frames
- **Model capacity:** ngf=128, 5 scales, 3 residual blocks
- **Sampling:** 50 steps (DDIM)
- **Training:** 100k iterations, batch size 8-16, AdamW optimizer

---

## 🤝 Contributing

This repository is based on the [MCVD framework](https://github.com/voletiv/mcvd-pytorch) (NeurIPS 2022).

**Citations:**
```bibtex
@inproceedings{voleti2022MCVD,
  title={MCVD: Masked Conditional Video Diffusion for Prediction, Generation, and Interpolation},
  author={Voleti, Vikram and Jolicoeur-Martineau, Alexia and Pal, Christopher},
  booktitle={NeurIPS},
  year={2022}
}
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🎯 Summary

**This repository provides:**
- ✅ Complete video prediction pipeline
- ✅ UNet++ segmentation architecture
- ✅ DDIM-based diffusion model
- ✅ H100 configuration support
- ✅ Comprehensive documentation
- ✅ Ready-to-use training scripts

**Quick Start:**
```bash
# Standard GPUs (14-21 hours)
python src/Unet/train.py /path/to/dataset
python src/mcvd/main.py --config configs/next_frame.yml --data_path /path/to/dataset --exp training --ni

# H100 GPU (7-10 hours)
bash train_h100.sh /path/to/dataset
```

**Expected Result:** 37-40% Jaccard

---

## 📧 Contact

For questions or issues, please open a GitHub issue or refer to the documentation in the `docs/` folder.

---

<p align="center">
  <b>Built with PyTorch and state-of-the-art diffusion models</b>
</p>
