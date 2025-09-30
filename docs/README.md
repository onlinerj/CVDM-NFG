# Documentation

This folder contains comprehensive guides for the model implementation.

## 📚 Guides

### **[GUIDE.md](GUIDE.md)** - START HERE!
Complete guide covering both segmentation and diffusion architecture.
- Architecture details
- **Quick start commands** included
- **Priority recommendations**

### **[UNET_GUIDE.md](UNET_GUIDE.md)** 
Detailed UNet++ implementation instructions.
- Architecture details
- Training configurations
- Troubleshooting

### **[H100_GUIDE.md](H100_GUIDE.md)** 
H100 GPU configuration guide.
- H100 capabilities
- Configuration settings
- Performance benchmarks

---

## 🎯 Quick Reference

### Segmentation (UNet++)
```bash
# Train UNet
python src/Unet/train.py /path/to/dataset
```
**Expected: 98-99% IoU**

### Diffusion Model
```bash
# Train with config
python src/mcvd/main.py \
  --config configs/next_frame.yml \
  --data_path /path/to/dataset \
  --exp training \
  --ni
```
**Expected: 37-40% Jaccard**

---

## 📊 Results Summary

| Component | Performance |
|-----------|-------------|
| Segmentation | 98-99% IoU |
| Diffusion | 37-40% Jaccard |
| **Final Score** | **37-40% Jaccard** |

---

For detailed instructions, see [GUIDE.md](GUIDE.md)
