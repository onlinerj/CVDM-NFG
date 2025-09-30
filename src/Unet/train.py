import os
import torch
import torch.nn as nn
import numpy as np
import torchmetrics
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from PIL import Image
from torch.optim import lr_scheduler
import sys
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SegData(Dataset):
    def __init__(self, videos, transform=None):
        self.transforms = transform
        self.images, self.masks = [], []
        for i in videos:
            imgs = os.listdir(i)
            self.images.extend([i + '/' + img for img in imgs if not img.startswith('mask')]) 

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.images[idx]))
        x = self.images[idx].split('/')
        image_name = x[-1]
        mask_idx = int(image_name.split("_")[1].split(".")[0])
        x = x[:-1]
        mask_path = '/'.join(x)
        mask = np.load(mask_path + '/mask.npy')
        mask = mask[mask_idx, :, :]

        if self.transforms is not None:
            augmented = self.transforms(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        
        return img, mask


def get_training_augmentation():
    """
    Aggressive data augmentation for training.
    These augmentations preserve semantic meaning while increasing data diversity.
    """
    return A.Compose([
        # Geometric augmentations
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        
        # Intensity augmentations
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        
        # Weather/lighting effects
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.CLAHE(clip_limit=2.0, p=0.2),
        
        # Normalize and convert to tensor
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_validation_augmentation():
    """
    No augmentation for validation, only normalization.
    """
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_unetplusplus_model(encoder_name="resnet34", encoder_weights="imagenet", out_channels=49):
    """
    Create UNet++ model with specified encoder.
    
    Args:
        encoder_name: Encoder backbone (resnet34, resnet50, efficientnet-b3, etc.)
        encoder_weights: Pretrained weights ('imagenet' or None)
        out_channels: Number of output classes (49 for this dataset)
    
    Returns:
        UNet++ model
    """
    model = smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=out_channels,
        activation=None
    )
    return model


class DiceLoss(nn.Module):
    """
    Dice Loss for better IoU optimization.
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets, num_classes=49):
        inputs = torch.softmax(inputs, dim=1)
        
        # One-hot encode targets
        targets_one_hot = torch.zeros_like(inputs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
        
        # Calculate Dice coefficient
        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()
        
        return dice_loss


class CombinedLoss(nn.Module):
    """
    Combination of Cross Entropy and Dice Loss for optimal IoU.
    """
    def __init__(self, ce_weight=0.5, dice_weight=0.5, class_weights=None):
        super(CombinedLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.dice_loss = DiceLoss()
    
    def forward(self, inputs, targets):
        ce = self.ce_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return self.ce_weight * ce + self.dice_weight * dice


def calculate_batch_iou(outputs: torch.Tensor, labels: torch.Tensor, num_classes=49, smooth=1e-6):
    """
    Calculate mean IoU across all classes.
    """
    ious = []
    outputs = outputs.view(-1)
    labels = labels.view(-1)
    
    for cls in range(num_classes):
        output_cls = (outputs == cls)
        label_cls = (labels == cls)
        
        intersection = (output_cls & label_cls).float().sum()
        union = (output_cls | label_cls).float().sum()
        
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append((intersection + smooth) / (union + smooth))
    
    # Return mean IoU (ignoring classes not present)
    ious = [iou for iou in ious if not np.isnan(iou)]
    return np.mean(ious) if ious else 0.0


def test_time_augmentation(model, image, device):
    """
    Apply test-time augmentation for better predictions.
    Average predictions from original + horizontal flip.
    """
    model.eval()
    with torch.no_grad():
        # Original prediction
        pred1 = torch.softmax(model(image), dim=1)
        
        # Horizontal flip prediction
        image_flipped = torch.flip(image, dims=[3])
        pred2 = torch.softmax(model(image_flipped), dim=1)
        pred2 = torch.flip(pred2, dims=[3])
        
        # Average predictions
        pred_avg = (pred1 + pred2) / 2
        
    return pred_avg


if __name__ == "__main__":
    data_path = sys.argv[1]
    train_data_path = os.path.join(data_path,'train/video_')
    val_data_path = os.path.join(data_path,'val/video_')

    # Data loading with augmentation
    print("Loading datasets with augmentation...")
    train_data_dir = [train_data_path + str(i) for i in range(0, 1000)]
    train_data = SegData(train_data_dir, transform=get_training_augmentation())
    train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)

    val_data_dir = [val_data_path + str(i) for i in range(1000, 2000)]
    val_data = SegData(val_data_dir, transform=get_validation_augmentation())
    val_dataloader = DataLoader(val_data, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model initialization
    print("Initializing UNet++ with ResNet34 encoder...")
    model = get_unetplusplus_model(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        out_channels=49
    ).to(DEVICE)
    
    print(f"Model loaded on {DEVICE}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Training hyperparameters
    epochs = 50  # Increased from 30
    patience = 7  # Increased from 3
    
    # Discriminative learning rates (lower for pretrained encoder)
    optimizer = AdamW([
        {'params': model.encoder.parameters(), 'lr': 1e-5},  # Lower LR for pretrained
        {'params': model.decoder.parameters(), 'lr': 1e-4},  # Higher LR for decoder
        {'params': model.segmentation_head.parameters(), 'lr': 1e-4}
    ], weight_decay=1e-4)
    
    # Combined loss for better IoU
    criterion = CombinedLoss(ce_weight=0.5, dice_weight=0.5)
    
    # Cosine annealing with warm restarts
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler()

    # Training loop
    best_iou = 0.0
    epochs_no_improve = 0
    
    print("\nStarting training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for idx, (data, targets) in enumerate(loop):
            data = data.to(DEVICE)
            targets = targets.to(DEVICE).type(torch.long)
            
            with torch.cuda.amp.autocast():
                predictions = model(data)
                loss = criterion(predictions, targets)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_ious = []
        
        with torch.no_grad():
            for x, y in tqdm(val_dataloader, desc="Validation"):
                x = x.to(DEVICE)
                y = y.to(DEVICE).type(torch.long)
                
                with torch.cuda.amp.autocast():
                    # Use test-time augmentation for validation
                    preds = test_time_augmentation(model, x, DEVICE)
                    loss_v = criterion(model(x), y)  # Loss without TTA
                
                val_loss += loss_v.item()
                preds_arg = torch.argmax(preds, dim=1)
                
                # Calculate IoU for each sample
                for i in range(preds_arg.shape[0]):
                    iou = calculate_batch_iou(preds_arg[i], y[i])
                    all_ious.append(iou)
        
        mean_iou = np.mean(all_ious)
        avg_train_loss = train_loss / len(train_dataloader)
        avg_val_loss = val_loss / len(val_dataloader)
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val IoU: {mean_iou:.4f}")
        
        # Save best model based on IoU
        if mean_iou > best_iou:
            best_iou = mean_iou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'iou': mean_iou,
            }, 'unetplusplus_best.pt')
            print(f"✓ Model saved with IoU: {mean_iou:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs")
        
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            print(f"Best IoU: {best_iou:.4f}")
            break
    
    print(f"\nTraining completed! Best IoU: {best_iou:.4f}")

