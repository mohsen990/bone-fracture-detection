import cv2
import numpy as np
from albumentations import (
    Compose, Resize, Normalize, HorizontalFlip, 
    Rotate, RandomBrightnessContrast, GaussNoise,
    ShiftScaleRotate, CLAHE
)
from albumentations.pytorch import ToTensorV2

def get_train_transforms(img_size=224):
    """Training augmentation pipeline"""
    return Compose([
        Resize(img_size, img_size),
        CLAHE(clip_limit=2.0, p=0.5),  # Enhance contrast
        Rotate(limit=20, p=0.5),
        HorizontalFlip(p=0.3),  # Be careful with medical images
        ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=15,
            p=0.5
        ),
        RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

def get_val_transforms(img_size=224):
    """Validation/Test transforms"""
    return Compose([
        Resize(img_size, img_size),
        CLAHE(clip_limit=2.0, p=1.0),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

def preprocess_xray(image_path, apply_clahe=True):
    """Preprocess single X-ray image"""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if apply_clahe:
        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # Merge and convert back
        lab = cv2.merge([l, a, b])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    return img