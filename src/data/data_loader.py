# src/data/data_loader.py
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

class FractureDataset(Dataset):
    """Custom dataset for fracture detection"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Assume folder structure: data_dir/class_name/images
        class_names = ['Normal', 'Fractured']
        for class_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(data_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        self.images.append(os.path.join(class_dir, img_name))
                        self.labels.append(class_idx)
        
        if len(self.images) == 0:
            raise ValueError(f"No images found in {data_dir}. "
                           f"Make sure you have 'Normal' and 'Fractured' folders with images.")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = np.array(Image.open(img_path).convert('RGB'))
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, label

def create_data_loaders(train_dir, val_dir, test_dir, batch_size=32, img_size=224, num_workers=4):
    """Create PyTorch data loaders"""

    # Import here to avoid circular imports
    from src.data.preprocessing import get_train_transforms, get_val_transforms
    
    # Create datasets
    train_dataset = FractureDataset(train_dir, transform=get_train_transforms(img_size))
    val_dataset = FractureDataset(val_dir, transform=get_val_transforms(img_size))
    test_dataset = FractureDataset(test_dir, transform=get_val_transforms(img_size))
    
    print(f"Dataset sizes:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val:   {len(val_dataset)}")
    print(f"  Test:  {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader