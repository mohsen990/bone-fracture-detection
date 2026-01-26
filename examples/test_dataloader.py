# test_dataloader.py
import sys
import os
sys.path.append('src')

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from src.utils.config import Config
from src.data.data_loader import create_data_loaders

def test_data_loading():
    """Test if data loaders work correctly"""
    
    print("="*70)
    print("Testing Data Loaders")
    print("="*70)
    
    config = Config()
    
    # Check if directories exist
    for dir_name, dir_path in [('Train', config.TRAIN_DIR), 
                                ('Validation', config.VAL_DIR), 
                                ('Test', config.TEST_DIR)]:
        if Path(dir_path).exists():
            print(f"✅ {dir_name}: {dir_path}")
        else:
            print(f"❌ {dir_name}: {dir_path} NOT FOUND")
            return False
    
    # Create data loaders
    print("\nCreating data loaders...")
    try:
        train_loader, val_loader, test_loader = create_data_loaders(
            config.TRAIN_DIR,
            config.VAL_DIR,
            config.TEST_DIR,
            batch_size=8,
            img_size=config.IMG_SIZE
        )
        
        print(f"✅ Data loaders created successfully!")
        print(f"\n   Train batches: {len(train_loader)}")
        print(f"   Train samples: {len(train_loader.dataset)}")
        print(f"   Val batches:   {len(val_loader)}")
        print(f"   Val samples:   {len(val_loader.dataset)}")
        print(f"   Test batches:  {len(test_loader)}")
        print(f"   Test samples:  {len(test_loader.dataset)}")
        
    except Exception as e:
        print(f"❌ Error creating data loaders: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test loading a batch
    print("\nTesting batch loading...")
    try:
        images, labels = next(iter(train_loader))
        print(f"✅ Successfully loaded batch")
        print(f"   Batch shape: {images.shape}")
        print(f"   Labels shape: {labels.shape}")
        print(f"   Labels: {labels.numpy()}")
        print(f"   Unique labels: {torch.unique(labels).numpy()}")
        
        # Visualize batch
        visualize_batch(images, labels, config.CLASS_NAMES)
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading batch: {e}")
        import traceback
        traceback.print_exc()
        return False

def visualize_batch(images, labels, class_names):
    """Visualize a batch of images"""
    
    print("\nCreating visualization...")
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx in range(min(8, len(images))):
        # Denormalize image
        img = images[idx].permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        axes[idx].imshow(img)
        axes[idx].set_title(f'Label: {class_names[labels[idx]]}', 
                           fontsize=12, fontweight='bold')
        axes[idx].axis('off')
    
    plt.suptitle('Sample Training Batch', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('test_batch_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ Saved visualization to 'test_batch_visualization.png'")

if __name__ == '__main__':
    success = test_data_loading()
    
    if success:
        print("\n" + "="*70)
        print("🎉 All tests passed! Ready to start training!")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("❌ Tests failed. Please fix the issues above.")
        print("="*70)