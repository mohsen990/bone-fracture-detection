# verify_data.py
import os
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import numpy as np

def verify_data_structure(base_dir='data'):
    """Verify the data is properly organized"""
    
    print("="*70)
    print("Data Verification")
    print("="*70)
    
    base = Path(base_dir)
    splits = ['train_classification', 'valid_classification', 'test_classification']
    classes = ['Normal', 'Fractured']
    
    total_stats = {}
    
    for split in splits:
        split_dir = base / split
        
        if not split_dir.exists():
            print(f"\n❌ {split}: NOT FOUND")
            continue
        
        print(f"\n✅ {split}:")
        split_stats = {}
        
        for class_name in classes:
            class_dir = split_dir / class_name
            
            if class_dir.exists():
                images = list(class_dir.glob('*.jpg')) + \
                        list(class_dir.glob('*.png')) + \
                        list(class_dir.glob('*.jpeg'))
                count = len(images)
                split_stats[class_name] = count
                print(f"   {class_name:10s}: {count:5d} images")
            else:
                print(f"   {class_name:10s}: NOT FOUND")
                split_stats[class_name] = 0
        
        total = sum(split_stats.values())
        if total > 0:
            normal_pct = split_stats['Normal'] / total * 100
            fractured_pct = split_stats['Fractured'] / total * 100
            print(f"   {'Total':10s}: {total:5d} images")
            print(f"   Balance: Normal {normal_pct:.1f}% | Fractured {fractured_pct:.1f}%")
        
        total_stats[split] = split_stats
    
    return total_stats

def visualize_sample_images(base_dir='data', split='train_classification', samples_per_class=4):
    """Visualize sample images from each class"""
    
    print(f"\n{'='*70}")
    print(f"Visualizing samples from {split}")
    print("="*70)
    
    base = Path(base_dir)
    split_dir = base / split
    
    fig, axes = plt.subplots(2, samples_per_class, figsize=(16, 8))
    
    for row, class_name in enumerate(['Normal', 'Fractured']):
        class_dir = split_dir / class_name
        
        if not class_dir.exists():
            print(f"❌ {class_name} directory not found")
            continue
        
        images = list(class_dir.glob('*.jpg')) + \
                list(class_dir.glob('*.png')) + \
                list(class_dir.glob('*.jpeg'))
        
        # Get random samples
        import random
        samples = random.sample(images, min(samples_per_class, len(images)))
        
        for col, img_path in enumerate(samples):
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            axes[row, col].imshow(img)
            axes[row, col].set_title(f'{class_name}\n{img_path.name}', fontsize=9)
            axes[row, col].axis('off')
    
    plt.suptitle(f'Sample Images from {split}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('data_verification_samples.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ Saved visualization to 'data_verification_samples.png'")

def check_data_balance(stats):
    """Check if data is balanced"""
    
    print(f"\n{'='*70}")
    print("Data Balance Analysis")
    print("="*70)
    
    for split, class_stats in stats.items():
        total = sum(class_stats.values())
        if total == 0:
            continue
        
        normal_pct = class_stats.get('Normal', 0) / total * 100
        fractured_pct = class_stats.get('Fractured', 0) / total * 100
        
        print(f"\n{split}:")
        
        # Check balance
        ratio = max(normal_pct, fractured_pct) / min(normal_pct, fractured_pct)
        
        if ratio < 1.5:
            print("   ✅ Well balanced")
        elif ratio < 3:
            print("   ⚠️  Slightly imbalanced - consider data augmentation")
        else:
            print("   ❌ Highly imbalanced - use weighted loss or oversampling")
        
        print(f"   Imbalance ratio: {ratio:.2f}:1")

def main():
    # Verify structure
    stats = verify_data_structure()
    
    # Check balance
    check_data_balance(stats)
    
    # Visualize samples
    try:
        visualize_sample_images()
    except Exception as e:
        print(f"\n⚠️  Could not visualize samples: {e}")
    
    # Summary
    print(f"\n{'='*70}")
    print("✅ Verification Complete!")
    print("="*70)
    
    total_images = sum(sum(class_stats.values()) for class_stats in stats.values())
    print(f"\nTotal images: {total_images}")
    
    if total_images > 0:
        print("\n🎯 Ready for next steps!")
    else:
        print("\n❌ No images found. Please organize your data first.")

if __name__ == '__main__':
    main()