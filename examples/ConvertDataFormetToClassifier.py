# organize_data.py
import os
import shutil
from pathlib import Path
from tqdm import tqdm
import cv2

def read_yolo_label(label_path):
    """
    Read YOLO format label file
    Returns: True if fracture detected, False if normal
    """
    if not os.path.exists(label_path):
        return False  # No label = Normal
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    # If file has any annotations, it means fracture detected
    return len(lines) > 0

def organize_yolo_to_classification(source_base_dir='data', create_validation=True):
    """
    Organize YOLO format data into classification format
    
    Structure expected:
    data/
        train/
            images/
            labels/
        valid/
            images/
            labels/
        test/
            images/
            labels/
    
    Will create:
    data/
        train_classification/
            Normal/
            Fractured/
        valid_classification/
            Normal/
            Fractured/
        test_classification/
            Normal/
            Fractured/
    """
    
    source_base = Path(source_base_dir)
    splits = ['train', 'valid', 'test']
    
    stats = {
        'train': {'normal': 0, 'fractured': 0},
        'valid': {'normal': 0, 'fractured': 0},
        'test': {'normal': 0, 'fractured': 0}
    }
    
    print("="*70)
    print("Organizing Data from YOLO to Classification Format")
    print("="*70)
    
    for split in splits:
        print(f"\n📁 Processing {split} set...")
        
        # Source paths
        images_dir = source_base / split / 'images'
        labels_dir = source_base / split / 'labels'
        
        # Destination paths
        dest_normal = source_base / f'{split}_classification' / 'Normal'
        dest_fractured = source_base / f'{split}_classification' / 'Fractured'
        
        # Create destination directories
        dest_normal.mkdir(parents=True, exist_ok=True)
        dest_fractured.mkdir(parents=True, exist_ok=True)
        
        if not images_dir.exists():
            print(f"  ⚠️  {images_dir} not found, skipping...")
            continue
        
        # Get all images
        image_files = list(images_dir.glob('*.*'))
        print(f"  Found {len(image_files)} images")
        
        for image_path in tqdm(image_files, desc=f"  Organizing {split}"):
            # Get corresponding label file
            label_path = labels_dir / (image_path.stem + '.txt')
            
            # Check if fracture exists
            has_fracture = read_yolo_label(label_path)
            
            # Copy to appropriate folder
            if has_fracture:
                dest_path = dest_fractured / image_path.name
                shutil.copy2(image_path, dest_path)
                stats[split]['fractured'] += 1
            else:
                dest_path = dest_normal / image_path.name
                shutil.copy2(image_path, dest_path)
                stats[split]['normal'] += 1
    
    # Print statistics
    print("\n" + "="*70)
    print("Organization Complete!")
    print("="*70)
    
    for split in splits:
        total = stats[split]['normal'] + stats[split]['fractured']
        if total > 0:
            print(f"\n{split.upper()} Set:")
            print(f"  Normal:    {stats[split]['normal']:5d} ({stats[split]['normal']/total*100:.1f}%)")
            print(f"  Fractured: {stats[split]['fractured']:5d} ({stats[split]['fractured']/total*100:.1f}%)")
            print(f"  Total:     {total:5d}")
    
    return stats

def verify_organization(base_dir='data'):
    """Verify the organization was successful"""
    print("\n" + "="*70)
    print("Verification")
    print("="*70)
    
    base = Path(base_dir)
    splits = ['train_classification', 'valid_classification', 'test_classification']
    
    for split in splits:
        split_dir = base / split
        if not split_dir.exists():
            print(f"\n{split}: NOT FOUND")
            continue
        
        normal_dir = split_dir / 'Normal'
        fractured_dir = split_dir / 'Fractured'
        
        normal_count = len(list(normal_dir.glob('*.*'))) if normal_dir.exists() else 0
        fractured_count = len(list(fractured_dir.glob('*.*'))) if fractured_dir.exists() else 0
        
        print(f"\n{split}:")
        print(f"  ✓ Normal: {normal_count} images")
        print(f"  ✓ Fractured: {fractured_count} images")

def main():
    # Run organization
    stats = organize_yolo_to_classification(source_base_dir='data')
    
    # Verify
    verify_organization(base_dir='data')
    
    print("\n✅ Done! You can now use the data for classification training.")
    print("\nNext steps:")
    print("  1. Run: python test_dataloader.py")
    print("  2. Run: python run_single_training.py")

if __name__ == '__main__':
    main()