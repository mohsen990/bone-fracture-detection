# run_yolo_training_improved.py
"""
Improved YOLO training script for better fracture detection
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from ultralytics import YOLO

def main():
    print("=" * 70)
    print("YOLOv8 Improved Fracture Detection Training")
    print("=" * 70)

    # Use absolute path
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_yaml_path = os.path.join(project_root, 'data', 'Yolo8', 'data.yaml')

    if not os.path.exists(data_yaml_path):
        print(f"Error: Dataset not found at {data_yaml_path}")
        return

    # Use MEDIUM model for better accuracy (vs small)
    # Options: 'yolov8n.pt' (nano), 'yolov8s.pt' (small), 'yolov8m.pt' (medium), 'yolov8l.pt' (large)
    print("\nLoading YOLOv8-medium model...")
    model = YOLO('yolov8m.pt')

    print("\nTraining with improved settings...")
    print("  - Model: YOLOv8-medium (better accuracy)")
    print("  - Epochs: 100 (more training)")
    print("  - Image size: 480 (higher resolution)")
    print("  - Batch size: 8 (adjust if out of memory)")
    print("")

    # Train with improved hyperparameters
    results = model.train(
        data=data_yaml_path,
        epochs=100,              # Increased from 30
        imgsz=480,               # Good balance: 480 is multiple of 32
        batch=8,                 # Reduced for memory
        patience=20,             # Early stopping patience
        save=True,
        device=0,                # Use GPU
        workers=4,
        project='runs/detect',
        name='fracture_detection_improved',
        exist_ok=True,
        pretrained=True,
        optimizer='AdamW',       # Better optimizer
        lr0=0.001,               # Initial learning rate
        lrf=0.01,                # Final learning rate factor
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,         # Warmup epochs
        warmup_momentum=0.8,
        box=7.5,                 # Box loss gain
        cls=0.5,                 # Class loss gain
        dfl=1.5,                 # DFL loss gain
        # Data augmentation
        hsv_h=0.015,             # HSV-Hue augmentation
        hsv_s=0.7,               # HSV-Saturation augmentation
        hsv_v=0.4,               # HSV-Value augmentation
        degrees=15,              # Rotation degrees
        translate=0.1,           # Translation
        scale=0.5,               # Scale
        shear=5,                 # Shear degrees
        perspective=0.0,         # Perspective
        flipud=0.5,              # Flip up-down probability
        fliplr=0.5,              # Flip left-right probability
        mosaic=1.0,              # Mosaic augmentation
        mixup=0.1,               # Mixup augmentation
        copy_paste=0.0,          # Copy-paste augmentation
    )

    # Validate
    print("\n" + "=" * 70)
    print("Validation")
    print("=" * 70)

    metrics = model.val()

    print(f"\nValidation Results:")
    print(f"  mAP@50: {metrics.box.map50:.4f}")
    print(f"  mAP@50-95: {metrics.box.map:.4f}")

    if hasattr(metrics.box, 'p') and len(metrics.box.p) > 0:
        print(f"  Precision: {metrics.box.p[0]:.4f}")
        print(f"  Recall: {metrics.box.r[0]:.4f}")

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Model saved to: runs/detect/fracture_detection_improved/weights/best.pt")

    return model, metrics


if __name__ == '__main__':
    main()
