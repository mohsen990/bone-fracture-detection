# run_yolo_training.py
"""
Quick YOLO training script - Run from project root directory
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from src.models.yolov8_detector import YOLOv8FractureDetector

def prepare_yolo_dataset():
    """
    Check if YOLO dataset exists
    """
    # Use absolute path based on script location
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_yaml_path = os.path.join(project_root, 'data', 'Yolo8', 'data.yaml')

    if not os.path.exists(data_yaml_path):
        print("✗ Error: YOLO dataset not found!")
        print(f"  Expected: {data_yaml_path}")
        print("\nPlease ensure your dataset is in YOLO format:")
        print("  data/Yolo8/")
        print("  ├── train/images/")
        print("  ├── train/labels/")
        print("  ├── valid/images/")
        print("  ├── valid/labels/")
        print("  ├── test/images/")
        print("  ├── test/labels/")
        print("  └── data.yaml")
        return None

    print(f"✓ Found YOLO dataset: {data_yaml_path}")
    return data_yaml_path

def main():
    print("=" * 70)
    print("YOLOv8 Fracture Detection Training")
    print("=" * 70)

    # Check dataset
    data_yaml_path = prepare_yolo_dataset()
    if data_yaml_path is None:
        return None, None

    # Initialize detector
    print("\n" + "=" * 70)
    print("Initializing YOLOv8 Detector")
    print("=" * 70)
    # Model sizes: 'n' (nano/fastest), 's' (small), 'm' (medium), 'l' (large)
    detector = YOLOv8FractureDetector(model_size='s')  # 's' for small - good balance
    print("✓ Using YOLOv8s (small) model - faster training")

    # Set dataset path
    detector.prepare_dataset(data_yaml_path)

    # Train
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)
    print("Training parameters:")
    print("  - Epochs: 50")
    print("  - Image size: 416 (faster)")
    print("  - Batch size: 16")
    print("")

    results = detector.train(
        epochs=100,
        img_size=512,
        batch_size=16
    )

    # Validate
    print("\n" + "=" * 70)
    print("Validation")
    print("=" * 70)
    metrics = detector.validate()

    print(f"\n✓ Validation Results:")
    print(f"  mAP@50: {metrics.box.map50:.4f}")
    print(f"  mAP@50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.p[0]:.4f}")
    print(f"  Recall: {metrics.box.r[0]:.4f}")

    # Test prediction
    print("\n" + "=" * 70)
    print("Testing Prediction")
    print("=" * 70)

    project_root = os.path.dirname(os.path.abspath(__file__))
    test_image_dir = os.path.join(project_root, 'data', 'Yolo8', 'test', 'images')

    if os.path.exists(test_image_dir):
        test_images = [f for f in os.listdir(test_image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        if test_images:
            sample_image = os.path.join(test_image_dir, test_images[0])
            results = detector.predict(sample_image, conf_threshold=0.5)
            print(f"✓ Prediction completed on: {test_images[0]}")
            print("  Check 'predictions' folder for results.")

    # Summary
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"✓ Model saved to: runs/detect/fracture_detection/weights/best.pt")
    print(f"✓ Training logs: runs/detect/fracture_detection/")

    return detector, metrics

if __name__ == '__main__':
    result = main()
    if result != (None, None):
        print("\n✓ YOLOv8 training completed successfully!")
