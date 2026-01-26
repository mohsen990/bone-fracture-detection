# evaluate_yolo.py
"""
Evaluate YOLOv8 model and display metrics
"""
from ultralytics import YOLO
import os

def evaluate_yolo_model():
    project_root = os.path.dirname(os.path.abspath(__file__))

    # Find the best model
    model_paths = [
        os.path.join(project_root, 'runs', 'detect', 'fracture_detection_improved', 'weights', 'best.pt'),
        os.path.join(project_root, 'runs', 'detect', 'runs', 'detect', 'fracture_detection_improved', 'weights', 'best.pt'),
        os.path.join(project_root, 'runs', 'detect', 'fracture_detection', 'weights', 'best.pt'),
        os.path.join(project_root, 'runs', 'detect', 'runs', 'detect', 'fracture_detection', 'weights', 'best.pt'),
    ]

    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break

    if model_path is None:
        print("No trained YOLO model found!")
        return

    print("=" * 70)
    print("YOLOv8 Model Evaluation")
    print("=" * 70)
    print(f"\nModel: {model_path}\n")

    # Load model
    model = YOLO(model_path)

    # Get data.yaml path
    data_yaml = os.path.join(project_root, 'data', 'Yolo8', 'data.yaml')

    # Run validation
    print("Running validation...")
    print("-" * 70)

    metrics = model.val(data=data_yaml)

    # Display results
    print("\n" + "=" * 70)
    print("YOLO DETECTION METRICS")
    print("=" * 70)

    print(f"\n{'Metric':<25} {'Value':<15} {'Description'}")
    print("-" * 70)
    print(f"{'mAP@50':<25} {metrics.box.map50:.4f}{'':>10} Mean Average Precision at IoU=0.50")
    print(f"{'mAP@50-95':<25} {metrics.box.map:.4f}{'':>10} Mean AP across IoU 0.50-0.95")

    if hasattr(metrics.box, 'p') and len(metrics.box.p) > 0:
        print(f"{'Mean Precision':<25} {sum(metrics.box.p)/len(metrics.box.p):.4f}{'':>10} Average precision across classes")
        print(f"{'Mean Recall':<25} {sum(metrics.box.r)/len(metrics.box.r):.4f}{'':>10} Average recall across classes")

    # Per-class metrics
    print("\n" + "=" * 70)
    print("PER-CLASS PERFORMANCE")
    print("=" * 70)

    class_names = model.names
    print(f"\n{'Class':<20} {'Precision':<12} {'Recall':<12} {'mAP50':<12}")
    print("-" * 70)

    if hasattr(metrics.box, 'p') and hasattr(metrics.box, 'r'):
        for i, (p, r) in enumerate(zip(metrics.box.p, metrics.box.r)):
            class_name = class_names.get(i, f'Class {i}')
            # Get per-class AP if available
            ap50 = metrics.box.ap50[i] if hasattr(metrics.box, 'ap50') and len(metrics.box.ap50) > i else 0
            print(f"{class_name:<20} {p:.4f}{'':>6} {r:.4f}{'':>6} {ap50:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    map50 = metrics.box.map50
    if map50 >= 0.5:
        status = "Good"
    elif map50 >= 0.3:
        status = "Moderate"
    else:
        status = "Needs Improvement"

    print(f"\nOverall Performance: {status}")
    print(f"mAP@50: {map50*100:.2f}%")

    if map50 < 0.3:
        print("\nRecommendations:")
        print("  - Dataset may have annotation quality issues")
        print("  - Consider using Classification mode for better accuracy")
        print("  - Lower confidence threshold in app (0.10-0.15)")

    return metrics


if __name__ == '__main__':
    evaluate_yolo_model()
