# convert_labels.py
"""
Convert YOLO segmentation labels (polygon) to detection labels (bounding box)
"""
import os
from pathlib import Path

def convert_polygon_to_bbox(label_line):
    """
    Convert polygon format to bounding box format

    Polygon: class_id x1 y1 x2 y2 x3 y3 ...
    BBox:    class_id x_center y_center width height
    """
    parts = label_line.strip().split()
    if len(parts) < 5:
        return None

    class_id = parts[0]
    coords = [float(x) for x in parts[1:]]

    # Extract x and y coordinates
    x_coords = coords[0::2]  # Every other starting from 0
    y_coords = coords[1::2]  # Every other starting from 1

    if len(x_coords) < 2 or len(y_coords) < 2:
        return None

    # Calculate bounding box
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)

    # Convert to YOLO format (center x, center y, width, height)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    # Ensure values are in valid range [0, 1]
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0.001, min(1, width))
    height = max(0.001, min(1, height))

    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def convert_label_file(input_path, output_path):
    """Convert a single label file"""
    with open(input_path, 'r') as f:
        lines = f.readlines()

    converted_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) == 5:
            # Already in bbox format
            converted_lines.append(line)
        elif len(parts) > 5:
            # Polygon format - convert
            converted = convert_polygon_to_bbox(line)
            if converted:
                converted_lines.append(converted)

    with open(output_path, 'w') as f:
        f.write('\n'.join(converted_lines))

    return len(converted_lines)


def convert_dataset(data_dir):
    """Convert all label files in the dataset"""
    data_path = Path(data_dir)

    # Process each split
    for split in ['train', 'valid', 'test']:
        labels_dir = data_path / split / 'labels'

        if not labels_dir.exists():
            print(f"⚠️ {split}/labels not found, skipping...")
            continue

        print(f"\n📁 Converting {split} labels...")

        label_files = list(labels_dir.glob('*.txt'))
        converted = 0

        for label_file in label_files:
            count = convert_label_file(label_file, label_file)  # Overwrite in place
            if count > 0:
                converted += 1

        print(f"   ✓ Converted {converted}/{len(label_files)} files")


if __name__ == '__main__':
    import sys

    # Default path
    data_dir = 'data/Yolo8'

    if len(sys.argv) > 1:
        data_dir = sys.argv[1]

    print("=" * 60)
    print("YOLO Label Converter: Polygon → Bounding Box")
    print("=" * 60)
    print(f"\nDataset path: {data_dir}")

    convert_dataset(data_dir)

    print("\n" + "=" * 60)
    print("✓ Conversion complete!")
    print("=" * 60)
    print("\nNow retrain YOLO:")
    print("  python run_yolo_training.py")
