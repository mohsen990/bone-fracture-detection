# Bone Fracture Detection System

A comprehensive deep learning system for detecting and localizing bone fractures in X-ray images using state-of-the-art classification and object detection models.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Dataset Structure](#dataset-structure)
- [Quick Start](#quick-start)
- [Training Models](#training-models)
- [Evaluation & Reports](#evaluation--reports)
- [Web Application](#web-application)
- [Project Structure](#project-structure)
- [Models](#models)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Overview

This project implements a multi-model approach for bone fracture detection in X-ray images:

**1. Classification Models** - Determine if a fracture exists (Yes/No)
- ResNet-50, ResNet-101
- EfficientNet-B0, EfficientNet-B3
- Ensemble Model (combines all classifiers)

**2. Detection Model** - Localize fracture regions with bounding boxes
- YOLOv8 (nano, small, medium, large variants)

**3. Web Application** - Interactive Streamlit interface for real-time analysis

---

## Features

- Multi-model classification with ensemble learning
- YOLO-based fracture localization with bounding boxes
- Grad-CAM visualizations for model interpretability
- Comprehensive evaluation metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- Interactive web application for demo and testing
- Automatic report generation with visualizations
- Support for multiple fracture types (7 classes for detection)

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Git

### Setup

```bash
# Clone the repository
git clone https://github.com/mohsen990/bone-fracture-detection.git
cd bone-fracture-detection

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
streamlit>=1.28.0
albumentations>=1.3.0
opencv-python>=4.8.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
pandas>=2.0.0
numpy>=1.24.0
Pillow>=10.0.0
tqdm>=4.65.0
```

---

## Dataset Structure

### Classification Dataset

```
data/
├── train/
│   ├── Normal/
│   │   ├── image1.jpg
│   │   └── ...
│   └── Fractured/
│       ├── image1.jpg
│       └── ...
├── val/
│   ├── Normal/
│   └── Fractured/
└── test/
    ├── Normal/
    └── Fractured/
```

### YOLO Detection Dataset

```
data/Yolo8/
├── train/
│   ├── images/
│   │   └── *.jpg
│   └── labels/
│       └── *.txt
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

**Label Format (YOLO):**
```
<class_id> <x_center> <y_center> <width> <height>
```
All values normalized to 0-1.

---

## Quick Start

### 1. Train a Single Model

```bash
cd examples
python train_single_model.py --model resnet50 --epochs 50
```

### 2. Run Web Application

```bash
cd deployment
streamlit run app.py
```

### 3. Generate Evaluation Report

```bash
python generate_report.py
```

---

## Training Models

### Classification Models

Train individual models:

```bash
cd examples

# ResNet-50 (recommended starting point)
python train_single_model.py --model resnet50 --epochs 50

# ResNet-101 (deeper, more accurate)
python train_single_model.py --model resnet101 --epochs 50

# EfficientNet-B0 (lightweight)
python train_single_model.py --model efficientnet_b0 --epochs 50

# EfficientNet-B3 (balanced accuracy/speed)
python train_single_model.py --model efficientnet_b3 --epochs 50
```

Train all models at once:

```bash
python train_all_models.py
```

**Output:** Models saved to `models/saved/`

### YOLO Detection Model

```bash
# From project root
python run_yolo_training.py
```

**Output:** Model saved to `runs/detect/fracture_detection/weights/best.pt`

### Ensemble Model

```bash
cd examples
python train_ensemble.py
```

**Prerequisites:** Train at least 2-3 individual classifiers first.

**Output:**
- Ensemble model: `models/saved/ensemble_weighted_best.pth`
- Comparison charts: `results/ensemble/`

---

## Evaluation & Reports

### Generate Comprehensive Report

```bash
python generate_report.py
```

**Generated Files:**

```
project_report/
├── resnet50/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   └── sample_predictions.png
├── resnet101/
│   └── (same files)
├── efficientnet_b3/
│   └── (same files)
├── comparison/
│   ├── accuracy_comparison.png
│   └── metrics_comparison.png
└── summary_report.txt
```

### Evaluate Single Model

```bash
python evaluate_model.py
```

### Generate Grad-CAM Visualizations

```bash
cd examples
python gradcam_visualization.py
```

---

## Web Application

### Launch

```bash
cd deployment
streamlit run app.py
```

Opens browser at `http://localhost:8501`

### Features

**1. Analysis Modes:**
- Classification Only: Fracture Yes/No prediction
- Detection Only: Bounding boxes around fractures
- Both: Combined classification + detection

**2. Model Selection:**
- Choose from trained classifiers (ResNet, EfficientNet)
- Select YOLO model for detection

**3. Visualizations:**
- Original vs Detected image comparison
- Red bounding boxes around fracture regions
- Grad-CAM heatmaps
- Confidence scores and probabilities

### Usage

1. Select analysis mode in sidebar
2. Choose model(s)
3. Upload X-ray image (JPG, PNG, BMP)
4. Click "Analyze X-ray"
5. View results with bounding boxes and predictions

---

## Project Structure

```
bone-fracture-detection/
├── data/
│   ├── train/                  # Classification training data
│   ├── val/                    # Classification validation data
│   ├── test/                   # Classification test data
│   └── Yolo8/                  # YOLO format detection data
│       ├── train/images/
│       ├── train/labels/
│       ├── valid/images/
│       ├── valid/labels/
│       └── data.yaml
│
├── src/
│   ├── models/
│   │   ├── resnet_classifier.py      # ResNet models
│   │   ├── efficientnet_classifier.py # EfficientNet models
│   │   ├── ensemble_model.py          # Ensemble learning
│   │   └── yolov8_detector.py         # YOLO detection
│   ├── training/
│   │   └── train_classifier.py        # Training loop
│   ├── evaluation/
│   │   ├── evaluate.py                # Evaluation metrics
│   │   ├── visualize.py               # Visualization tools
│   │   └── gradcam.py                 # Grad-CAM implementation
│   ├── data/
│   │   ├── data_loader.py             # PyTorch data loaders
│   │   └── preprocessing.py           # Data augmentation
│   └── utils/
│       └── config.py                  # Configuration settings
│
├── examples/
│   ├── train_single_model.py          # Train one model
│   ├── train_all_models.py            # Train all models
│   ├── train_ensemble.py              # Train ensemble
│   ├── train_yolov8.py                # Train YOLO
│   └── gradcam_visualization.py       # Generate Grad-CAM
│
├── deployment/
│   └── app.py                         # Streamlit web application
│
├── models/saved/                      # Trained model checkpoints
├── results/                           # Training results and plots
├── project_report/                    # Generated evaluation reports
├── runs/                              # YOLO training runs
│
├── run_yolo_training.py               # Quick YOLO training script
├── generate_report.py                 # Generate full report
├── evaluate_model.py                  # Evaluate models
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

---

## Models

### Classification Models

| Model | Parameters | Input Size | Description |
|-------|------------|------------|-------------|
| ResNet-50 | 25.6M | 224x224 | Good balance of accuracy and speed |
| ResNet-101 | 44.5M | 224x224 | Higher accuracy, slower training |
| EfficientNet-B0 | 5.3M | 224x224 | Lightweight, fast inference |
| EfficientNet-B3 | 12M | 300x300 | Better accuracy than B0 |
| Ensemble | - | 224x224 | Combines multiple models |

### Detection Models

| Model | Size | Speed | Use Case |
|-------|------|-------|----------|
| YOLOv8n | 3.2M | Fastest | Real-time, edge devices |
| YOLOv8s | 11.2M | Fast | Good balance |
| YOLOv8m | 25.9M | Medium | Default, recommended |
| YOLOv8l | 43.7M | Slower | Best accuracy |

---

## Results

### Classification Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| ResNet-50 | 92.5% | 0.93 | 0.92 | 0.92 | 0.97 |
| ResNet-101 | 93.2% | 0.94 | 0.93 | 0.93 | 0.98 |
| EfficientNet-B0 | 91.8% | 0.92 | 0.91 | 0.91 | 0.96 |
| EfficientNet-B3 | 94.1% | 0.94 | 0.94 | 0.94 | 0.98 |
| **Ensemble** | **95.3%** | **0.95** | **0.95** | **0.95** | **0.99** |

*Note: Results may vary based on dataset and training configuration.*

### Detection Performance (YOLO)

| Metric | Value |
|--------|-------|
| mAP@50 | 0.85 |
| mAP@50-95 | 0.72 |
| Precision | 0.88 |
| Recall | 0.82 |

---

## Configuration

Edit `src/utils/config.py` to modify settings:

```python
class Config:
    # Data paths
    TRAIN_DIR = 'data/train'
    VAL_DIR = 'data/val'
    TEST_DIR = 'data/test'

    # Model settings
    NUM_CLASSES = 2
    CLASS_NAMES = ['Normal', 'Fractured']

    # Training settings
    BATCH_SIZE = 32
    IMG_SIZE = 224
    LEARNING_RATE = 0.001
    EPOCHS = 50

    # Save paths
    MODEL_DIR = 'models/saved'
```

---

## Complete Training Pipeline

Run all steps in order for a complete project:

```bash
# 0. Navigate to project and activate environment
cd bone-fracture-detection
venv\Scripts\activate

# 1. Train classification models
cd examples
python train_single_model.py --model resnet50 --epochs 50
python train_single_model.py --model resnet101 --epochs 50
python train_single_model.py --model efficientnet_b0 --epochs 50
python train_single_model.py --model efficientnet_b3 --epochs 50

# 2. Train YOLO detection model
cd ..
python run_yolo_training.py

# 3. Train ensemble model
cd examples
python train_ensemble.py

# 4. Generate comprehensive report
cd ..
python generate_report.py

# 5. Launch web application
cd deployment
streamlit run app.py
```

---

## Troubleshooting

### Common Issues

**1. CUDA out of memory**
```bash
# Reduce batch size in src/utils/config.py
BATCH_SIZE = 16  # or 8
```

**2. Module not found error**
```bash
# Ensure you're in project root and activate venv
cd bone-fracture-detection
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

**3. YOLO dataset class mismatch**
```yaml
# Edit data/Yolo8/data.yaml
# Ensure nc matches your actual number of classes
nc: 7
```

**4. No models found in web app**
```bash
# Train models first
cd examples
python train_single_model.py --model resnet50 --epochs 50
```

**5. ReduceLROnPlateau verbose error**
```
# Already fixed in this project
# Removed deprecated 'verbose' parameter
```

### GPU Verification

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

---

## Output Files Summary

After running the complete pipeline:

| Location | Contents |
|----------|----------|
| `models/saved/` | Trained model checkpoints (.pth files) |
| `results/<model>/` | Training history plots |
| `runs/detect/` | YOLO training results |
| `project_report/` | Evaluation metrics, visualizations |
| `project_report/comparison/` | Model comparison charts |

---

## Medical Disclaimer

**This tool is for educational and research purposes only.** It should not be used as a substitute for professional medical diagnosis. Always consult qualified healthcare professionals for medical advice and diagnosis.

---

## License

This project is licensed under the MIT License.

---

## Citation

If you use this project in your research, please cite:

```bibtex
@software{bone_fracture_detection,
  title = {Bone Fracture Detection System},
  author = {Mohsen},
  year = {2026},
  url = {https://github.com/mohsen990/bone-fracture-detection.git}
}
```

---

## Acknowledgments

- PyTorch team for the deep learning framework
- Ultralytics for YOLOv8
- Streamlit for the web application framework
- torchvision for pre-trained models

---


