# Bone Fracture Detection Codebase Guide

## Architecture Overview

**Binary Classification System** for detecting bone fractures in X-ray images using multiple deep learning architectures. The codebase follows a modular pattern:

- **`src/data/`** - Data pipeline: custom PyTorch Dataset (`FractureDataset`), data loaders, and augmentation chains using Albumentations
- **`src/models/`** - Model factories: ResNet (resnet50/101), EfficientNet (B0-B7), Ensemble voting/averaging, and detection models (YOLOv8, RetinaNet)
- **`src/training/`** - `Trainer` class with standardized training loop, loss tracking, checkpoint saving, and learning rate scheduling
- **`src/evaluation/`** - Model evaluation metrics (sklearn-based), Grad-CAM visualization, and result plotting
- **`examples/`** - Runnable training scripts for single models, ensembles, and YOLO
- **`deployment/`** - Streamlit web app for inference
- **`src/utils/config.py`** - Centralized configuration (data paths, batch size, model parameters, class names)

**Data Flow**: `data/{train,validation,test}/{Normal,Fractured}/` → `FractureDataset` → Augmentation pipeline → Model training

## Critical Patterns

### 1. Model Creation Pattern
All models use factory functions returning PyTorch modules:
```python
# src/models/resnet_classifier.py
model = create_resnet_model('resnet50', num_classes=2)  # Returns nn.Module with custom head

# src/models/efficientnet_classifier.py
model = create_efficientnet_model('efficientnet_b3', num_classes=2)
```
Replace final FC layers with dropout + two linear layers (512 hidden units). All models assume **ImageNet normalization** (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).

### 2. Augmentation Strategy (Medical Imaging Sensitive)
Training uses **Albumentations** pipeline with X-ray specific enhancements:
- CLAHE (Contrast Limited Adaptive Histogram Equalization) - always applied in validation, 50% in training
- **Cautious rotation**: ±20° max (medical images require care with spatial transforms)
- HorizontalFlip only 30% (X-rays have anatomical orientation concerns)
- ShiftScaleRotate, RandomBrightnessContrast, GaussNoise for regularization

Validation/test use CLAHE + Normalize only (100% CLAHE application).

### 3. Training Loop with `Trainer` Class
Standard PyTorch pattern from `src/training/train_classifier.py`:
- **Optimizer**: Adam (lr=1e-4, weight_decay=1e-5)
- **LR Scheduling**: ReduceLROnPlateau (factor=0.5, patience=5) + optional CosineAnnealingLR
- **Loss**: CrossEntropyLoss (single-class weighting available for imbalance)
- **Checkpoints**: Saves best model by validation accuracy + current epoch state
- **Tracking**: Maintains lists of train/val losses and accuracies

### 4. Ensemble Pattern (Soft Voting)
From `src/models/ensemble_model.py`:
- **EnsembleModel**: Softmax predictions + weighted averaging (default: equal weights)
- **EnsembleVoting**: Hard voting (mode of argmax predictions)
- All member models run in `.eval()` mode with `torch.no_grad()`

### 5. Evaluation Pipeline
`ModelEvaluator` computes: accuracy, precision, recall, F1, sensitivity, specificity, ROC-AUC, confusion matrix, per-class reports. Binary classification sensitivity/specificity treat class 1 (Fractured) as positive.

## Key Workflows

### Training Single Model
```bash
python train_single_model.py --model resnet50 --epochs 50
```
- Creates data loaders from Config paths
- Initializes model with ImageNet weights
- Runs `Trainer.train()` → saves checkpoint to `models/saved/best_model.pth`
- Generates loss/accuracy plots to `results/{model_name}/`

### Training Ensemble
```bash
python examples/train_ensemble.py
```
- Loads multiple pre-trained checkpoints
- Wraps in `EnsembleModel` with weighted averaging
- Evaluates combined predictions (typically 1-2% accuracy improvement)

### Deployment
```bash
cd deployment && streamlit run app.py
```
- Loads single/ensemble model + config
- Web UI for image upload + real-time prediction + Grad-CAM visualization

### Verification Scripts
- `test_dataloader.py` - Validates data loading pipeline and augmentation
- `test_setup.py` - Checks PyTorch/CUDA availability and config paths
- `verify_data.py` - Counts images per class, detects corrupted files

## Configuration & Paths
All paths relative to project root via `Config` class:
- `TRAIN_DIR`, `VAL_DIR`, `TEST_DIR` - Dataset folders (expect `{Normal,Fractured}/` subfolders)
- `MODEL_DIR = models/saved/` - Checkpoint storage
- `RESULTS_DIR = results/` - Plots and evaluation outputs
- `IMG_SIZE` - Model input: 224 (ResNet/EfficientNet-B0), 300 (B3), 600 (B7)
- Modify in `src/utils/config.py` if dataset paths differ

## Adding New Components

### New Model Architecture
1. Create `src/models/{model_name}_classifier.py`
2. Implement factory: `def create_{model_name}_model(name, num_classes) → nn.Module`
3. Ensure ImageNet normalization constants in augmentation pipeline
4. Add to `train_single_model.py` choices argument

### New Data Augmentation
Modify `get_train_transforms()` in `src/data/preprocessing.py`. Import additional transforms from `albumentations`. Maintain CLAHE for X-ray enhancement.

### Custom Evaluation Metric
Extend `ModelEvaluator._calculate_metrics()` or add new method. Return dict with metric name keys for `src/evaluation/visualize.py` compatibility.

## Dependencies & Environment
- **PyTorch 2.0+** with torchvision (models, transforms)
- **Ultralytics YOLOv8** (detection baseline)
- **Albumentations** (augmentation pipeline - not torchvision.transforms)
- **scikit-learn** (metrics computation)
- **Streamlit** (web deployment)
- GPU recommended (CUDA 11.8+)

## Import Conventions
All scripts add `src` to sys.path: `sys.path.append('src')` before importing `src.*` modules. Relative imports within `src/` use `from utils.config import Config` (no `src.` prefix).

## Debugging Tips
- Data loading issues → Run `test_dataloader.py` to visualize augmentation pipeline
- OOM errors → Reduce `BATCH_SIZE` in config or use smaller model (efficientnet_b0)
- Poor metrics → Check CLAHE application in validation (should be 100%), verify class balance
- Checkpoint loading → Ensure checkpoint dict has `model_state_dict` key (not full model pickle)
