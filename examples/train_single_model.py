# train_single_model.py
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import argparse

from src.utils.config import Config
from src.data.data_loader import create_data_loaders
from src.models.resnet_classifier import create_resnet_model
from src.models.efficientnet_classifier import create_efficientnet_model
from src.training.train_classifier import Trainer
from src.evaluation.visualize import Visualizer

def train_model(model_name='resnet50', epochs=50):
    """
    Train a single classification model
    
    Args:
        model_name: 'resnet50', 'resnet101', 'efficientnet_b0', 'efficientnet_b3'
        epochs: Number of training epochs
    """
    
    print("="*70)
    print(f"Training {model_name.upper()}")
    print("="*70)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n✅ Using device: {device}')
    
    if torch.cuda.is_available():
        print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
        print(f'✅ CUDA Version: {torch.version.cuda}')
    
    config = Config()
    
    # Create data loaders
    print("\n" + "="*70)
    print("Loading Data")
    print("="*70)
    
    train_loader, val_loader, test_loader = create_data_loaders(
        config.TRAIN_DIR,
        config.VAL_DIR,
        config.TEST_DIR,
        batch_size=config.BATCH_SIZE,
        img_size=config.IMG_SIZE
    )
    
    print(f"✅ Train samples: {len(train_loader.dataset)}")
    print(f"✅ Validation samples: {len(val_loader.dataset)}")
    print(f"✅ Test samples: {len(test_loader.dataset)}")
    print(f"✅ Batch size: {config.BATCH_SIZE}")
    
    # Create model
    print("\n" + "="*70)
    print("Creating Model")
    print("="*70)
    
    if 'resnet' in model_name.lower():
        model = create_resnet_model(model_name, num_classes=config.NUM_CLASSES)
    elif 'efficientnet' in model_name.lower():
        model = create_efficientnet_model(model_name, num_classes=config.NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    print(f"✅ Model: {model_name}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Total parameters: {total_params:,}")
    print(f"✅ Trainable parameters: {trainable_params:,}")
    
    # Train
    print("\n" + "="*70)
    print("Training")
    print("="*70)

    trainer = Trainer(model, train_loader, val_loader, device, config, model_name=model_name)
    train_losses, val_losses, train_accs, val_accs = trainer.train(epochs)
    
    # Visualize training history
    print("\n" + "="*70)
    print("Creating Visualizations")
    print("="*70)
    
    viz = Visualizer(save_dir=f'results/{model_name}')
    viz.plot_training_history(
        train_losses, val_losses, train_accs, val_accs,
        save_name=f'{model_name}_training_history.png'
    )
    
    print(f"✅ Training history saved to results/{model_name}/")
    
    # Final summary
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"✅ Best validation accuracy: {trainer.best_val_acc:.2f}%")
    save_name = f'{model_name}_best_model.pth' if model_name != 'resnet50' else 'best_model.pth'
    print(f"✅ Model saved to: {os.path.join(config.MODEL_DIR, save_name)}")

    return model, trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a fracture detection model')
    parser.add_argument('--model', type=str, default='resnet50',
                       choices=['resnet50', 'resnet101', 'efficientnet_b0', 'efficientnet_b3', 'efficientnet_b7'],
                       help='Model architecture')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    
    args = parser.parse_args()
    
    model, trainer = train_model(args.model, args.epochs)