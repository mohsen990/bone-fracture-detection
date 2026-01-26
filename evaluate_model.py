# evaluate_model.py
import sys
import os
sys.path.append('src')

import torch
import argparse

from src.utils.config import Config
from src.data.data_loader import create_data_loaders
from src.models.resnet_classifier import create_resnet_model
from src.models.efficientnet_classifier import create_efficientnet_model
from src.evaluation.evaluate import ModelEvaluator
from src.evaluation.visualize import Visualizer

def evaluate_model(model_name='resnet50', checkpoint_path=None):
    """Evaluate a trained model"""
    
    print("="*70)
    print(f"Evaluating {model_name.upper()}")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = Config()
    
    # Load data
    print("\nLoading test data...")
    _, _, test_loader = create_data_loaders(
        config.TRAIN_DIR,
        config.VAL_DIR,
        config.TEST_DIR,
        batch_size=config.BATCH_SIZE,
        img_size=config.IMG_SIZE
    )
    
    # Create model
    print(f"Loading {model_name}...")
    if 'resnet' in model_name.lower():
        model = create_resnet_model(model_name, num_classes=config.NUM_CLASSES)
    elif 'efficientnet' in model_name.lower():
        model = create_efficientnet_model(model_name, num_classes=config.NUM_CLASSES)
    
    # Load checkpoint
    if checkpoint_path is None:
        checkpoint_path = os.path.join(config.MODEL_DIR, 'best_model.pth')
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"✅ Model loaded from: {checkpoint_path}")
    
    # Evaluate
    print("\nEvaluating on test set...")
    evaluator = ModelEvaluator(model, test_loader, device, config.CLASS_NAMES)
    metrics, preds, labels, probs = evaluator.evaluate()
    evaluator.print_metrics(metrics)
    
    # Visualize
    print("\nCreating visualizations...")
    viz = Visualizer(save_dir=f'results/{model_name}')
    
    viz.plot_confusion_matrix(
        metrics['confusion_matrix'],
        config.CLASS_NAMES,
        save_name=f'{model_name}_confusion_matrix.png'
    )
    
    viz.plot_roc_curve(
        labels, probs,
        save_name=f'{model_name}_roc_curve.png'
    )
    
    viz.plot_sample_predictions(
        model, test_loader, device, config.CLASS_NAMES,
        save_name=f'{model_name}_predictions.png'
    )
    
    print(f"\n✅ Visualizations saved to results/{model_name}/")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a trained model')
    parser.add_argument('--model', type=str, default='resnet50',
                       help='Model architecture')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint')
    
    args = parser.parse_args()
    
    evaluate_model(args.model, args.checkpoint)