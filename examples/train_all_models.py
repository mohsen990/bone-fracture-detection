import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
from src.utils.config import Config
from src.data.data_loader import create_data_loaders
from src.models.resnet_classifier import create_resnet_model
from src.models.efficientnet_classifier import create_efficientnet_model
from src.training.train_classifier import Trainer
from src.evaluation.evaluate import ModelEvaluator
from src.evaluation.visualize import Visualizer

def train_single_model(model_name, config, train_loader, val_loader, test_loader, device):
    """Train and evaluate a single model"""
    
    print(f"\n{'='*80}")
    print(f"Training {model_name}")
    print(f"{'='*80}\n")
    
    # Create model
    if 'resnet' in model_name:
        model = create_resnet_model(model_name, num_classes=config.NUM_CLASSES)
    elif 'efficientnet' in model_name:
        model = create_efficientnet_model(model_name, num_classes=config.NUM_CLASSES)
    
    # Train
    trainer = Trainer(model, train_loader, val_loader, device, config, model_name=model_name)
    train_losses, val_losses, train_accs, val_accs = trainer.train(config.EPOCHS)
    
    # Visualize training history
    viz = Visualizer(save_dir=f'results/{model_name}')
    viz.plot_training_history(train_losses, val_losses, train_accs, val_accs,
                              save_name=f'{model_name}_history.png')
    
    # Evaluate on test set
    evaluator = ModelEvaluator(model, test_loader, device, config.CLASS_NAMES)
    metrics, preds, labels, probs = evaluator.evaluate()
    evaluator.print_metrics(metrics)
    
    # Visualize results
    viz.plot_confusion_matrix(metrics['confusion_matrix'], config.CLASS_NAMES,
                              save_name=f'{model_name}_confusion_matrix.png')
    viz.plot_roc_curve(labels, probs, save_name=f'{model_name}_roc_curve.png')
    viz.plot_sample_predictions(model, test_loader, device, config.CLASS_NAMES,
                               save_name=f'{model_name}_predictions.png')
    
    return model, metrics

def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    config = Config()
    
    # Create data loaders
    print("Loading datasets...")
    train_loader, val_loader, test_loader = create_data_loaders(
        config.TRAIN_DIR,
        config.VAL_DIR,
        config.TEST_DIR,
        batch_size=config.BATCH_SIZE,
        img_size=config.IMG_SIZE
    )
    
    # Models to train
    models_to_train = [
        'resnet50',
        'resnet101',
        'efficientnet_b0',
        'efficientnet_b3',
    ]
    
    trained_models = {}
    all_metrics = {}
    
    # Train each model
    for model_name in models_to_train:
        model, metrics = train_single_model(
            model_name, config, train_loader, val_loader, test_loader, device
        )
        trained_models[model_name] = model
        all_metrics[model_name] = metrics
    
    # Compare models
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    for model_name, metrics in all_metrics.items():
        print(f"\n{model_name}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC:  {metrics.get('roc_auc', 'N/A')}")
    
    return trained_models, all_metrics

if __name__ == '__main__':
    trained_models, metrics = main()