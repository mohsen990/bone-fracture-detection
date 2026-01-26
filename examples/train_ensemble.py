import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.utils.config import Config
from src.data.data_loader import create_data_loaders
from src.models.resnet_classifier import create_resnet_model
from src.models.efficientnet_classifier import create_efficientnet_model
from src.models.ensemble_model import EnsembleModel, EnsembleVoting
from src.evaluation.evaluate import ModelEvaluator
from src.evaluation.visualize import Visualizer


def load_trained_model(model_type, model_path, num_classes, device):
    """
    Load a trained model from checkpoint
    
    Args:
        model_type: Type of model ('resnet50', 'efficientnet_b3', etc.)
        model_path: Path to checkpoint file
        num_classes: Number of classes
        device: Device to load model on
    
    Returns:
        Loaded model
    """
    print(f"Loading {model_type}...")
    
    # Create model
    if 'resnet' in model_type.lower():
        model = create_resnet_model(model_type, num_classes=num_classes)
    elif 'efficientnet' in model_type.lower():
        model = create_efficientnet_model(model_type, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load checkpoint
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  ✓ Loaded from {model_path}")
        
        if 'best_val_acc' in checkpoint:
            print(f"  ✓ Best Val Accuracy: {checkpoint['best_val_acc']:.2f}%")
    else:
        print(f"  ✗ Checkpoint not found: {model_path}")
        print(f"  → Using pretrained model (not trained on fracture data)")
    
    model = model.to(device)
    model.eval()
    
    return model


def evaluate_individual_models(models, model_names, test_loader, device, class_names):
    """
    Evaluate individual models before ensembling
    
    Args:
        models: List of models
        model_names: List of model names
        test_loader: Test data loader
        device: Device
        class_names: List of class names
    
    Returns:
        Dictionary of individual model metrics
    """
    print("\n" + "="*80)
    print("EVALUATING INDIVIDUAL MODELS")
    print("="*80)
    
    individual_metrics = {}
    
    for model, name in zip(models, model_names):
        print(f"\nEvaluating {name}...")
        print("-"*60)
        
        evaluator = ModelEvaluator(model, test_loader, device, class_names)
        metrics, preds, labels, probs = evaluator.evaluate()
        
        print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        
        if 'roc_auc' in metrics:
            print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        individual_metrics[name] = metrics
    
    return individual_metrics


def optimize_ensemble_weights(models, val_loader, device, num_classes):
    """
    Find optimal weights for ensemble using validation set
    
    Args:
        models: List of models
        val_loader: Validation data loader
        device: Device
        num_classes: Number of classes
    
    Returns:
        Optimal weights for each model
    """
    print("\n" + "="*80)
    print("OPTIMIZING ENSEMBLE WEIGHTS")
    print("="*80)
    
    # Collect predictions from all models
    all_predictions = [[] for _ in models]
    all_labels = []
    
    print("\nCollecting predictions from individual models...")
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Processing'):
            images = images.to(device)
            
            for idx, model in enumerate(models):
                model.eval()
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                all_predictions[idx].append(probs.cpu())
            
            all_labels.append(labels)
    
    # Concatenate all predictions
    for idx in range(len(models)):
        all_predictions[idx] = torch.cat(all_predictions[idx], dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Grid search for best weights
    print("\nSearching for optimal weights...")
    best_acc = 0
    best_weights = None
    
    # Generate weight combinations
    from itertools import product
    weight_options = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # For 3 models, try different combinations
    tested = 0
    for weights in product(weight_options, repeat=len(models)):
        # Normalize weights to sum to 1
        total = sum(weights)
        if total == 0:
            continue
        
        normalized_weights = [w / total for w in weights]
        
        # Calculate ensemble prediction
        ensemble_pred = torch.zeros_like(all_predictions[0])
        for pred, weight in zip(all_predictions, normalized_weights):
            ensemble_pred += weight * pred
        
        # Calculate accuracy
        _, predicted = torch.max(ensemble_pred, 1)
        acc = (predicted == all_labels).float().mean().item()
        
        if acc > best_acc:
            best_acc = acc
            best_weights = normalized_weights
        
        tested += 1
    
    print(f"\n✓ Tested {tested} weight combinations")
    print(f"✓ Best validation accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
    print(f"✓ Optimal weights: {[f'{w:.3f}' for w in best_weights]}")
    
    return best_weights


def compare_ensemble_strategies(models, model_names, test_loader, device, 
                               class_names, weights=None):
    """
    Compare different ensemble strategies
    
    Args:
        models: List of models
        model_names: List of model names
        test_loader: Test data loader
        device: Device
        class_names: List of class names
        weights: Optional weights for weighted average
    
    Returns:
        Dictionary of ensemble metrics
    """
    print("\n" + "="*80)
    print("COMPARING ENSEMBLE STRATEGIES")
    print("="*80)
    
    ensemble_results = {}
    
    # Strategy 1: Simple Average (Equal Weights)
    print("\n1. Simple Average Ensemble (Equal Weights)")
    print("-"*60)
    ensemble_avg = EnsembleModel(models, weights=None)
    ensemble_avg = ensemble_avg.to(device)
    
    evaluator_avg = ModelEvaluator(ensemble_avg, test_loader, device, class_names)
    metrics_avg, preds_avg, labels_avg, probs_avg = evaluator_avg.evaluate()
    evaluator_avg.print_metrics(metrics_avg)
    
    ensemble_results['simple_average'] = {
        'metrics': metrics_avg,
        'predictions': preds_avg,
        'labels': labels_avg,
        'probabilities': probs_avg
    }
    
    # Strategy 2: Weighted Average (Optimized Weights)
    if weights is not None:
        print("\n2. Weighted Average Ensemble (Optimized Weights)")
        print("-"*60)
        print(f"Weights: {[f'{w:.3f}' for w in weights]}")
        
        ensemble_weighted = EnsembleModel(models, weights=weights)
        ensemble_weighted = ensemble_weighted.to(device)
        
        evaluator_weighted = ModelEvaluator(ensemble_weighted, test_loader, device, class_names)
        metrics_weighted, preds_weighted, labels_weighted, probs_weighted = evaluator_weighted.evaluate()
        evaluator_weighted.print_metrics(metrics_weighted)
        
        ensemble_results['weighted_average'] = {
            'metrics': metrics_weighted,
            'predictions': preds_weighted,
            'labels': labels_weighted,
            'probabilities': probs_weighted
        }
    
    # Strategy 3: Voting Ensemble
    print("\n3. Hard Voting Ensemble")
    print("-"*60)
    ensemble_voting = EnsembleVoting(models, num_classes=2)
    ensemble_voting = ensemble_voting.to(device)
    
    evaluator_voting = ModelEvaluator(ensemble_voting, test_loader, device, class_names)
    metrics_voting, preds_voting, labels_voting, probs_voting = evaluator_voting.evaluate()
    evaluator_voting.print_metrics(metrics_voting)
    
    ensemble_results['voting'] = {
        'metrics': metrics_voting,
        'predictions': preds_voting,
        'labels': labels_voting,
        'probabilities': probs_voting
    }
    
    return ensemble_results


def visualize_ensemble_results(ensemble_results, individual_metrics, 
                               model_names, class_names, save_dir='results/ensemble'):
    """
    Visualize and compare ensemble results
    
    Args:
        ensemble_results: Dictionary of ensemble results
        individual_metrics: Dictionary of individual model metrics
        model_names: List of model names
        class_names: List of class names
        save_dir: Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    viz = Visualizer(save_dir=save_dir)
    
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    # 1. Compare accuracies
    print("\n1. Creating accuracy comparison plot...")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    models = list(individual_metrics.keys()) + list(ensemble_results.keys())
    accuracies = ([individual_metrics[m]['accuracy'] for m in individual_metrics.keys()] +
                 [ensemble_results[m]['metrics']['accuracy'] for m in ensemble_results.keys()])
    
    colors = ['skyblue'] * len(individual_metrics) + ['coral'] * len(ensemble_results)
    
    bars = ax.bar(range(len(models)), accuracies, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc*100:.2f}%',
                ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Model Accuracy Comparison: Individual vs Ensemble', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m.replace('_', ' ').title() for m in models], 
                       rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0.8, 1.0])
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='skyblue', edgecolor='black', label='Individual Models'),
        Patch(facecolor='coral', edgecolor='black', label='Ensemble Models')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'accuracy_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    print("  ✓ Saved accuracy_comparison.png")
    
    # 2. Confusion matrices for best ensemble
    best_ensemble = max(ensemble_results.keys(), 
                       key=lambda k: ensemble_results[k]['metrics']['accuracy'])
    
    print(f"\n2. Creating confusion matrix for best ensemble ({best_ensemble})...")
    best_metrics = ensemble_results[best_ensemble]['metrics']
    viz.plot_confusion_matrix(
        best_metrics['confusion_matrix'],
        class_names,
        save_name=f'{best_ensemble}_confusion_matrix.png'
    )
    print(f"  ✓ Saved {best_ensemble}_confusion_matrix.png")
    
    # 3. ROC curves
    print("\n3. Creating ROC curves...")
    best_labels = ensemble_results[best_ensemble]['labels']
    best_probs = ensemble_results[best_ensemble]['probabilities']
    viz.plot_roc_curve(best_labels, best_probs, 
                      save_name=f'{best_ensemble}_roc_curve.png')
    print(f"  ✓ Saved {best_ensemble}_roc_curve.png")
    
    # 4. Precision-Recall curve
    print("\n4. Creating Precision-Recall curve...")
    viz.plot_precision_recall_curve(best_labels, best_probs,
                                   save_name=f'{best_ensemble}_pr_curve.png')
    print(f"  ✓ Saved {best_ensemble}_pr_curve.png")
    
    # 5. Metrics comparison heatmap
    print("\n5. Creating metrics comparison heatmap...")
    create_metrics_heatmap(individual_metrics, ensemble_results, save_dir)
    print("  ✓ Saved metrics_heatmap.png")


def create_metrics_heatmap(individual_metrics, ensemble_results, save_dir):
    """Create heatmap comparing all metrics across models"""
    import seaborn as sns
    import pandas as pd
    
    # Prepare data
    all_models = {}
    all_models.update(individual_metrics)
    all_models.update({k: v['metrics'] for k, v in ensemble_results.items()})
    
    metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1_score']
    
    data = []
    for model_name, metrics in all_models.items():
        row = [metrics[m] for m in metrics_to_compare]
        data.append(row)
    
    df = pd.DataFrame(
        data,
        columns=[m.replace('_', ' ').title() for m in metrics_to_compare],
        index=[m.replace('_', ' ').title() for m in all_models.keys()]
    )
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, fmt='.4f', cmap='RdYlGn', 
                cbar_kws={'label': 'Score'}, 
                vmin=0.85, vmax=1.0,
                linewidths=0.5, linecolor='gray')
    
    plt.title('Comprehensive Metrics Comparison', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Metrics', fontsize=12, fontweight='bold')
    plt.ylabel('Models', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_heatmap.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()


def save_ensemble_model(ensemble, weights, model_names, save_path):
    """
    Save ensemble model and configuration

    Args:
        ensemble: Ensemble model
        weights: Model weights
        model_names: List of model names
        save_path: Path to save ensemble
    """
    print(f"\nSaving ensemble model to {save_path}...")

    # Create directory if it doesn't exist
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    checkpoint = {
        'ensemble_state_dict': ensemble.state_dict(),
        'weights': weights,
        'model_names': model_names,
        'num_models': len(model_names)
    }

    torch.save(checkpoint, save_path)
    print(f"✓ Ensemble model saved successfully!")


def main():
    """Main ensemble training and evaluation pipeline"""
    
    print("\n" + "="*80)
    print("BONE FRACTURE DETECTION - ENSEMBLE MODEL TRAINING")
    print("="*80)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n✓ Using device: {device}')
    
    if torch.cuda.is_available():
        print(f'✓ GPU: {torch.cuda.get_device_name(0)}')
        print(f'✓ CUDA Version: {torch.version.cuda}')
    
    config = Config()
    
    # Create data loaders
    print("\n" + "="*80)
    print("LOADING DATASETS")
    print("="*80)
    
    train_loader, val_loader, test_loader = create_data_loaders(
        config.TRAIN_DIR,
        config.VAL_DIR,
        config.TEST_DIR,
        batch_size=config.BATCH_SIZE,
        img_size=config.IMG_SIZE
    )
    
    print(f"\n✓ Training samples: {len(train_loader.dataset)}")
    print(f"✓ Validation samples: {len(val_loader.dataset)}")
    print(f"✓ Test samples: {len(test_loader.dataset)}")
    
    # Define models to ensemble
    # Note: resnet50 saves as 'best_model.pth', others use '{model_name}_best_model.pth'
    # Paths are relative to examples/ directory (use ../ to go to project root)
    model_configs = [
        {'name': 'resnet50', 'path': '../models/saved/best_model.pth'},
        {'name': 'resnet101', 'path': '../models/saved/resnet101_best_model.pth'},
        {'name': 'efficientnet_b3', 'path': '../models/saved/efficientnet_b3_best_model.pth'},
    ]
    
    # Load individual models
    print("\n" + "="*80)
    print("LOADING INDIVIDUAL MODELS")
    print("="*80)
    
    models = []
    model_names = []
    
    for config_dict in model_configs:
        try:
            model = load_trained_model(
                config_dict['name'],
                config_dict['path'],
                config.NUM_CLASSES,
                device
            )
            models.append(model)
            model_names.append(config_dict['name'])
        except Exception as e:
            print(f"✗ Error loading {config_dict['name']}: {e}")
            print(f"  Skipping this model...")
    
    if len(models) < 2:
        print("\n✗ Error: Need at least 2 models for ensemble!")
        print("  Please train individual models first using train_all_models.py")
        return
    
    print(f"\n✓ Successfully loaded {len(models)} models for ensemble")
    
    # Evaluate individual models
    individual_metrics = evaluate_individual_models(
        models, model_names, test_loader, device, config.CLASS_NAMES
    )
    
    # Optimize ensemble weights
    optimal_weights = optimize_ensemble_weights(
        models, val_loader, device, config.NUM_CLASSES
    )
    
    # Compare ensemble strategies
    ensemble_results = compare_ensemble_strategies(
        models, model_names, test_loader, device, 
        config.CLASS_NAMES, weights=optimal_weights
    )
    
    # Final comparison
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    
    print("\nIndividual Models:")
    print("-"*60)
    for name, metrics in individual_metrics.items():
        print(f"{name:20s} - Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    
    print("\nEnsemble Models:")
    print("-"*60)
    for name, results in ensemble_results.items():
        metrics = results['metrics']
        print(f"{name:20s} - Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    
    # Find best model overall
    all_results = {**individual_metrics, 
                  **{k: v['metrics'] for k, v in ensemble_results.items()}}
    best_model = max(all_results.keys(), key=lambda k: all_results[k]['accuracy'])
    best_acc = all_results[best_model]['accuracy']
    
    print("\n" + "="*60)
    print(f"🏆 BEST MODEL: {best_model}")
    print(f"🏆 BEST ACCURACY: {best_acc:.4f} ({best_acc*100:.2f}%)")
    print("="*60)
    
    # Visualize results
    visualize_ensemble_results(
        ensemble_results, individual_metrics,
        model_names, config.CLASS_NAMES,
        save_dir='../results/ensemble'
    )
    
    # Save best ensemble
    if 'weighted_average' in ensemble_results:
        best_ensemble = EnsembleModel(models, weights=optimal_weights)
        save_ensemble_model(
            best_ensemble, optimal_weights, model_names,
            '../models/saved/ensemble_weighted_best.pth'
        )
    
    print("\n" + "="*80)
    print("✓ ENSEMBLE TRAINING AND EVALUATION COMPLETED!")
    print("="*80)
    
    return ensemble_results, individual_metrics


if __name__ == '__main__':
    results = main()