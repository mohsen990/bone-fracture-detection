# generate_report.py
"""
Comprehensive evaluation script for project report
Generates all necessary metrics and visualizations
"""

import sys
import os
sys.path.append('src')

import torch
import argparse
from pathlib import Path

from src.utils.config import Config
from src.data.data_loader import create_data_loaders
from src.models.resnet_classifier import create_resnet_model
from src.models.efficientnet_classifier import create_efficientnet_model
from src.evaluation.evaluate import ModelEvaluator
from src.evaluation.visualize import Visualizer

def generate_comprehensive_report(models_to_evaluate=None):
    """
    Generate comprehensive evaluation report for all models

    Args:
        models_to_evaluate: List of (model_name, checkpoint_path) tuples
                           If None, evaluates all available models
    """

    print("="*80)
    print("GENERATING COMPREHENSIVE PROJECT REPORT")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = Config()

    # Create report directory
    report_dir = Path('project_report')
    report_dir.mkdir(exist_ok=True)

    # Load test data
    print("\n📊 Loading test data...")
    _, _, test_loader = create_data_loaders(
        config.TRAIN_DIR,
        config.VAL_DIR,
        config.TEST_DIR,
        batch_size=config.BATCH_SIZE,
        img_size=config.IMG_SIZE
    )

    # Default models to evaluate
    if models_to_evaluate is None:
        models_to_evaluate = []
        model_dir = Path(config.MODEL_DIR)

        # Look for trained models
        model_configs = [
            ('resnet50', 'best_model.pth'),
            ('resnet101', 'resnet101_best_model.pth'),
            ('efficientnet_b0', 'efficientnet_b0_best_model.pth'),
            ('efficientnet_b3', 'efficientnet_b3_best_model.pth'),
        ]

        for model_name, checkpoint_file in model_configs:
            checkpoint_path = model_dir / checkpoint_file
            if checkpoint_path.exists():
                models_to_evaluate.append((model_name, str(checkpoint_path)))

    if not models_to_evaluate:
        print("⚠️  No trained models found!")
        print("Please train models first using:")
        print("  python train_single_model.py --model resnet50")
        return

    print(f"\n✅ Found {len(models_to_evaluate)} models to evaluate")

    # Store all results
    all_results = {}

    # Evaluate each model
    for model_name, checkpoint_path in models_to_evaluate:
        print("\n" + "="*80)
        print(f"📈 EVALUATING: {model_name.upper()}")
        print("="*80)

        try:
            # Load model
            if 'resnet' in model_name.lower():
                model = create_resnet_model(model_name, num_classes=config.NUM_CLASSES)
            elif 'efficientnet' in model_name.lower():
                model = create_efficientnet_model(model_name, num_classes=config.NUM_CLASSES)
            else:
                print(f"⚠️  Unknown model type: {model_name}, skipping...")
                continue

            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)

            print(f"✅ Loaded from: {checkpoint_path}")

            # Evaluate
            evaluator = ModelEvaluator(model, test_loader, device, config.CLASS_NAMES)
            metrics, preds, labels, probs = evaluator.evaluate()
            evaluator.print_metrics(metrics)

            # Save results
            all_results[model_name] = {
                'metrics': metrics,
                'predictions': preds,
                'labels': labels,
                'probabilities': probs
            }

            # Create visualizations
            model_report_dir = report_dir / model_name
            model_report_dir.mkdir(exist_ok=True)

            viz = Visualizer(save_dir=str(model_report_dir))

            print(f"\n📊 Generating visualizations for {model_name}...")

            # Confusion Matrix
            viz.plot_confusion_matrix(
                metrics['confusion_matrix'],
                config.CLASS_NAMES,
                save_name='confusion_matrix.png'
            )

            # ROC Curve
            viz.plot_roc_curve(
                labels, probs,
                save_name='roc_curve.png'
            )

            # Precision-Recall Curve
            viz.plot_precision_recall_curve(
                labels, probs,
                save_name='precision_recall_curve.png'
            )

            # Sample Predictions
            viz.plot_sample_predictions(
                model, test_loader, device, config.CLASS_NAMES,
                num_samples=16,
                save_name='sample_predictions.png'
            )

            print(f"✅ Visualizations saved to: {model_report_dir}")

        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")
            continue

    # Generate comparison report
    if len(all_results) > 1:
        print("\n" + "="*80)
        print("📊 GENERATING MODEL COMPARISON")
        print("="*80)

        generate_comparison_report(all_results, report_dir)

    # Generate summary report
    print("\n" + "="*80)
    print("📄 GENERATING SUMMARY REPORT")
    print("="*80)

    generate_summary_document(all_results, report_dir)

    print("\n" + "="*80)
    print("✅ REPORT GENERATION COMPLETE!")
    print("="*80)
    print(f"\n📁 All results saved to: {report_dir.absolute()}")
    print("\nGenerated files:")
    print("  - Individual model evaluations in project_report/<model_name>/")
    print("  - Model comparison charts in project_report/comparison/")
    print("  - Summary report in project_report/summary_report.txt")


def generate_comparison_report(all_results, report_dir):
    """Generate model comparison visualizations"""
    import matplotlib.pyplot as plt
    import numpy as np

    comparison_dir = report_dir / 'comparison'
    comparison_dir.mkdir(exist_ok=True)

    # Extract metrics
    model_names = list(all_results.keys())
    accuracies = [all_results[m]['metrics']['accuracy'] for m in model_names]
    precisions = [all_results[m]['metrics']['precision'] for m in model_names]
    recalls = [all_results[m]['metrics']['recall'] for m in model_names]
    f1_scores = [all_results[m]['metrics']['f1_score'] for m in model_names]

    # 1. Accuracy comparison bar chart
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(model_names)), accuracies, color='skyblue', edgecolor='black', alpha=0.7)

    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc*100:.2f}%',
                ha='center', va='bottom', fontweight='bold')

    plt.xlabel('Model', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
    plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    plt.xticks(range(len(model_names)), [m.replace('_', ' ').title() for m in model_names], rotation=45, ha='right')
    plt.ylim([0.8, 1.0])
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(comparison_dir / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. All metrics comparison
    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(model_names))
    width = 0.2

    bars1 = ax.bar(x - 1.5*width, accuracies, width, label='Accuracy', alpha=0.8)
    bars2 = ax.bar(x - 0.5*width, precisions, width, label='Precision', alpha=0.8)
    bars3 = ax.bar(x + 0.5*width, recalls, width, label='Recall', alpha=0.8)
    bars4 = ax.bar(x + 1.5*width, f1_scores, width, label='F1-Score', alpha=0.8)

    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Comprehensive Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in model_names], rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0.8, 1.0])

    plt.tight_layout()
    plt.savefig(comparison_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Comparison charts saved to: {comparison_dir}")


def generate_summary_document(all_results, report_dir):
    """Generate text summary report"""

    summary_path = report_dir / 'summary_report.txt'

    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("BONE FRACTURE DETECTION - COMPREHENSIVE EVALUATION REPORT\n")
        f.write("="*80 + "\n\n")

        f.write("PROJECT OVERVIEW\n")
        f.write("-"*80 + "\n")
        f.write("This report presents a comprehensive evaluation of multiple deep learning\n")
        f.write("models for bone fracture detection in X-ray images.\n\n")

        f.write("MODELS EVALUATED\n")
        f.write("-"*80 + "\n")
        for idx, model_name in enumerate(all_results.keys(), 1):
            f.write(f"{idx}. {model_name.replace('_', ' ').title()}\n")
        f.write("\n")

        f.write("DETAILED RESULTS\n")
        f.write("="*80 + "\n\n")

        for model_name, results in all_results.items():
            metrics = results['metrics']

            f.write(f"{model_name.upper()}\n")
            f.write("-"*80 + "\n")
            f.write(f"Accuracy:     {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
            f.write(f"Precision:    {metrics['precision']:.4f}\n")
            f.write(f"Recall:       {metrics['recall']:.4f}\n")
            f.write(f"F1-Score:     {metrics['f1_score']:.4f}\n")
            f.write(f"Sensitivity:  {metrics['sensitivity']:.4f}\n")
            f.write(f"Specificity:  {metrics['specificity']:.4f}\n")

            if 'roc_auc' in metrics:
                f.write(f"ROC-AUC:      {metrics['roc_auc']:.4f}\n")

            f.write("\nConfusion Matrix:\n")
            cm = metrics['confusion_matrix']
            f.write(f"{cm}\n\n")

        # Best model
        best_model = max(all_results.keys(),
                        key=lambda k: all_results[k]['metrics']['accuracy'])
        best_acc = all_results[best_model]['metrics']['accuracy']

        f.write("\n" + "="*80 + "\n")
        f.write("SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"Best Model: {best_model.replace('_', ' ').title()}\n")
        f.write(f"Best Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)\n")
        f.write("\n" + "="*80 + "\n")

    print(f"✅ Summary report saved to: {summary_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate comprehensive evaluation report')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                       help='Specific models to evaluate (default: all available)')

    args = parser.parse_args()

    generate_comprehensive_report(args.models)
