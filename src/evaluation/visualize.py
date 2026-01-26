import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
import torch
from sklearn.metrics import roc_curve, auc, confusion_matrix
import os

class Visualizer:
    def __init__(self, save_dir='results/plots'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        sns.set_style('whitegrid')
    
    def plot_training_history(self, train_losses, val_losses, train_accs, val_accs, save_name='training_history.png'):
        """Plot training and validation metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        epochs = range(1, len(train_losses) + 1)
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, cm, class_names, save_name='confusion_matrix.png'):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        
        # Normalize confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(
            cm_norm, 
            annot=True, 
            fmt='.2%', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Percentage'},
            square=True,
            linewidths=1,
            linecolor='gray'
        )
        
        plt.title('Confusion Matrix (Normalized)', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        # Add counts
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                plt.text(j + 0.5, i + 0.7, f'n={cm[i, j]}', 
                        ha='center', va='center', fontsize=9, color='gray')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, y_true, y_probs, save_name='roc_curve.png'):
        """Plot ROC curve"""
        fpr, tpr, thresholds = roc_curve(y_true, y_probs[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curve(self, y_true, y_probs, save_name='precision_recall.png'):
        """Plot Precision-Recall curve"""
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_probs[:, 1])
        avg_precision = average_precision_score(y_true, y_probs[:, 1])
        
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AP = {avg_precision:.4f})')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left", fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_sample_predictions(self, model, test_loader, device, class_names, 
                               num_samples=16, save_name='sample_predictions.png'):
        """Plot sample predictions"""
        model.eval()
        
        images_list = []
        labels_list = []
        preds_list = []
        probs_list = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                images_list.extend(images.cpu())
                labels_list.extend(labels.cpu().numpy())
                preds_list.extend(predicted.cpu().numpy())
                probs_list.extend(probs.cpu().numpy())
                
                if len(images_list) >= num_samples:
                    break
        
        # Plot
        rows = 4
        cols = 4
        fig, axes = plt.subplots(rows, cols, figsize=(16, 16))
        axes = axes.flatten()
        
        for idx in range(min(num_samples, len(images_list))):
            ax = axes[idx]
            
            # Denormalize image
            img = images_list[idx].permute(1, 2, 0).numpy()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)
            
            # Display
            ax.imshow(img)
            
            true_label = class_names[labels_list[idx]]
            pred_label = class_names[preds_list[idx]]
            confidence = probs_list[idx][preds_list[idx]] * 100
            
            color = 'green' if labels_list[idx] == preds_list[idx] else 'red'
            title = f'True: {true_label}\nPred: {pred_label} ({confidence:.1f}%)'
            
            ax.set_title(title, color=color, fontsize=10, fontweight='bold')
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_detection_results(self, image, boxes, scores, labels, 
                                   class_names, save_name='detection_result.png'):
        """Visualize object detection results"""
        img_display = image.copy()
        
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box.astype(int)
            
            # Draw bounding box
            cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label_text = f'{class_names[label]}: {score:.2f}'
            (text_width, text_height), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            cv2.rectangle(img_display, (x1, y1 - text_height - 10), 
                         (x1 + text_width, y1), (0, 255, 0), -1)
            cv2.putText(img_display, label_text, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title('Fracture Detection Results', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.show()