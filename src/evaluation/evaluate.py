import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class ModelEvaluator:
    def __init__(self, model, test_loader, device, class_names):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.class_names = class_names
        
    def evaluate(self):
        """Comprehensive model evaluation"""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc='Evaluating'):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_labels, all_preds, all_probs)
        
        return metrics, all_preds, all_labels, all_probs
    
    def _calculate_metrics(self, y_true, y_pred, y_probs):
        """Calculate all evaluation metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'sensitivity': recall_score(y_true, y_pred, pos_label=1),
            'specificity': recall_score(y_true, y_pred, pos_label=0),
        }
        
        # ROC-AUC for binary classification
        if y_probs.shape[1] == 2:
            metrics['roc_auc'] = roc_auc_score(y_true, y_probs[:, 1])
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        # Classification report
        metrics['classification_report'] = classification_report(
            y_true, y_pred, 
            target_names=self.class_names,
            output_dict=True
        )
        
        return metrics
    
    def print_metrics(self, metrics):
        """Print evaluation metrics"""
        print("\n" + "="*60)
        print("EVALUATION METRICS")
        print("="*60)
        print(f"Accuracy:     {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"Precision:    {metrics['precision']:.4f}")
        print(f"Recall:       {metrics['recall']:.4f}")
        print(f"F1-Score:     {metrics['f1_score']:.4f}")
        print(f"Sensitivity:  {metrics['sensitivity']:.4f}")
        print(f"Specificity:  {metrics['specificity']:.4f}")
        
        if 'roc_auc' in metrics:
            print(f"ROC-AUC:      {metrics['roc_auc']:.4f}")
        
        print("\n" + "-"*60)
        print("CLASSIFICATION REPORT")
        print("-"*60)
        
        for class_name in self.class_names:
            report = metrics['classification_report'][class_name]
            print(f"\n{class_name}:")
            print(f"  Precision: {report['precision']:.4f}")
            print(f"  Recall:    {report['recall']:.4f}")
            print(f"  F1-Score:  {report['f1-score']:.4f}")
            print(f"  Support:   {report['support']}")
        
        print("\n" + "="*60)

class DetectionEvaluator:
    """Evaluator for object detection models (YOLO, RetinaNet)"""
    
    def __init__(self, model, test_loader, device, iou_threshold=0.5):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.iou_threshold = iou_threshold
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def calculate_map(self, predictions, ground_truths, iou_thresholds=[0.5, 0.75]):
        """Calculate mean Average Precision"""
        aps = []
        
        for iou_thresh in iou_thresholds:
            # Calculate AP at this IoU threshold
            ap = self._calculate_ap(predictions, ground_truths, iou_thresh)
            aps.append(ap)
        
        return np.mean(aps)
    
    def _calculate_ap(self, predictions, ground_truths, iou_threshold):
        """Calculate Average Precision at specific IoU threshold"""
        # Sort predictions by confidence
        pred_sorted = sorted(predictions, key=lambda x: x['score'], reverse=True)
        
        tp = np.zeros(len(pred_sorted))
        fp = np.zeros(len(pred_sorted))
        
        gt_matched = set()
        
        for i, pred in enumerate(pred_sorted):
            best_iou = 0
            best_gt_idx = -1
            
            # Find best matching ground truth
            for j, gt in enumerate(ground_truths):
                if j in gt_matched:
                    continue
                
                iou = self.calculate_iou(pred['box'], gt['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_iou >= iou_threshold:
                tp[i] = 1
                gt_matched.add(best_gt_idx)
            else:
                fp[i] = 1
        
        # Calculate precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / len(ground_truths)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        # Calculate AP using 11-point interpolation
        ap = 0
        for t in np.linspace(0, 1, 11):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11
        
        return ap