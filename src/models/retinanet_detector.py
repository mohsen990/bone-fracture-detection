import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.models.detection.retinanet import RetinaNetHead
import torchvision.transforms as T
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

class RetinaNetFractureDetector:
    def __init__(self, num_classes=2, pretrained=True):
        """
        Initialize RetinaNet model for fracture detection
        
        Args:
            num_classes: number of classes (background + fracture = 2)
            pretrained: use pretrained weights on COCO
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load pretrained RetinaNet with ResNet50-FPN backbone
        self.model = retinanet_resnet50_fpn(pretrained=pretrained)
        
        # Modify head for custom number of classes
        in_features = self.model.head.classification_head.conv[0].in_channels
        num_anchors = self.model.head.classification_head.num_anchors
        
        self.model.head = RetinaNetHead(
            in_features,
            num_anchors,
            num_classes
        )
        
        self.model = self.model.to(self.device)
        self.num_classes = num_classes
        
        print(f"RetinaNet initialized with {num_classes} classes")
    
    def get_transform(self, train=False):
        """Get image transforms for training or inference"""
        transforms = []
        transforms.append(T.ToTensor())
        
        if train:
            # Training augmentations
            transforms.append(T.RandomHorizontalFlip(0.5))
            # Add more augmentations if needed
        
        return T.Compose(transforms)
    
    def train_model(self, train_loader, val_loader, epochs=50, lr=0.001, 
                   save_dir='models/saved', model_name='retinanet_fracture'):
        """
        Train RetinaNet model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            lr: Learning rate
            save_dir: Directory to save models
            model_name: Name for saved model
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Optimizer - using SGD as recommended for object detection
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params,
            lr=lr,
            momentum=0.9,
            weight_decay=0.0005
        )
        
        # Learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.1
        )
        
        # Alternative: Cosine Annealing
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer,
        #     T_max=epochs,
        #     eta_min=1e-6
        # )
        
        best_loss = float('inf')
        train_losses = []
        val_losses = []
        
        print("\nStarting training...")
        print("="*60)
        
        for epoch in range(epochs):
            print(f'\nEpoch {epoch + 1}/{epochs}')
            print('-' * 60)
            
            # Training phase
            train_loss = self._train_epoch(train_loader, optimizer)
            train_losses.append(train_loss)
            print(f'Train Loss: {train_loss:.4f}')
            
            # Validation phase
            val_loss = self._validate_epoch(val_loader)
            val_losses.append(val_loss)
            print(f'Val Loss: {val_loss:.4f}')
            
            # Update learning rate
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Learning Rate: {current_lr:.6f}')
            lr_scheduler.step()
            
            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                self._save_checkpoint(
                    epoch, optimizer, lr_scheduler, 
                    os.path.join(save_dir, f'{model_name}_best.pth')
                )
                print(f'✓ Best model saved! (Val Loss: {val_loss:.4f})')
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(
                    epoch, optimizer, lr_scheduler,
                    os.path.join(save_dir, f'{model_name}_epoch_{epoch+1}.pth')
                )
                print(f'✓ Checkpoint saved at epoch {epoch+1}')
        
        print('\n' + '='*60)
        print('Training completed!')
        print(f'Best Validation Loss: {best_loss:.4f}')
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_loss': best_loss
        }
    
    def _train_epoch(self, train_loader, optimizer):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for images, targets in pbar:
            # Move to device
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} 
                      for t in targets]
            
            # Forward pass
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass
            optimizer.zero_grad()
            losses.backward()
            
            # Gradient clipping (optional but recommended)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            
            optimizer.step()
            
            epoch_loss += losses.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{losses.item():.4f}',
                'cls_loss': f'{loss_dict["classification"].item():.4f}',
                'box_loss': f'{loss_dict["bbox_regression"].item():.4f}'
            })
        
        return epoch_loss / len(train_loader)
    
    def _validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.train()  # Keep in training mode to get losses
        val_loss = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation')
            for images, targets in pbar:
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} 
                          for t in targets]
                
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()
                
                pbar.set_postfix({'loss': f'{losses.item():.4f}'})
        
        return val_loss / len(val_loader)
    
    def _save_checkpoint(self, epoch, optimizer, scheduler, path):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Model loaded from {checkpoint_path}")
        
        if 'epoch' in checkpoint:
            print(f"  Epoch: {checkpoint['epoch'] + 1}")
    
    def predict(self, image_path, conf_threshold=0.5, nms_threshold=0.5):
        """
        Predict fractures in a single image
        
        Args:
            image_path: Path to image file
            conf_threshold: Confidence threshold for detections
            nms_threshold: NMS IoU threshold
        
        Returns:
            Dictionary with boxes, scores, and labels
        """
        self.model.eval()
        
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.get_transform()(image).to(self.device)
        
        # Predict
        with torch.no_grad():
            predictions = self.model([image_tensor])[0]
        
        # Filter by confidence
        keep = predictions['scores'] > conf_threshold
        boxes = predictions['boxes'][keep].cpu().numpy()
        scores = predictions['scores'][keep].cpu().numpy()
        labels = predictions['labels'][keep].cpu().numpy()
        
        # Apply NMS (Non-Maximum Suppression)
        if len(boxes) > 0:
            keep_nms = self._nms(boxes, scores, nms_threshold)
            boxes = boxes[keep_nms]
            scores = scores[keep_nms]
            labels = labels[keep_nms]
        
        return {
            'boxes': boxes,
            'scores': scores,
            'labels': labels,
            'image': np.array(image)
        }
    
    def predict_batch(self, image_paths, conf_threshold=0.5, nms_threshold=0.5):
        """
        Predict on multiple images
        
        Args:
            image_paths: List of image paths
            conf_threshold: Confidence threshold
            nms_threshold: NMS threshold
        
        Returns:
            List of prediction dictionaries
        """
        all_predictions = []
        
        for image_path in tqdm(image_paths, desc='Predicting'):
            pred = self.predict(image_path, conf_threshold, nms_threshold)
            all_predictions.append(pred)
        
        return all_predictions
    
    def _nms(self, boxes, scores, threshold):
        """
        Non-Maximum Suppression
        
        Args:
            boxes: Bounding boxes (N, 4)
            scores: Confidence scores (N,)
            threshold: IoU threshold
        
        Returns:
            Indices of boxes to keep
        """
        if len(boxes) == 0:
            return np.array([])
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(iou <= threshold)[0]
            order = order[inds + 1]
        
        return np.array(keep)
    
    def visualize_predictions(self, image_path, conf_threshold=0.5, 
                            save_path=None, class_names=None):
        """
        Visualize predictions on image
        
        Args:
            image_path: Path to image
            conf_threshold: Confidence threshold
            save_path: Path to save visualization (optional)
            class_names: List of class names
        """
        # Get predictions
        results = self.predict(image_path, conf_threshold)
        
        # Load image
        image = results['image'].copy()
        boxes = results['boxes']
        scores = results['scores']
        labels = results['labels']
        
        if class_names is None:
            class_names = [f'Class {i}' for i in range(self.num_classes)]
        
        # Draw boxes
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box.astype(int)
            
            # Draw bounding box
            color = (0, 255, 0)  # Green
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label_text = f'{class_names[label]}: {score:.2f}'
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Background for text
            cv2.rectangle(
                image,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # Text
            cv2.putText(
                image,
                label_text,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )
        
        # Display
        plt.figure(figsize=(12, 8))
        plt.imshow(image)
        plt.axis('off')
        plt.title(f'Fracture Detection Results (Threshold: {conf_threshold})', 
                 fontsize=14, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Visualization saved to {save_path}")
        
        plt.show()
    
    def evaluate(self, test_loader, iou_threshold=0.5, conf_threshold=0.5):
        """
        Evaluate model on test set
        
        Args:
            test_loader: Test data loader
            iou_threshold: IoU threshold for matching
            conf_threshold: Confidence threshold
        
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        
        print("Evaluating model...")
        with torch.no_grad():
            for images, targets in tqdm(test_loader):
                images = [img.to(self.device) for img in images]
                
                predictions = self.model(images)
                
                all_predictions.extend(predictions)
                all_targets.extend(targets)
        
        # Calculate metrics
        metrics = self._calculate_detection_metrics(
            all_predictions, all_targets, iou_threshold, conf_threshold
        )
        
        return metrics
    
    def _calculate_detection_metrics(self, predictions, targets, 
                                    iou_threshold=0.5, conf_threshold=0.5):
        """Calculate detection metrics (mAP, precision, recall)"""
        
        tp = 0  # True positives
        fp = 0  # False positives
        fn = 0  # False negatives
        
        for pred, target in zip(predictions, targets):
            # Filter predictions by confidence
            keep = pred['scores'] > conf_threshold
            pred_boxes = pred['boxes'][keep].cpu().numpy()
            pred_scores = pred['scores'][keep].cpu().numpy()
            
            target_boxes = target['boxes'].cpu().numpy()
            
            matched_targets = set()
            
            # Match predictions to targets
            for pred_box in pred_boxes:
                best_iou = 0
                best_target_idx = -1
                
                for idx, target_box in enumerate(target_boxes):
                    if idx in matched_targets:
                        continue
                    
                    iou = self._calculate_iou(pred_box, target_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_target_idx = idx
                
                if best_iou >= iou_threshold:
                    tp += 1
                    matched_targets.add(best_target_idx)
                else:
                    fp += 1
            
            # Count unmatched targets as false negatives
            fn += len(target_boxes) - len(matched_targets)
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
        
        return metrics
    
    def _calculate_iou(self, box1, box2):
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
    
    def export_to_onnx(self, output_path='retinanet_fracture.onnx', 
                       img_size=640):
        """
        Export model to ONNX format
        
        Args:
            output_path: Path to save ONNX model
            img_size: Input image size
        """
        self.model.eval()
        
        dummy_input = torch.randn(1, 3, img_size, img_size).to(self.device)
        
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        print(f"✓ Model exported to {output_path}")


# Custom Dataset for RetinaNet (COCO format)
class FractureDetectionDataset(torch.utils.data.Dataset):
    """
    Custom dataset for fracture detection
    
    Expected annotation format:
    {
        'image_id': 'image_001.jpg',
        'boxes': [[x1, y1, x2, y2], ...],  # List of bounding boxes
        'labels': [1, 1, ...],              # List of labels (1 for fracture)
    }
    """
    
    def __init__(self, image_dir, annotations, transforms=None):
        """
        Args:
            image_dir: Directory containing images
            annotations: List of annotation dictionaries
            transforms: Albumentations transforms
        """
        self.image_dir = image_dir
        self.annotations = annotations
        self.transforms = transforms
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        # Load image
        img_info = self.annotations[idx]
        img_path = os.path.join(self.image_dir, img_info['image_id'])
        
        # Read image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get bounding boxes and labels
        boxes = np.array(img_info['boxes'], dtype=np.float32)
        labels = np.array(img_info['labels'], dtype=np.int64)
        
        # Apply transforms if provided
        if self.transforms:
            import albumentations as A
            transformed = self.transforms(
                image=image,
                bboxes=boxes,
                labels=labels
            )
            image = transformed['image']
            boxes = np.array(transformed['bboxes'])
            labels = np.array(transformed['labels'])
        else:
            # Convert to tensor
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # Convert boxes to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Create target dictionary
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx])
        }
        
        return image, target


# Utility function to create data loaders
def create_detection_dataloaders(train_dir, val_dir, train_annotations, 
                                val_annotations, batch_size=4, num_workers=4):
    """
    Create data loaders for detection
    
    Args:
        train_dir: Training images directory
        val_dir: Validation images directory
        train_annotations: Training annotations
        val_annotations: Validation annotations
        batch_size: Batch size
        num_workers: Number of workers
    
    Returns:
        train_loader, val_loader
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    
    # Training transforms
    train_transform = A.Compose([
        A.Resize(640, 640),
        A.HorizontalFlip(p=0.3),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    
    # Validation transforms
    val_transform = A.Compose([
        A.Resize(640, 640),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    
    # Create datasets
    train_dataset = FractureDetectionDataset(train_dir, train_annotations, train_transform)
    val_dataset = FractureDetectionDataset(val_dir, val_annotations, val_transform)
    
    # Custom collate function for detection
    def collate_fn(batch):
        return tuple(zip(*batch))
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader

