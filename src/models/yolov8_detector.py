from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

class YOLOv8FractureDetector:
    def __init__(self, model_size='n'):
        """
        Initialize YOLOv8 model
        
        Args:
            model_size: 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (xlarge)
        """
        self.model = YOLO(f'yolov8{model_size}.pt')
        self.model_size = model_size
    
    def prepare_dataset(self, data_yaml_path):
        """
        Prepare dataset in YOLO format
        
        Dataset structure should be:
        dataset/
            images/
                train/
                val/
                test/
            labels/
                train/
                val/
                test/
            data.yaml
        
        data.yaml example:
```
        path: /path/to/dataset
        train: images/train
        val: images/val
        test: images/test
        
        nc: 1  # number of classes
        names: ['fracture']
```
        """
        self.data_yaml = data_yaml_path
    
    def train(self, epochs=100, img_size=640, batch_size=16):
        """Train YOLOv8 model"""
        results = self.model.train(
            data=self.data_yaml,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            patience=20,
            save=True,
            device=0,  # GPU device
            workers=4,  # Reduced for Windows compatibility
            project='runs/detect',
            name='fracture_detection',
            exist_ok=True,
            pretrained=True,
            optimizer='Adam',
            lr0=0.001,
            weight_decay=0.0005,
            augment=True,
            mosaic=0.5,  # Reduced for faster training
            mixup=0.0,   # Disabled for speed
            copy_paste=0.0,  # Disabled for speed
        )
        return results
    
    def predict(self, image_path, conf_threshold=0.5):
        """Predict fractures in an image"""
        results = self.model.predict(
            source=image_path,
            conf=conf_threshold,
            save=True,
            project='predictions',
            name='fracture_detection'
        )
        return results
    
    def predict_batch(self, image_dir, conf_threshold=0.5):
        """Predict on multiple images"""
        results = self.model.predict(
            source=image_dir,
            conf=conf_threshold,
            save=True,
            project='predictions',
            name='batch_detection'
        )
        return results
    
    def validate(self):
        """Validate model on test set"""
        metrics = self.model.val()
        return metrics
    
    def export_model(self, format='onnx'):
        """Export model to different formats"""
        self.model.export(format=format)

# Usage example
if __name__ == '__main__':
    # Initialize detector
    detector = YOLOv8FractureDetector(model_size='m')
    
    # Prepare dataset
    detector.prepare_dataset('data/fracture_dataset/data.yaml')
    
    # Train
    results = detector.train(epochs=100, img_size=640, batch_size=16)
    
    # Validate
    metrics = detector.validate()
    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP50-95: {metrics.box.map}")
    
    # Predict
    results = detector.predict('test_image.jpg', conf_threshold=0.5)