import sys
sys.path.append('..')

import torch
import cv2
import numpy as np
from PIL import Image
from src.utils.config import Config
from src.models.resnet_classifier import create_resnet_model
from src.evaluation.gradcam import GradCAM, visualize_gradcam_batch, get_target_layer
from src.data.preprocessing import get_val_transforms

def visualize_single_image(model_path, image_path, model_type='resnet50'):
    """Visualize Grad-CAM for a single image"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = Config()
    
    # Load model
    print("Loading model...")
    if 'resnet' in model_type:
        model = create_resnet_model(model_type, num_classes=config.NUM_CLASSES)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load and preprocess image
    print("Processing image...")
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    transform = get_val_transforms(config.IMG_SIZE)
    image_tensor = transform(image=original_image)['image'].unsqueeze(0)
    
    # Get prediction
    with torch.no_grad():
        output = model(image_tensor.to(device))
        prob = torch.softmax(output, dim=1)
        pred_class = output.argmax(dim=1).item()
        confidence = prob[0, pred_class].item()
    
    print(f"Prediction: {config.CLASS_NAMES[pred_class]}")
    print(f"Confidence: {confidence*100:.2f}%")
    
    # Generate Grad-CAM
    print("Generating Grad-CAM...")
    target_layer = get_target_layer(model, model_type)
    gradcam = GradCAM(model, target_layer)
    
    superimposed, heatmap = gradcam.visualize(
        image_tensor.to(device),
        original_image,
        target_class=pred_class
    )
    
    # Display results
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(original_image)
    axes[0].set_title('Original X-ray', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(heatmap)
    axes[1].set_title('Grad-CAM Heatmap', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(superimposed)
    title = f'Prediction: {config.CLASS_NAMES[pred_class]}\nConfidence: {confidence*100:.1f}%'
    axes[2].set_title(title, fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    import os
    os.makedirs('../results/gradcam', exist_ok=True)
    plt.savefig('../results/gradcam/gradcam_single.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("✓ Visualization saved to 'results/gradcam/gradcam_single.png'")

def main():
    import os
    from glob import glob

    # Create results directory
    os.makedirs('../results/gradcam', exist_ok=True)

    # Model path (relative to examples/ directory)
    model_path = '../models/saved/best_model.pth'

    if not os.path.exists(model_path):
        print(f"✗ Model not found: {model_path}")
        print("  Please train a model first using train_single_model.py")
        return

    # Find a test image
    test_dirs = [
        '../data/test/Fractured',
        '../data/test/Normal',
        '../data/val/Fractured',
        '../data/val/Normal',
    ]

    image_path = None
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            images = glob(os.path.join(test_dir, '*.jpg')) + \
                     glob(os.path.join(test_dir, '*.png')) + \
                     glob(os.path.join(test_dir, '*.jpeg'))
            if images:
                image_path = images[0]
                break

    if image_path is None:
        print("✗ No test images found")
        return

    print(f"Using image: {image_path}")
    visualize_single_image(model_path, image_path, model_type='resnet50')


if __name__ == '__main__':
    main()