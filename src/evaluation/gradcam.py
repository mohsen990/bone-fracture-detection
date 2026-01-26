import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        """
        Initialize Grad-CAM
        
        Args:
            model: PyTorch model
            target_layer: Layer to compute gradients (e.g., model.layer4 for ResNet)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Hook to save forward activations"""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward gradients"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image, target_class=None):
        """
        Generate Class Activation Map
        
        Args:
            input_image: Input tensor (1, C, H, W)
            target_class: Target class index (if None, use predicted class)
        
        Returns:
            cam: Class activation map
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_image)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        class_loss = output[0, target_class]
        class_loss.backward()
        
        # Get gradients and activations
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Calculate weights (global average pooling of gradients)
        weights = gradients.mean(dim=(1, 2))  # (C,)

        # Weighted combination of activation maps
        # Create cam on the same device as activations
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU (only positive influence)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / cam.max()
        
        return cam.cpu().numpy()
    
    def visualize(self, input_image, original_image, target_class=None, 
                 alpha=0.4, colormap=cv2.COLORMAP_JET):
        """
        Visualize Grad-CAM overlay on original image
        
        Args:
            input_image: Preprocessed input tensor (1, C, H, W)
            original_image: Original image (H, W, C) numpy array
            target_class: Target class
            alpha: Overlay transparency
            colormap: OpenCV colormap
        
        Returns:
            superimposed_img: Visualization with CAM overlay
        """
        # Generate CAM
        cam = self.generate_cam(input_image, target_class)
        
        # Resize CAM to match original image
        h, w = original_image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        
        # Convert CAM to heatmap
        cam_resized = np.uint8(255 * cam_resized)
        heatmap = cv2.applyColorMap(cam_resized, colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Ensure original image is RGB
        if len(original_image.shape) == 2:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        
        # Normalize original image to 0-255
        if original_image.max() <= 1:
            original_image = (original_image * 255).astype(np.uint8)
        
        # Superimpose heatmap on original image
        superimposed_img = heatmap * alpha + original_image * (1 - alpha)
        superimposed_img = superimposed_img.astype(np.uint8)
        
        return superimposed_img, heatmap

def visualize_gradcam_batch(model, images, original_images, target_layer, 
                           class_names, device, save_path='gradcam_results.png'):
    """Visualize Grad-CAM for a batch of images"""
    
    gradcam = GradCAM(model, target_layer)
    
    num_images = min(len(images), 8)
    fig, axes = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))
    
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(num_images):
        # Get prediction
        model.eval()
        with torch.no_grad():
            output = model(images[idx:idx+1].to(device))
            prob = F.softmax(output, dim=1)
            pred_class = output.argmax(dim=1).item()
            confidence = prob[0, pred_class].item()
        
        # Generate Grad-CAM
        superimposed, heatmap = gradcam.visualize(
            images[idx:idx+1].to(device),
            original_images[idx]
        )
        
        # Plot original image
        axes[idx, 0].imshow(original_images[idx])
        axes[idx, 0].set_title('Original X-ray', fontsize=12, fontweight='bold')
        axes[idx, 0].axis('off')
        
        # Plot heatmap
        axes[idx, 1].imshow(heatmap)
        axes[idx, 1].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
        axes[idx, 1].axis('off')
        
        # Plot overlay
        axes[idx, 2].imshow(superimposed)
        title = f'Prediction: {class_names[pred_class]}\nConfidence: {confidence*100:.1f}%'
        axes[idx, 2].set_title(title, fontsize=12, fontweight='bold')
        axes[idx, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# Example usage for different models
def get_target_layer(model, model_type='resnet50'):
    """Get the appropriate target layer for Grad-CAM"""
    if 'resnet' in model_type.lower():
        return model.backbone.layer4[-1]
    elif 'efficientnet' in model_type.lower():
        return model.backbone.features[-1]
    elif 'densenet' in model_type.lower():
        return model.backbone.features[-1]
    else:
        raise ValueError(f"Unknown model type: {model_type}")