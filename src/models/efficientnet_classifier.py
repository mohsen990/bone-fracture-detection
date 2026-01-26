import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights, EfficientNet_B3_Weights, EfficientNet_B7_Weights

class EfficientNetClassifier(nn.Module):
    def __init__(self, model_name='efficientnet_b0', num_classes=2, pretrained=True):
        super(EfficientNetClassifier, self).__init__()

        # Load pretrained EfficientNet with modern weights API
        if model_name == 'efficientnet_b0':
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
        elif model_name == 'efficientnet_b3':
            weights = EfficientNet_B3_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b3(weights=weights)
        elif model_name == 'efficientnet_b7':
            weights = EfficientNet_B7_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b7(weights=weights)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Get number of features
        num_features = self.backbone.classifier[1].in_features
        
        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

# Usage
def create_efficientnet_model(model_name='efficientnet_b0', num_classes=2):
    model = EfficientNetClassifier(model_name, num_classes, pretrained=True)
    return model