import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights, ResNet101_Weights

class ResNetClassifier(nn.Module):
    def __init__(self, model_name='resnet50', num_classes=2, pretrained=True):
        super(ResNetClassifier, self).__init__()

        # Load pretrained ResNet with modern weights API
        if model_name == 'resnet50':
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
        elif model_name == 'resnet101':
            weights = ResNet101_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet101(weights=weights)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Get number of features from last layer
        num_features = self.backbone.fc.in_features
        
        # Replace the final fully connected layer
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

# Usage example
def create_resnet_model(model_name='resnet50', num_classes=2):
    model = ResNetClassifier(model_name, num_classes, pretrained=True)
    return model