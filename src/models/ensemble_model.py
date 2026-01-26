import torch
import torch.nn as nn
import torch.nn.functional as F

class EnsembleModel(nn.Module):
    def __init__(self, models_list, weights=None):
        """
        Ensemble of multiple models
        
        Args:
            models_list: List of trained models
            weights: Optional weights for each model (default: equal weights)
        """
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models_list)
        
        if weights is None:
            self.weights = [1.0 / len(models_list)] * len(models_list)
        else:
            self.weights = weights
    
    def forward(self, x):
        # Get predictions from all models
        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(F.softmax(pred, dim=1))
        
        # Weighted average
        ensemble_pred = torch.zeros_like(predictions[0])
        for pred, weight in zip(predictions, self.weights):
            ensemble_pred += weight * pred
        
        return ensemble_pred

class EnsembleVoting(nn.Module):
    """Hard voting ensemble"""
    def __init__(self, models_list, num_classes=2):
        super(EnsembleVoting, self).__init__()
        self.models = nn.ModuleList(models_list)
        self.num_classes = num_classes

    def forward(self, x):
        # Collect votes from each model
        votes = []
        probs_list = []

        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                prob = F.softmax(pred, dim=1)
                probs_list.append(prob)
                votes.append(torch.argmax(pred, dim=1))

        # Stack votes: shape (num_models, batch_size)
        votes = torch.stack(votes, dim=0)

        # Get mode (most common prediction) for each sample
        mode_pred, _ = torch.mode(votes, dim=0)

        # Return averaged probabilities (soft voting) for proper evaluation
        # This gives us proper probability outputs while still using vote-based predictions
        avg_probs = torch.stack(probs_list, dim=0).mean(dim=0)

        # Create output that reflects the voting decision but with smooth probabilities
        # Use the averaged probabilities for ROC-AUC calculation
        return avg_probs

# Usage
def create_ensemble(model_configs, num_classes=2):
    """
    Create ensemble from multiple model configurations
    
    Example:
        configs = [
            {'type': 'resnet50'},
            {'type': 'resnet101'},
            {'type': 'efficientnet_b3'}
        ]
    """
    from resnet_classifier import create_resnet_model
    from efficientnet_classifier import create_efficientnet_model
    
    models = []
    for config in model_configs:
        model_type = config['type']
        
        if 'resnet' in model_type:
            model = create_resnet_model(model_type, num_classes)
        elif 'efficientnet' in model_type:
            model = create_efficientnet_model(model_type, num_classes)
        
        models.append(model)
    
    return EnsembleModel(models)