import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm
import numpy as np
import os

class Trainer:
    def __init__(self, model, train_loader, val_loader, device, config, model_name='model'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.model_name = model_name  # For saving with model-specific names

        # Loss function (can use weighted loss for imbalanced data)
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
    
    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='Validation'):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def save_checkpoint(self, epoch, model_name='best_model.pth'):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
        }
        # Create directory if it doesn't exist
        os.makedirs(self.config.MODEL_DIR, exist_ok=True)
        save_path = os.path.join(self.config.MODEL_DIR, model_name)
        torch.save(checkpoint, save_path)
        print(f'Model saved to {save_path}')
    
    def train(self, epochs):
        print(f"Training for {epochs} epochs...")
        
        for epoch in range(epochs):
            print(f'\nEpoch {epoch + 1}/{epochs}')
            print('-' * 60)
            
            # Training phase
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # Validation phase
            val_loss, val_acc = self.validate()
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Save best model with model-specific name
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                save_name = f'{self.model_name}_best_model.pth' if self.model_name != 'resnet50' else 'best_model.pth'
                self.save_checkpoint(epoch, save_name)
                print(f'✓ New best model saved! (Val Acc: {val_acc:.2f}%)')

            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, f'{self.model_name}_checkpoint_epoch_{epoch+1}.pth')
        
        print('\nTraining completed!')
        print(f'Best Val Accuracy: {self.best_val_acc:.2f}%')
        print(f'Best Val Loss: {self.best_val_loss:.4f}')
        
        return self.train_losses, self.val_losses, self.train_accs, self.val_accs

# Main training script
if __name__ == '__main__':
    import sys
    sys.path.append('..')
    from utils.config import Config
    from data.data_loader import create_data_loaders
    from models.resnet_classifier import create_resnet_model
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    config = Config()
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        config.TRAIN_DIR,
        config.VAL_DIR,
        config.TEST_DIR,
        batch_size=config.BATCH_SIZE,
        img_size=config.IMG_SIZE
    )
    
    # Create model
    model = create_resnet_model('resnet50', num_classes=config.NUM_CLASSES)
    
    # Train
    trainer = Trainer(model, train_loader, val_loader, device, config)
    history = trainer.train(config.EPOCHS)