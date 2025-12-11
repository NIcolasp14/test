"""
Training Module for HGT Model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
import numpy as np
from typing import Dict, Optional, Tuple
from tqdm import tqdm
import os
import json

import config_hgt as config
from model_hgt import HGTClassifier, create_loss_function, compute_class_weights


class Trainer:
    """Train HGT model with early stopping and checkpointing"""
    
    def __init__(
        self,
        model: HGTClassifier,
        device: torch.device = config.DEVICE,
        learning_rate: float = config.LEARNING_RATE,
        weight_decay: float = config.WEIGHT_DECAY,
        verbose: bool = True
    ):
        self.model = model.to(device)
        self.device = device
        self.verbose = verbose
        
        # Optimizer
        if config.OPTIMIZER == "adam":
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif config.OPTIMIZER == "adamw":
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {config.OPTIMIZER}")
        
        # Learning rate scheduler
        self.scheduler = None
        if config.USE_LR_SCHEDULER:
            if config.SCHEDULER_TYPE == "cosine":
                self.scheduler = CosineAnnealingLR(
                    self.optimizer,
                    T_max=config.T_MAX
                )
            elif config.SCHEDULER_TYPE == "step":
                self.scheduler = StepLR(
                    self.optimizer,
                    step_size=config.SCHEDULER_STEP_SIZE,
                    gamma=config.SCHEDULER_FACTOR
                )
            elif config.SCHEDULER_TYPE == "plateau":
                self.scheduler = ReduceLROnPlateau(
                    self.optimizer,
                    mode='max',
                    factor=config.SCHEDULER_FACTOR,
                    patience=config.SCHEDULER_PATIENCE,
                    verbose=self.verbose
                )
        
        # Loss function (will be set during training)
        self.criterion = None
        
        # Early stopping
        self.best_metric = -np.inf
        self.patience_counter = 0
        self.best_model_state = None
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
    
    def setup_loss_function(self, train_labels: torch.Tensor):
        """
        Setup loss function with class weights
        
        Args:
            train_labels: Training labels for computing class weights
        """
        if config.USE_CLASS_WEIGHTS:
            class_weights = compute_class_weights(train_labels)
            class_weights = class_weights.to(self.device)
            if self.verbose:
                print(f"Class weights: {class_weights.cpu().numpy()}")
        else:
            class_weights = None
        
        self.criterion = create_loss_function(class_weights)
    
    def train_epoch(self, hetero_data) -> Tuple[float, Dict]:
        """
        Train for one epoch
        
        Args:
            hetero_data: HeteroData object with train_mask set
        
        Returns:
            Tuple of (average loss, metrics dict)
        """
        self.model.train()
        
        # Move data to device
        hetero_data = hetero_data.to(self.device)
        
        # Get masks and labels
        train_mask = hetero_data['patient'].train_mask
        labels = hetero_data['patient'].y
        
        # Forward pass
        self.optimizer.zero_grad()
        logits = self.model(
            hetero_data.x_dict,
            hetero_data.edge_index_dict
        )
        
        # Compute loss (only on training nodes)
        loss = self.criterion(logits[train_mask], labels[train_mask])
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if config.GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                config.GRAD_CLIP
            )
        
        self.optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            train_metrics = self._compute_metrics(
                logits[train_mask],
                labels[train_mask]
            )
        
        return loss.item(), train_metrics
    
    @torch.no_grad()
    def evaluate(self, hetero_data, mask_name='val') -> Tuple[float, Dict]:
        """
        Evaluate model
        
        Args:
            hetero_data: HeteroData object
            mask_name: 'val' or 'test'
        
        Returns:
            Tuple of (average loss, metrics dict)
        """
        self.model.eval()
        
        # Move data to device
        hetero_data = hetero_data.to(self.device)
        
        # Get mask and labels
        if mask_name == 'val':
            mask = hetero_data['patient'].val_mask
        elif mask_name == 'test':
            mask = hetero_data['patient'].test_mask
        else:
            raise ValueError(f"Unknown mask: {mask_name}")
        
        labels = hetero_data['patient'].y
        
        # Forward pass
        logits = self.model(
            hetero_data.x_dict,
            hetero_data.edge_index_dict
        )
        
        # Compute loss
        loss = self.criterion(logits[mask], labels[mask])
        
        # Compute metrics
        metrics = self._compute_metrics(logits[mask], labels[mask])
        
        return loss.item(), metrics
    
    def _compute_metrics(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute classification metrics
        
        Args:
            logits: Model predictions (N, num_classes)
            labels: True labels (N,)
        
        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import (
            accuracy_score, balanced_accuracy_score,
            f1_score, precision_score, recall_score,
            roc_auc_score, confusion_matrix
        )
        
        # Convert to numpy
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        metrics = {}
        
        # Accuracy
        metrics['accuracy'] = accuracy_score(labels_np, preds)
        metrics['balanced_accuracy'] = balanced_accuracy_score(labels_np, preds)
        
        # F1, Precision, Recall
        metrics['f1_macro'] = f1_score(labels_np, preds, average='macro', zero_division=0)
        metrics['f1_weighted'] = f1_score(labels_np, preds, average='weighted', zero_division=0)
        metrics['precision_macro'] = precision_score(labels_np, preds, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(labels_np, preds, average='macro', zero_division=0)
        
        # AUROC (handle multi-class)
        try:
            metrics['auroc_ovr'] = roc_auc_score(
                labels_np, probs, multi_class='ovr', average='macro'
            )
        except:
            metrics['auroc_ovr'] = 0.0
        
        return metrics
    
    def train(
        self,
        hetero_data,
        num_epochs: int = config.NUM_EPOCHS,
        patience: int = config.EARLY_STOPPING_PATIENCE,
        min_delta: float = config.EARLY_STOPPING_MIN_DELTA,
        save_path: Optional[str] = None
    ) -> Dict:
        """
        Train model with early stopping
        
        Args:
            hetero_data: HeteroData object
            num_epochs: Number of epochs
            patience: Early stopping patience
            min_delta: Minimum improvement for early stopping
            save_path: Path to save best model
        
        Returns:
            Training history
        """
        # Setup loss function
        train_labels = hetero_data['patient'].y[hetero_data['patient'].train_mask]
        self.setup_loss_function(train_labels)
        
        if self.verbose:
            print("\n" + "=" * 80)
            print("Training HGT Model")
            print("=" * 80)
            print(f"Epochs: {num_epochs}")
            print(f"Learning rate: {config.LEARNING_RATE}")
            print(f"Optimizer: {config.OPTIMIZER}")
            print(f"Early stopping patience: {patience}")
        
        # Training loop
        for epoch in range(num_epochs):
            # Train
            train_loss, train_metrics = self.train_epoch(hetero_data)
            
            # Validate
            val_loss, val_metrics = self.evaluate(hetero_data, mask_name='val')
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_metrics'].append(train_metrics)
            self.history['val_metrics'].append(val_metrics)
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics[config.PRIMARY_METRIC])
                else:
                    self.scheduler.step()
            
            # Print progress
            if self.verbose and (epoch + 1) % config.LOG_INTERVAL == 0:
                print(f"\nEpoch {epoch + 1}/{num_epochs}")
                print(f"  Train Loss: {train_loss:.4f} | "
                      f"F1: {train_metrics['f1_macro']:.4f} | "
                      f"Acc: {train_metrics['accuracy']:.4f}")
                print(f"  Val Loss: {val_loss:.4f} | "
                      f"F1: {val_metrics['f1_macro']:.4f} | "
                      f"Acc: {val_metrics['accuracy']:.4f}")
                
                if self.scheduler is not None:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(f"  Learning rate: {current_lr:.6f}")
            
            # Early stopping check
            if config.USE_EARLY_STOPPING:
                current_metric = val_metrics[config.EARLY_STOPPING_METRIC]
                
                if current_metric > self.best_metric + min_delta:
                    self.best_metric = current_metric
                    self.patience_counter = 0
                    self.best_model_state = self.model.state_dict().copy()
                    
                    # Save best model
                    if save_path is not None:
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.best_model_state,
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'metrics': val_metrics,
                            'history': self.history
                        }, save_path)
                        
                        if self.verbose:
                            print(f"  → Best model saved (metric: {current_metric:.4f})")
                else:
                    self.patience_counter += 1
                    
                    if self.patience_counter >= patience:
                        if self.verbose:
                            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                        break
        
        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            if self.verbose:
                print(f"\nRestored best model (metric: {self.best_metric:.4f})")
        
        return self.history
    
    @torch.no_grad()
    def predict(self, hetero_data, mask_name='test') -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions
        
        Args:
            hetero_data: HeteroData object
            mask_name: Mask to use for prediction
        
        Returns:
            Tuple of (predictions, probabilities)
        """
        self.model.eval()
        
        hetero_data = hetero_data.to(self.device)
        
        # Get mask
        if mask_name == 'test':
            mask = hetero_data['patient'].test_mask
        elif mask_name == 'val':
            mask = hetero_data['patient'].val_mask
        elif mask_name == 'train':
            mask = hetero_data['patient'].train_mask
        else:
            raise ValueError(f"Unknown mask: {mask_name}")
        
        # Forward pass
        logits = self.model(
            hetero_data.x_dict,
            hetero_data.edge_index_dict
        )
        
        # Get predictions and probabilities
        probs = torch.softmax(logits[mask], dim=1).cpu().numpy()
        preds = logits[mask].argmax(dim=1).cpu().numpy()
        
        return preds, probs
    
    def save_history(self, save_path: str):
        """Save training history to JSON"""
        # Convert numpy arrays to lists for JSON serialization
        history_json = {
            'train_loss': self.history['train_loss'],
            'val_loss': self.history['val_loss'],
            'train_metrics': [
                {k: float(v) for k, v in m.items()}
                for m in self.history['train_metrics']
            ],
            'val_metrics': [
                {k: float(v) for k, v in m.items()}
                for m in self.history['val_metrics']
            ]
        }
        
        with open(save_path, 'w') as f:
            json.dump(history_json, f, indent=2)
        
        if self.verbose:
            print(f"Training history saved to {save_path}")


def test_trainer():
    """Test trainer module"""
    print("Testing Trainer...")
    
    from model_hgt import HGTClassifier
    
    # Create dummy model
    node_feature_dims = {
        'patient': 20,
        'diagnosis': 32,
        'procedure': 32,
        'provider': 8,
        'hospital': 8
    }
    
    model = HGTClassifier(
        node_types=list(node_feature_dims.keys()),
        edge_types=[
            ('patient', 'has_diagnosis', 'diagnosis'),
            ('diagnosis', 'diagnosed_in', 'patient')
        ],
        node_feature_dims=node_feature_dims,
        hidden_dim=64,
        num_layers=2,
        num_heads=2,
        num_classes=3
    )
    
    # Create trainer
    trainer = Trainer(model, device=torch.device('cpu'), verbose=True)
    
    print("\nTrainer test completed successfully!")
    
    return trainer


if __name__ == "__main__":
    test_trainer()
