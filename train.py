"""
Training Pipeline for ED Utilization Prediction
Handles training loop, optimization, and checkpointing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from pathlib import Path
import pickle
import numpy as np
from tqdm import tqdm

import config
from evaluate import evaluate_model, MetricsTracker, print_metrics


class EarlyStopper:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=config.PATIENCE, min_delta=config.MIN_DELTA):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_metric):
        """
        Check if should stop training
        
        Args:
            val_metric: Validation metric (higher is better)
        """
        if self.best_score is None:
            self.best_score = val_metric
        elif val_metric < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_metric
            self.counter = 0
        
        return self.early_stop


def create_optimizer(model, optimizer_name=config.OPTIMIZER, lr=config.LEARNING_RATE):
    """Create optimizer"""
    if optimizer_name.lower() == 'adam':
        return Adam(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    elif optimizer_name.lower() == 'adamw':
        return AdamW(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    elif optimizer_name.lower() == 'sgd':
        return SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=config.WEIGHT_DECAY)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def create_scheduler(optimizer, scheduler_type=config.SCHEDULER_TYPE):
    """Create learning rate scheduler"""
    if not config.USE_SCHEDULER:
        return None
    
    if scheduler_type == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=config.T_MAX)
    elif scheduler_type == 'step':
        return StepLR(optimizer, step_size=10, gamma=0.5)
    elif scheduler_type == 'plateau':
        return ReduceLROnPlateau(optimizer, mode='max', patience=config.SCHEDULER_PATIENCE)
    else:
        return None


def compute_loss(delta_pred, binary_logits, delta_true, binary_true, censored_mask=None):
    """
    Compute multi-task loss: MAE + BCE
    
    Args:
        delta_pred: Predicted time-to-event (batch_size, 1)
        binary_logits: Binary classification logits (batch_size, 1)
        delta_true: True time-to-event (batch_size,)
        binary_true: True binary labels (batch_size,)
        censored_mask: Boolean mask for censored samples
    
    Returns:
        total_loss, mae_loss, bce_loss
    """
    delta_pred = delta_pred.squeeze()
    binary_logits = binary_logits.squeeze()
    
    # MAE loss (only on observed samples)
    if censored_mask is not None:
        observed_mask = ~censored_mask & (delta_true > 0)
    else:
        observed_mask = delta_true > 0
    
    if observed_mask.sum() > 0:
        mae_loss = F.l1_loss(delta_pred[observed_mask], delta_true[observed_mask])
    else:
        mae_loss = torch.tensor(0.0, device=delta_pred.device)
    
    # BCE loss for binary classification
    if config.USE_MULTI_TASK:
        bce_loss = F.binary_cross_entropy_with_logits(
            binary_logits,
            binary_true.float()
        )
    else:
        bce_loss = torch.tensor(0.0, device=delta_pred.device)
    
    # Combined loss
    total_loss = config.LAMBDA_MAE * mae_loss + config.LAMBDA_BCE * bce_loss
    
    return total_loss, mae_loss, bce_loss


def train_epoch(model, train_data, optimizer, device, epoch):
    """
    Train for one epoch
    
    Args:
        model: Model to train
        train_data: Dictionary with training data
        optimizer: Optimizer
        device: torch device
        epoch: Current epoch number
    
    Returns:
        avg_loss: Average training loss
    """
    model.train()
    total_loss = 0
    total_mae = 0
    total_bce = 0
    n_batches = 0
    
    # For simplified version, we'll train on all patient embeddings at once
    # In production, you'd use mini-batch sampling
    
    patient_nodes = torch.tensor(
        list(train_data['node_id_maps']['patient'].values()),
        dtype=torch.long
    ).to(device)
    
    node_features = train_data['node_features']['patient'].to(device)
    
    # Get labels
    labels_df = train_data['labels']
    patient_ids = [train_data['reverse_node_maps']['patient'][i] for i in range(len(patient_nodes))]
    
    # Match labels to patient nodes
    delta_targets = []
    binary_targets = []
    censored_flags = []
    
    for pid in patient_ids:
        patient_labels = labels_df[labels_df['patient_id'] == pid]
        if len(patient_labels) > 0:
            # Use first label for simplicity
            label = patient_labels.iloc[0]
            delta_targets.append(max(0, label['days_to_next_ed']))
            binary_targets.append(label['has_next_ed_30d'])
            censored_flags.append(label['days_to_next_ed'] < 0)
        else:
            delta_targets.append(0.0)
            binary_targets.append(0)
            censored_flags.append(True)
    
    delta_targets = torch.tensor(delta_targets, dtype=torch.float32).to(device)
    binary_targets = torch.tensor(binary_targets, dtype=torch.float32).to(device)
    censored_mask = torch.tensor(censored_flags, dtype=torch.bool).to(device)
    
    # Forward pass
    optimizer.zero_grad()
    
    if hasattr(model, 'get_embeddings') and hasattr(model, 'pred_head'):
        # TGN-style model
        embeddings = model.get_embeddings(patient_nodes, node_features)
        delta_pred, binary_logits = model.pred_head(embeddings)
    else:
        # Direct forward
        delta_pred, binary_logits = model(patient_nodes, node_features)
    
    # Compute loss
    loss, mae_loss, bce_loss = compute_loss(
        delta_pred, binary_logits, delta_targets, binary_targets, censored_mask
    )
    
    # Backward pass
    loss.backward()
    
    # Gradient clipping
    if config.GRAD_CLIP > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
    
    optimizer.step()
    
    total_loss += loss.item()
    total_mae += mae_loss.item()
    total_bce += bce_loss.item()
    n_batches += 1
    
    avg_loss = total_loss / n_batches if n_batches > 0 else 0
    avg_mae = total_mae / n_batches if n_batches > 0 else 0
    avg_bce = total_bce / n_batches if n_batches > 0 else 0
    
    if config.VERBOSE and epoch % config.LOG_INTERVAL == 0:
        print(f"  Train Loss: {avg_loss:.4f} (MAE: {avg_mae:.4f}, BCE: {avg_bce:.4f})")
    
    return avg_loss


def validate(model, val_data, device):
    """
    Validate model
    
    Args:
        model: Model to validate
        val_data: Validation data dictionary
        device: torch device
    
    Returns:
        metrics: Dictionary of validation metrics
    """
    model.eval()
    
    # Prepare data
    patient_nodes = torch.tensor(
        list(val_data['node_id_maps']['patient'].values()),
        dtype=torch.long
    ).to(device)
    
    node_features = val_data['node_features']['patient'].to(device)
    
    # Get labels
    labels_df = val_data['labels']
    patient_ids = [val_data['reverse_node_maps']['patient'][i] for i in range(len(patient_nodes))]
    
    delta_targets = []
    binary_targets = []
    censored_flags = []
    
    for pid in patient_ids:
        patient_labels = labels_df[labels_df['patient_id'] == pid]
        if len(patient_labels) > 0:
            label = patient_labels.iloc[0]
            delta_targets.append(max(0, label['days_to_next_ed']))
            binary_targets.append(label['has_next_ed_30d'])
            censored_flags.append(label['days_to_next_ed'] < 0)
        else:
            delta_targets.append(0.0)
            binary_targets.append(0)
            censored_flags.append(True)
    
    delta_targets = torch.tensor(delta_targets, dtype=torch.float32)
    binary_targets = torch.tensor(binary_targets, dtype=torch.float32)
    censored_mask = torch.tensor(censored_flags, dtype=torch.bool)
    
    # Evaluate
    metrics = evaluate_model(
        model, patient_nodes, node_features,
        delta_targets, binary_targets, censored_mask, device
    )
    
    return metrics


def train_model(model, train_data, val_data, model_name, device=config.DEVICE):
    """
    Full training pipeline
    
    Args:
        model: Model to train
        train_data: Training data dictionary
        val_data: Validation data dictionary
        model_name: Name of the model ('TGN', 'TGAT', 'HGT')
        device: torch device
    
    Returns:
        model: Trained model
        metrics_tracker: Metrics tracking object
    """
    print(f"\n{'='*80}")
    print(f"TRAINING {model_name}")
    print(f"{'='*80}")
    
    # CRITICAL DIAGNOSTIC: Check training data quality
    labels_df = train_data['labels']
    uncensored = (labels_df['days_to_next_ed'] >= 0).sum()
    positive_30d = labels_df['has_next_ed_30d'].sum()
    
    print(f"\n  📊 Training Data Quality Check:")
    print(f"    Total patients: {len(labels_df)}")
    print(f"    Uncensored samples (have next ED): {uncensored} ({100*uncensored/len(labels_df):.1f}%)")
    print(f"    Positive samples (ED within 30d): {positive_30d} ({100*positive_30d/len(labels_df):.1f}%)")
    
    if uncensored == 0:
        print(f"\n    ⚠️  CRITICAL: NO UNCENSORED SAMPLES TO LEARN FROM!")
        print(f"    ⚠️  MAE will be 0.0 and model will only learn from BCE loss")
        print(f"    ⚠️  This indicates a data preprocessing issue.")
    elif uncensored < 10:
        print(f"\n    ⚠️  WARNING: Very few uncensored samples ({uncensored})")
        print(f"    ⚠️  Training may be unstable")
    
    if positive_30d == 0:
        print(f"\n    ⚠️  WARNING: No positive samples for 30-day prediction")
        print(f"    ⚠️  BCE loss may not be meaningful")
    
    model = model.to(device)
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model)
    scheduler = create_scheduler(optimizer)
    
    # Early stopping
    early_stopper = EarlyStopper() if config.EARLY_STOPPING else None
    
    # Metrics tracker
    metrics_tracker = MetricsTracker()
    
    # Training loop
    best_val_metric = -float('inf')
    best_epoch = 0
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        if config.VERBOSE:
            print(f"\nEpoch {epoch}/{config.NUM_EPOCHS}")
        
        # Train
        train_loss = train_epoch(model, train_data, optimizer, device, epoch)
        
        # Validate
        if epoch % config.SAVE_INTERVAL == 0:
            val_metrics = validate(model, val_data, device)
            metrics_tracker.update('val', val_metrics, epoch)
            
            if config.VERBOSE:
                print(f"  Val {config.PRIMARY_METRIC}: {val_metrics[config.PRIMARY_METRIC]:.4f}")
            
            # Save best model
            if val_metrics[config.PRIMARY_METRIC] > best_val_metric:
                best_val_metric = val_metrics[config.PRIMARY_METRIC]
                best_epoch = epoch
                
                # Save checkpoint
                model_dir = Path(config.MODEL_SAVE_DIR)
                model_dir.mkdir(exist_ok=True)
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': val_metrics,
                }
                
                torch.save(checkpoint, model_dir / f'{model_name}_best.pt')
                
                if config.VERBOSE:
                    print(f"  ✓ Saved best model (epoch {epoch})")
            
            # Early stopping check
            if early_stopper and early_stopper(val_metrics[config.PRIMARY_METRIC]):
                print(f"\n  Early stopping triggered at epoch {epoch}")
                break
        
        # Learning rate scheduling
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_metrics[config.PRIMARY_METRIC])
            else:
                scheduler.step()
    
    print(f"\n  Best validation {config.PRIMARY_METRIC}: {best_val_metric:.4f} (epoch {best_epoch})")
    
    # Load best model (PyTorch 2.6+ requires weights_only=False for checkpoints with numpy)
    checkpoint = torch.load(
        Path(config.MODEL_SAVE_DIR) / f'{model_name}_best.pt',
        map_location=device,
        weights_only=False  # Required for PyTorch 2.6+ with numpy objects
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, metrics_tracker


if __name__ == "__main__":
    print("Training module loaded successfully")
    print(f"  Device: {config.DEVICE}")
    print(f"  Learning rate: {config.LEARNING_RATE}")
    print(f"  Num epochs: {config.NUM_EPOCHS}")
    print(f"  Batch size: {config.BATCH_SIZE}")


