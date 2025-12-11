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


def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """
    Focal Loss for binary classification with extreme class imbalance
    
    Args:
        logits: Raw model outputs (batch_size,)
        targets: Binary targets (batch_size,)
        alpha: Weight for positive class (default 0.25)
        gamma: Focusing parameter (default 2.0)
    
    Returns:
        Focal loss value
    
    Reference: https://arxiv.org/abs/1708.02002
    """
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    probs = torch.sigmoid(logits)
    
    # p_t = p if y == 1, else 1-p
    p_t = probs * targets + (1 - probs) * (1 - targets)
    
    # Focal term: (1 - p_t)^gamma
    focal_weight = (1 - p_t) ** gamma
    
    # Alpha weighting
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    
    loss = alpha_t * focal_weight * bce_loss
    
    return loss.mean()


def focal_loss_multiclass(logits, targets, alpha=None, gamma=2.0):
    """
    Focal Loss for multi-class classification with class imbalance
    
    Args:
        logits: Raw model outputs (batch_size, num_classes)
        targets: Class labels (batch_size,) with values in [0, num_classes-1]
        alpha: Per-class weights (num_classes,) or None for equal weights
        gamma: Focusing parameter (default 2.0)
    
    Returns:
        Focal loss value
    
    Reference: https://arxiv.org/abs/1708.02002
    """
    # Compute cross-entropy loss (no reduction yet)
    ce_loss = F.cross_entropy(logits, targets, weight=alpha, reduction='none')
    
    # Compute probabilities and pt
    probs = F.softmax(logits, dim=1)
    pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    
    # Focal term: (1 - pt)^gamma
    focal_weight = (1 - pt) ** gamma
    
    # Apply focal weighting
    loss = focal_weight * ce_loss
    
    return loss.mean()


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


def compute_loss_classification(logits, targets, class_weights=None):
    """
    Compute loss for multi-class classification
    
    Args:
        logits: Model outputs (batch_size, num_classes)
        targets: Class labels (batch_size,) with values in [0, num_classes-1]
        class_weights: Per-class weights for imbalance (num_classes,) or None
    
    Returns:
        loss (scalar)
    """
    if config.USE_FOCAL_LOSS:
        # Use focal loss for class imbalance
        loss = focal_loss_multiclass(
            logits,
            targets,
            alpha=class_weights,
            gamma=config.FOCAL_GAMMA
        )
    else:
        # Standard cross-entropy with optional label smoothing
        loss = F.cross_entropy(
            logits,
            targets,
            weight=class_weights,
            label_smoothing=config.LABEL_SMOOTHING if hasattr(config, 'LABEL_SMOOTHING') else 0.0
        )
    
    return loss


def compute_loss(delta_pred, binary_logits, delta_true, binary_true, censored_mask=None, pos_weight=None, use_focal=True):
    """
    Compute multi-task loss for survival/regression: MAE + Focal/Weighted BCE
    
    DEPRECATED: Use compute_loss_classification for classification tasks.
    
    Args:
        delta_pred: Predicted time-to-event (batch_size, 1) - normalized [0, 1]
        binary_logits: Binary classification logits (batch_size, 1)
        delta_true: True time-to-event (batch_size,) - normalized [0, 1] or -1 for censored
        binary_true: True binary labels (batch_size,)
        censored_mask: Boolean mask for censored samples
        pos_weight: Weight for positive class in BCE (for class imbalance) - ignored if use_focal=True
        use_focal: Use focal loss instead of weighted BCE (better for extreme imbalance)
    
    Returns:
        total_loss, mae_loss, bce_loss
    """
    delta_pred = delta_pred.squeeze()
    binary_logits = binary_logits.squeeze()
    
    # MAE loss (only on observed samples)
    # Note: delta_true is now normalized to [0, 1]
    if censored_mask is not None:
        observed_mask = ~censored_mask & (delta_true >= 0)
    else:
        observed_mask = delta_true >= 0
    
    if observed_mask.sum() > 0:
        mae_loss = F.l1_loss(delta_pred[observed_mask], delta_true[observed_mask])
    else:
        mae_loss = torch.tensor(0.0, device=delta_pred.device)
    
    # Classification loss: Focal Loss (better) or Weighted BCE
    if config.USE_MULTI_TASK:
        if use_focal and hasattr(config, 'USE_FOCAL_LOSS') and config.USE_FOCAL_LOSS:
            # Focal Loss for extreme imbalance (5.4% positive class)
            # alpha=0.25 means positive class gets 0.25 weight, negative gets 0.75
            # gamma=2.0 focuses on hard examples
            bce_loss = focal_loss(
                binary_logits,
                binary_true.float(),
                alpha=config.FOCAL_ALPHA,
                gamma=config.FOCAL_GAMMA
            )
        else:
            # Fallback to weighted BCE
            if pos_weight is not None:
                bce_loss = F.binary_cross_entropy_with_logits(
                    binary_logits,
                    binary_true.float(),
                    pos_weight=pos_weight
                )
            else:
                bce_loss = F.binary_cross_entropy_with_logits(
                    binary_logits,
                    binary_true.float()
                )
    else:
        bce_loss = torch.tensor(0.0, device=delta_pred.device)
    
    # Combined loss
    total_loss = config.LAMBDA_MAE * mae_loss + config.LAMBDA_BCE * bce_loss
    
    return total_loss, mae_loss, bce_loss


def train_epoch_classification(model, train_data, optimizer, device, epoch, class_weights=None):
    """
    Train for one epoch - CLASSIFICATION MODE
    
    Args:
        model: Model to train
        train_data: Dictionary with training data (must have 'labels' with 'utilization_class')
        optimizer: Optimizer
        device: torch device
        epoch: Current epoch number
        class_weights: Per-class weights for loss (optional)
    
    Returns:
        avg_loss: Average training loss
    """
    model.train()
    total_loss = 0
    n_batches = 0
    
    labels_df = train_data['labels']
    patients_with_labels = labels_df['patient_id'].unique()
    
    # Get node features
    node_features_full = train_data['node_features']['patient'].to(device)
    
    # Map patient IDs to feature indices
    reverse_node_map = train_data.get('reverse_node_maps', {}).get('patient', {})
    pid_to_feat_idx = {pid: feat_idx for feat_idx, pid in reverse_node_map.items()}
    
    patient_nodes_list = []
    patient_ids = []
    
    for pid in patients_with_labels:
        if pid in pid_to_feat_idx:
            feat_idx = pid_to_feat_idx[pid]
            if feat_idx < len(node_features_full):
                patient_nodes_list.append(feat_idx)
                patient_ids.append(pid)
    
    if len(patient_nodes_list) == 0:
        raise ValueError(f"No valid patient nodes found for training!")
    
    patient_nodes = torch.tensor(patient_nodes_list, dtype=torch.long).to(device)
    node_features = node_features_full[patient_nodes]
    patient_nodes_reindexed = torch.arange(len(patient_nodes), dtype=torch.long).to(device)
    
    # Get utilization class labels
    class_targets = []
    for pid in patient_ids:
        patient_label = labels_df[labels_df['patient_id'] == pid].iloc[0]
        class_targets.append(patient_label['utilization_class'])
    
    class_targets = torch.tensor(class_targets, dtype=torch.long).to(device)
    
    # Forward pass
    optimizer.zero_grad()
    
    if hasattr(model, 'get_embeddings') and hasattr(model, 'pred_head'):
        embeddings = model.get_embeddings(patient_nodes_reindexed, node_features)
        logits = model.pred_head(embeddings)
    else:
        logits = model(patient_nodes_reindexed, node_features)
    
    # Compute classification loss
    if class_weights is not None:
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    else:
        class_weights_tensor = None
    
    loss = compute_loss_classification(logits, class_targets, class_weights_tensor)
    
    # Backward pass
    loss.backward()
    
    if config.GRAD_CLIP > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
    
    optimizer.step()
    
    total_loss += loss.item()
    n_batches += 1
    
    avg_loss = total_loss / n_batches if n_batches > 0 else 0
    
    # Diagnostics
    if config.VERBOSE and epoch % config.LOG_INTERVAL == 0:
        with torch.no_grad():
            pred_classes = torch.argmax(logits, dim=1)
            accuracy = (pred_classes == class_targets).float().mean().item()
            
            # Class distribution
            for c in range(config.NUM_CLASSES):
                pred_count = (pred_classes == c).sum().item()
                true_count = (class_targets == c).sum().item()
                print(f"    Class {c}: {pred_count} pred, {true_count} true")
        
        print(f"  Train Loss: {avg_loss:.4f}, Accuracy: {accuracy*100:.1f}%")
    
    return avg_loss


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
    
    # CRITICAL FIX: Only train on patients that have labels in labels_df
    # Enriched labels create multiple samples per patient, but we need to ensure
    # patient_nodes matches the labels we're using
    
    labels_df = train_data['labels']
    
    # Get unique patient IDs that have labels
    patients_with_labels = labels_df['patient_id'].unique()
    
    # Filter patient nodes to only those with labels
    # CRITICAL: Map patient IDs to their position in node_features, not to original graph node IDs
    patient_nodes_list = []
    patient_ids = []
    
    # Get node features
    node_features_full = train_data['node_features']['patient'].to(device)
    
    # Create mapping from patient_id to index in node_features
    # This handles cases where node_features might be filtered/aligned differently than node_id_map
    reverse_node_map = train_data.get('reverse_node_maps', {}).get('patient', {})
    
    # Build patient_id -> feature_index mapping
    pid_to_feat_idx = {}
    for feat_idx, pid in reverse_node_map.items():
        pid_to_feat_idx[pid] = feat_idx
    
    for pid in patients_with_labels:
        if pid in pid_to_feat_idx:
            # Use the index in node_features, not the original graph node ID
            feat_idx = pid_to_feat_idx[pid]
            # Validate index is in bounds
            if feat_idx < len(node_features_full):
                patient_nodes_list.append(feat_idx)
                patient_ids.append(pid)
    
    if len(patient_nodes_list) == 0:
        raise ValueError(f"No valid patient nodes found for training! Check node_features alignment.")
    
    patient_nodes = torch.tensor(patient_nodes_list, dtype=torch.long).to(device)
    
    # Index only the patients we're using
    node_features = node_features_full[patient_nodes]
    
    # Reset patient_nodes to be 0-indexed for the current batch
    patient_nodes_reindexed = torch.arange(len(patient_nodes), dtype=torch.long).to(device)
    
    # Match labels to patient nodes
    delta_targets = []
    delta_targets_normalized = []
    binary_targets = []
    censored_flags = []
    
    for pid in patient_ids:
        patient_labels = labels_df[labels_df['patient_id'] == pid]
        if len(patient_labels) > 0:
            # Use first label for simplicity (for patients with multiple observation points)
            label = patient_labels.iloc[0]
            delta_targets.append(max(0, label['days_to_next_ed']))
            # Use normalized target for training (better scale)
            delta_targets_normalized.append(label.get('days_to_next_ed_normalized', -1.0))
            binary_targets.append(label['has_next_ed_30d'])
            censored_flags.append(label['days_to_next_ed'] < 0)
        else:
            # This shouldn't happen now, but keep as fallback
            delta_targets.append(0.0)
            delta_targets_normalized.append(-1.0)
            binary_targets.append(0)
            censored_flags.append(True)
    
    delta_targets = torch.tensor(delta_targets, dtype=torch.float32).to(device)
    delta_targets_norm = torch.tensor(delta_targets_normalized, dtype=torch.float32).to(device)
    binary_targets = torch.tensor(binary_targets, dtype=torch.float32).to(device)
    censored_mask = torch.tensor(censored_flags, dtype=torch.bool).to(device)
    
    # Compute class weight for severe imbalance (5.4% positive class)
    num_positive = binary_targets.sum().item()
    num_negative = len(binary_targets) - num_positive
    if num_positive > 0:
        pos_weight = torch.tensor([num_negative / num_positive], device=device)
    else:
        pos_weight = None
    
    # Forward pass
    optimizer.zero_grad()
    
    if hasattr(model, 'get_embeddings') and hasattr(model, 'pred_head'):
        # TGN-style model - use reindexed nodes
        embeddings = model.get_embeddings(patient_nodes_reindexed, node_features)
        delta_pred, binary_logits = model.pred_head(embeddings)
    else:
        # Direct forward - use reindexed nodes
        delta_pred, binary_logits = model(patient_nodes_reindexed, node_features)
    
    # Compute loss (use normalized targets for MAE)
    loss, mae_loss, bce_loss = compute_loss(
        delta_pred, binary_logits, delta_targets_norm, binary_targets, censored_mask, pos_weight
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
        # DIAGNOSTIC: Show predictions vs true labels to check if model is learning correctly
        with torch.no_grad():
            if hasattr(model, 'get_embeddings') and hasattr(model, 'pred_head'):
                embeddings = model.get_embeddings(patient_nodes_reindexed, node_features)
                delta_pred_diag, binary_logits_diag = model.pred_head(embeddings)
            else:
                delta_pred_diag, binary_logits_diag = model(patient_nodes_reindexed, node_features)
            
            binary_probs_diag = torch.sigmoid(binary_logits_diag).squeeze()
            pred_positive_rate = (binary_probs_diag > 0.5).float().mean().item()
            true_positive_rate = binary_targets.mean().item()
            avg_pred_prob = binary_probs_diag.mean().item()
            
        print(f"  Train Loss: {avg_loss:.4f} (MAE: {avg_mae:.4f}, BCE: {avg_bce:.4f})")
        print(f"    📊 Predictions: {pred_positive_rate*100:.1f}% pred positive (true: {true_positive_rate*100:.1f}%), "
              f"avg prob: {avg_pred_prob:.3f}, pos_weight: {pos_weight[0].item() if pos_weight is not None else 1.0:.1f}x")
    
    return avg_loss


def validate_classification(model, val_data, device):
    """
    Validate model - CLASSIFICATION MODE
    
    Args:
        model: Model to validate
        val_data: Validation data dictionary
        device: torch device
    
    Returns:
        predictions: Array of predicted class labels
        true_labels: Array of true class labels
        logits: Array of raw logits for probability-based metrics
    """
    model.eval()
    
    labels_df = val_data['labels']
    patients_with_labels = labels_df['patient_id'].unique()
    
    # Get node features
    node_features_full = val_data['node_features']['patient'].to(device)
    
    # Map patient IDs to feature indices
    reverse_node_map = val_data.get('reverse_node_maps', {}).get('patient', {})
    pid_to_feat_idx = {pid: feat_idx for feat_idx, pid in reverse_node_map.items()}
    
    patient_nodes_list = []
    patient_ids = []
    
    for pid in patients_with_labels:
        if pid in pid_to_feat_idx:
            feat_idx = pid_to_feat_idx[pid]
            if feat_idx < len(node_features_full):
                patient_nodes_list.append(feat_idx)
                patient_ids.append(pid)
    
    if len(patient_nodes_list) == 0:
        return np.array([]), np.array([]), np.array([])
    
    patient_nodes = torch.tensor(patient_nodes_list, dtype=torch.long).to(device)
    node_features = node_features_full[patient_nodes]
    patient_nodes_reindexed = torch.arange(len(patient_nodes), dtype=torch.long).to(device)
    
    # Get true labels
    class_targets = []
    for pid in patient_ids:
        patient_label = labels_df[labels_df['patient_id'] == pid].iloc[0]
        class_targets.append(patient_label['utilization_class'])
    
    class_targets = torch.tensor(class_targets, dtype=torch.long).to(device)
    
    # Forward pass
    with torch.no_grad():
        if hasattr(model, 'get_embeddings') and hasattr(model, 'pred_head'):
            embeddings = model.get_embeddings(patient_nodes_reindexed, node_features)
            logits = model.pred_head(embeddings)
        else:
            logits = model(patient_nodes_reindexed, node_features)
        
        predictions = torch.argmax(logits, dim=1)
    
    # Convert to numpy
    predictions = predictions.cpu().numpy()
    true_labels = class_targets.cpu().numpy()
    logits = logits.cpu().numpy()
    
    return predictions, true_labels, logits


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
    
    # CRITICAL FIX: Only validate on patients that have labels
    labels_df = val_data['labels']
    
    # Get unique patient IDs that have labels
    patients_with_labels = labels_df['patient_id'].unique()
    
    # CRITICAL: Map patient IDs to their position in node_features, not to original graph node IDs
    patient_nodes_list = []
    patient_ids = []
    
    # Get node features
    node_features_full = val_data['node_features']['patient'].to(device)
    
    # Create mapping from patient_id to index in node_features
    reverse_node_map = val_data.get('reverse_node_maps', {}).get('patient', {})
    
    # Build patient_id -> feature_index mapping
    pid_to_feat_idx = {}
    for feat_idx, pid in reverse_node_map.items():
        pid_to_feat_idx[pid] = feat_idx
    
    for pid in patients_with_labels:
        if pid in pid_to_feat_idx:
            # Use the index in node_features, not the original graph node ID
            feat_idx = pid_to_feat_idx[pid]
            # Validate index is in bounds
            if feat_idx < len(node_features_full):
                patient_nodes_list.append(feat_idx)
                patient_ids.append(pid)
    
    if len(patient_nodes_list) == 0:
        raise ValueError(f"No valid patient nodes found for validation! Check node_features alignment.")
    
    patient_nodes = torch.tensor(patient_nodes_list, dtype=torch.long).to(device)
    
    # Index only the patients we're using
    node_features = node_features_full[patient_nodes]
    
    # Reset patient_nodes to be 0-indexed
    patient_nodes_reindexed = torch.arange(len(patient_nodes), dtype=torch.long).to(device)
    
    # Match labels
    delta_targets = []
    binary_targets = []
    censored_flags = []
    
    for pid in patient_ids:
        patient_labels = labels_df[labels_df['patient_id'] == pid]
        if len(patient_labels) > 0:
            label = patient_labels.iloc[0]
            # Use original days for evaluation metrics (not normalized)
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
    
    # Evaluate - use reindexed patient nodes
    metrics = evaluate_model(
        model, patient_nodes_reindexed, node_features,
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
    
    print(f"\n  📊 Training Data Quality Check:")
    print(f"    Total samples: {len(labels_df)}")
    print(f"    Task type: {config.TASK_TYPE}")
    
    # Check data based on task type
    if config.TASK_TYPE == 'classification':
        # Check class distribution
        class_counts = labels_df['utilization_class'].value_counts().sort_index()
        print(f"\n    Class Distribution:")
        for class_idx in range(config.NUM_CLASSES):
            class_name = ['low', 'medium', 'high'][class_idx]
            count = class_counts.get(class_idx, 0)
            pct = 100 * count / len(labels_df) if len(labels_df) > 0 else 0
            print(f"      Class {class_idx} ({class_name}): {count} ({pct:.1f}%)")
        
        # Compute class weights for imbalanced data
        if config.CLASS_WEIGHTS is None:
            from sklearn.utils.class_weight import compute_class_weight
            unique_classes = np.unique(labels_df['utilization_class'])
            class_weights = compute_class_weight('balanced', classes=unique_classes, y=labels_df['utilization_class'])
            class_weights = class_weights.tolist()
            print(f"    Computed class weights: {class_weights}")
        else:
            class_weights = config.CLASS_WEIGHTS
    else:
        # Survival/regression mode
        uncensored = (labels_df['days_to_next_ed'] >= 0).sum()
        positive_30d = labels_df['has_next_ed_30d'].sum()
        
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
        
        class_weights = None  # Not used in survival mode
    
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
        
        # Train - dispatch to appropriate function based on task type
        if config.TASK_TYPE == 'classification':
            train_loss = train_epoch_classification(model, train_data, optimizer, device, epoch, class_weights)
        else:
            train_loss = train_epoch(model, train_data, optimizer, device, epoch)
        
        # Validate - dispatch to appropriate function based on task type
        if epoch % config.SAVE_INTERVAL == 0:
            if config.TASK_TYPE == 'classification':
                # Get predictions and compute classification metrics
                predictions, true_labels, logits = validate_classification(model, val_data, device)
                from evaluate import evaluate_model_classification
                val_metrics = evaluate_model_classification(predictions, true_labels, logits)
            else:
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


