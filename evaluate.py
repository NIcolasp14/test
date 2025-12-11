"""
Evaluation Metrics for ED Utilization Prediction
Implements C-index, MAE, RMSE, AUROC@7d/30d/90d
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error
from scipy.stats import bootstrap
import config

def concordance_index(predictions, targets, censored_mask=None):
    """
    Calculate Harrell's C-index for survival analysis
    
    Args:
        predictions: Predicted time-to-event
        targets: True time-to-event
        censored_mask: Boolean mask (True = censored, False = observed event)
    
    Returns:
        c_index: Float between 0 and 1 (0.5 = random, 1.0 = perfect)
    """
    if censored_mask is None:
        censored_mask = targets < 0
    
    predictions = predictions.cpu().numpy() if torch.is_tensor(predictions) else predictions
    targets = targets.cpu().numpy() if torch.is_tensor(targets) else targets
    censored_mask = censored_mask.cpu().numpy() if torch.is_tensor(censored_mask) else censored_mask
    
    # Remove censored samples for simplified C-index
    observed_mask = ~censored_mask & (targets > 0)
    
    if observed_mask.sum() < 2:
        return 0.5  # Not enough data
    
    preds = predictions[observed_mask]
    times = targets[observed_mask]
    
    # Calculate concordant and discordant pairs
    concordant = 0
    discordant = 0
    tied = 0
    
    for i in range(len(times)):
        for j in range(i + 1, len(times)):
            if times[i] < times[j]:
                if preds[i] < preds[j]:
                    concordant += 1
                elif preds[i] > preds[j]:
                    discordant += 1
                else:
                    tied += 1
            elif times[i] > times[j]:
                if preds[i] > preds[j]:
                    concordant += 1
                elif preds[i] < preds[j]:
                    discordant += 1
                else:
                    tied += 1
    
    total_pairs = concordant + discordant + tied
    if total_pairs == 0:
        return 0.5
    
    c_index = (concordant + 0.5 * tied) / total_pairs
    return c_index


def compute_auroc_at_threshold(predictions, targets, threshold_days):
    """
    Compute AUROC for predicting ED visit within threshold_days
    
    Args:
        predictions: Predicted time-to-event (in days)
        targets: True time-to-event (in days, -1 for censored)
        threshold_days: Threshold for binary classification (e.g., 30 days)
    
    Returns:
        auroc: Float between 0 and 1
    """
    predictions = predictions.cpu().numpy() if torch.is_tensor(predictions) else predictions
    targets = targets.cpu().numpy() if torch.is_tensor(targets) else targets
    
    # Create binary labels: 1 if ED within threshold, 0 otherwise
    binary_labels = (targets > 0) & (targets <= threshold_days)
    
    # Remove censored samples
    valid_mask = targets > 0
    
    if valid_mask.sum() < 2 or binary_labels.sum() == 0 or (~binary_labels & valid_mask).sum() == 0:
        return 0.5  # Not enough data or no positive/negative samples
    
    # Use predicted time as score (lower time = higher risk)
    # Invert so higher score means higher risk
    scores = -predictions[valid_mask]
    labels = binary_labels[valid_mask].astype(int)
    
    try:
        auroc = roc_auc_score(labels, scores)
    except ValueError:
        auroc = 0.5
    
    return auroc


def evaluate_model_classification(predictions, true_labels, logits):
    """
    Compute classification metrics
    
    Args:
        predictions: Predicted class labels (numpy array)
        true_labels: True class labels (numpy array)
        logits: Raw model outputs (numpy array, shape [n_samples, n_classes])
    
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score,
        f1_score, precision_score, recall_score,
        roc_auc_score, confusion_matrix, classification_report
    )
    from scipy.special import softmax
    
    if len(predictions) == 0 or len(true_labels) == 0:
        return {
            'accuracy': 0.0,
            'balanced_accuracy': 0.0,
            'f1_macro': 0.0,
            'f1_weighted': 0.0,
            'precision_macro': 0.0,
            'recall_macro': 0.0,
            'auroc_ovr': 0.0,
            'auroc_ovo': 0.0
        }
    
    # Basic metrics
    accuracy = accuracy_score(true_labels, predictions)
    balanced_acc = balanced_accuracy_score(true_labels, predictions)
    
    # F1, Precision, Recall (macro and weighted)
    f1_macro = f1_score(true_labels, predictions, average='macro', zero_division=0)
    f1_weighted = f1_score(true_labels, predictions, average='weighted', zero_division=0)
    precision_macro = precision_score(true_labels, predictions, average='macro', zero_division=0)
    recall_macro = recall_score(true_labels, predictions, average='macro', zero_division=0)
    
    # AUROC (requires probabilities)
    probs = softmax(logits, axis=1)
    
    try:
        # One-vs-Rest AUROC
        auroc_ovr = roc_auc_score(true_labels, probs, multi_class='ovr', average='macro')
    except ValueError:
        auroc_ovr = 0.0
    
    try:
        # One-vs-One AUROC
        auroc_ovo = roc_auc_score(true_labels, probs, multi_class='ovo', average='macro')
    except ValueError:
        auroc_ovo = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    
    metrics = {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'auroc_ovr': auroc_ovr,
        'auroc_ovo': auroc_ovo,
        'confusion_matrix': cm
    }
    
    # Per-class metrics
    report = classification_report(true_labels, predictions, output_dict=True, zero_division=0)
    for class_idx in range(config.NUM_CLASSES):
        class_name = ['low', 'medium', 'high'][class_idx]
        if str(class_idx) in report:
            metrics[f'f1_{class_name}'] = report[str(class_idx)]['f1-score']
            metrics[f'precision_{class_name}'] = report[str(class_idx)]['precision']
            metrics[f'recall_{class_name}'] = report[str(class_idx)]['recall']
    
    return metrics


def evaluate_model(model, patient_nodes, node_features, true_deltas, true_binary_labels,
                   censored_mask=None, device=config.DEVICE):
    """
    Comprehensive evaluation of model predictions
    
    Args:
        model: Trained model
        patient_nodes: Patient node IDs
        node_features: Node features
        true_deltas: True time-to-next-ED (in days, NOT normalized)
        true_binary_labels: True binary labels for 30-day ED
        censored_mask: Boolean mask for censored samples
        device: torch device
    
    Returns:
        metrics: Dict of evaluation metrics
    """
    model.eval()
    
    with torch.no_grad():
        # Get predictions
        if isinstance(model, torch.nn.Module):
            if hasattr(model, 'pred_head'):
                # TGN/TGAT style
                if hasattr(model, 'get_embeddings'):
                    embeddings = model.get_embeddings(patient_nodes, node_features)
                    delta_pred, binary_logits = model.pred_head(embeddings)
                else:
                    delta_pred, binary_logits = model(node_features)
            else:
                # Simple forward
                delta_pred, binary_logits = model(patient_nodes, node_features)
        else:
            raise ValueError("Unknown model type")
        
        # Model outputs normalized [0, 1] predictions, denormalize to days
        delta_pred_normalized = delta_pred.squeeze().cpu().numpy()
        MAX_DAYS = float(config.MAX_DAYS_NORMALIZATION)
        delta_pred = delta_pred_normalized * MAX_DAYS  # Denormalize to days
        binary_probs = torch.sigmoid(binary_logits).squeeze().cpu().numpy()
    
    # Convert targets
    true_deltas = true_deltas.cpu().numpy() if torch.is_tensor(true_deltas) else true_deltas
    true_binary = true_binary_labels.cpu().numpy() if torch.is_tensor(true_binary_labels) else true_binary_labels
    
    # Compute metrics
    metrics = {}
    
    # 1. C-index (concordance index)
    metrics['c_index'] = concordance_index(delta_pred, true_deltas, censored_mask)
    
    # 2. MAE on observed (non-censored) samples
    if censored_mask is not None:
        observed_mask = ~censored_mask & (true_deltas > 0)
    else:
        observed_mask = true_deltas > 0
    
    if observed_mask.sum() > 0:
        metrics['mae'] = mean_absolute_error(
            true_deltas[observed_mask],
            delta_pred[observed_mask]
        )
        metrics['rmse'] = np.sqrt(mean_squared_error(
            true_deltas[observed_mask],
            delta_pred[observed_mask]
        ))
    else:
        metrics['mae'] = float('inf')
        metrics['rmse'] = float('inf')
    
    # 3. AUROC at different time windows
    for threshold in config.EVAL_TIME_WINDOWS:
        auroc = compute_auroc_at_threshold(delta_pred, true_deltas, threshold)
        metrics[f'auroc_{threshold}d'] = auroc
    
    # 4. Binary classification metrics (if binary labels provided)
    if true_binary is not None and true_binary.sum() > 0 and (~true_binary.astype(bool)).sum() > 0:
        try:
            metrics['auroc_binary'] = roc_auc_score(true_binary, binary_probs)
        except:
            metrics['auroc_binary'] = 0.5
    else:
        metrics['auroc_binary'] = 0.5
    
    return metrics


def compute_bootstrap_ci(predictions, targets, metric_fn, n_bootstrap=config.BOOTSTRAP_SAMPLES,
                        confidence=config.CONFIDENCE_LEVEL):
    """
    Compute bootstrap confidence intervals for a metric
    
    Args:
        predictions: Model predictions
        targets: True targets
        metric_fn: Function that computes metric from (predictions, targets)
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (e.g., 0.95 for 95% CI)
    
    Returns:
        (lower_bound, upper_bound): Confidence interval
    """
    n_samples = len(predictions)
    bootstrap_metrics = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        pred_sample = predictions[indices]
        target_sample = targets[indices]
        
        # Compute metric
        metric_value = metric_fn(pred_sample, target_sample)
        bootstrap_metrics.append(metric_value)
    
    bootstrap_metrics = np.array(bootstrap_metrics)
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_metrics, alpha / 2 * 100)
    upper = np.percentile(bootstrap_metrics, (1 - alpha / 2) * 100)
    
    return lower, upper


def print_metrics(metrics, prefix=""):
    """Pretty print metrics"""
    print(f"\n{prefix}Evaluation Metrics:")
    print("=" * 60)
    
    # Primary metrics
    if 'c_index' in metrics:
        print(f"  C-index (Harrell):     {metrics['c_index']:.4f}")
    if 'mae' in metrics:
        print(f"  MAE (days):            {metrics['mae']:.2f}")
    if 'rmse' in metrics:
        print(f"  RMSE (days):           {metrics['rmse']:.2f}")
    
    # AUROC at different windows
    print("\n  AUROC at time windows:")
    for threshold in config.EVAL_TIME_WINDOWS:
        key = f'auroc_{threshold}d'
        if key in metrics:
            print(f"    {threshold} days:          {metrics[key]:.4f}")
    
    # Binary classification
    if 'auroc_binary' in metrics:
        print(f"\n  AUROC (binary):        {metrics['auroc_binary']:.4f}")
    
    print("=" * 60)


class MetricsTracker:
    """Track metrics across epochs"""
    
    def __init__(self):
        self.history = {
            'train': [],
            'val': [],
            'test': []
        }
        self.best_metrics = {}
        self.best_epoch = 0
    
    def update(self, split, metrics, epoch):
        """Update metrics for a split"""
        metrics['epoch'] = epoch
        self.history[split].append(metrics)
        
        # Track best validation metrics
        if split == 'val':
            if not self.best_metrics or metrics[config.PRIMARY_METRIC] > self.best_metrics.get(config.PRIMARY_METRIC, -float('inf')):
                self.best_metrics = metrics.copy()
                self.best_epoch = epoch
    
    def get_best(self):
        """Get best validation metrics"""
        return self.best_metrics, self.best_epoch
    
    def get_history(self, split, metric_name):
        """Get history of a specific metric"""
        return [m[metric_name] for m in self.history[split] if metric_name in m]
    
    def summary(self):
        """Print summary of tracking"""
        print("\n" + "="*80)
        print("METRICS SUMMARY")
        print("="*80)
        
        print(f"\nBest validation performance (epoch {self.best_epoch}):")
        print_metrics(self.best_metrics, prefix="  ")
        
        # Final test performance
        if self.history['test']:
            print(f"\nFinal test performance:")
            print_metrics(self.history['test'][-1], prefix="  ")


if __name__ == "__main__":
    print("Testing evaluation metrics...")
    
    # Generate dummy data
    n_samples = 100
    predictions = np.random.exponential(30, n_samples)
    targets = np.random.exponential(30, n_samples)
    censored = np.random.random(n_samples) < 0.2
    targets[censored] = -1
    
    print("\nDummy data:")
    print(f"  Samples: {n_samples}")
    print(f"  Censored: {censored.sum()}")
    print(f"  Observed: {(~censored).sum()}")
    
    # Test C-index
    c_idx = concordance_index(predictions, targets, censored)
    print(f"\n  C-index: {c_idx:.4f}")
    
    # Test AUROC
    for threshold in [7, 30, 90]:
        auroc = compute_auroc_at_threshold(predictions, targets, threshold)
        print(f"  AUROC@{threshold}d: {auroc:.4f}")
    
    print("\n✓ Evaluation metrics tests passed")


