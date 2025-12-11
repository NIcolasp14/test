"""
Stratified K-Fold Cross-Validation for ED Utilization Prediction
Addresses distribution mismatch in temporal splits
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
from pathlib import Path
import pickle
import torch

import config
from train import train_model, train_epoch, train_epoch_classification, validate, validate_classification
from evaluate import evaluate_model, evaluate_model_classification, print_metrics


def create_stratified_folds(labels_df, n_folds=5, random_state=42):
    """
    Create stratified k-folds based on binary 30-day ED label
    Ensures each fold has similar positive class distribution
    
    Args:
        labels_df: DataFrame with patient labels (must have 'has_next_ed_30d')
        n_folds: Number of folds
        random_state: Random seed for reproducibility
    
    Returns:
        folds: List of (train_indices, test_indices) tuples
    """
    print(f"\n{'='*80}")
    print(f"CREATING STRATIFIED {n_folds}-FOLD CROSS-VALIDATION")
    print(f"{'='*80}")
    
    # Get unique patients and their labels
    # Group by patient_id and take first label (patients may have multiple labels)
    patient_labels = labels_df.groupby('patient_id').agg({
        'has_next_ed_30d': 'max'  # Use max to capture if ANY visit had ED within 30d
    }).reset_index()
    
    patient_ids = patient_labels['patient_id'].values
    binary_labels = patient_labels['has_next_ed_30d'].values
    
    print(f"\n  Total patients: {len(patient_ids)}")
    print(f"  Positive class (ED within 30d): {binary_labels.sum()} ({100*binary_labels.mean():.1f}%)")
    print(f"  Negative class: {(~binary_labels.astype(bool)).sum()} ({100*(1-binary_labels.mean()):.1f}%)")
    
    # Check if we have enough positive samples for stratification
    n_positive = binary_labels.sum()
    if n_positive < n_folds:
        print(f"\n  ⚠️  WARNING: Only {n_positive} positive samples for {n_folds} folds!")
        print(f"     Some folds may have 0 positive samples.")
        print(f"     Consider reducing n_folds or using different split strategy.")
    
    # Create stratified folds
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    folds = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(patient_ids, binary_labels)):
        train_patients = patient_ids[train_idx]
        test_patients = patient_ids[test_idx]
        
        train_pos_rate = binary_labels[train_idx].mean()
        test_pos_rate = binary_labels[test_idx].mean()
        
        print(f"\n  Fold {fold_idx + 1}:")
        print(f"    Train: {len(train_patients)} patients, {binary_labels[train_idx].sum()} positive ({train_pos_rate*100:.1f}%)")
        print(f"    Test:  {len(test_patients)} patients, {binary_labels[test_idx].sum()} positive ({test_pos_rate*100:.1f}%)")
        
        folds.append((train_patients, test_patients))
    
    print(f"\n  ✓ Created {n_folds} stratified folds")
    print(f"    Average positive rate across folds: {binary_labels.mean()*100:.1f}%")
    
    return folds


def prepare_fold_data(train_patients, test_patients, full_data, labels_df, graphs, node_id_maps, reverse_node_maps):
    """
    Prepare train/test data for a single fold
    
    Args:
        train_patients: Array of patient IDs for training
        test_patients: Array of patient IDs for testing
        full_data: Full dataset dictionary
        labels_df: Full labels DataFrame
        graphs: Graph data (if using graph models)
        node_id_maps: Node ID mappings
        reverse_node_maps: Reverse node ID mappings
    
    Returns:
        train_data, test_data: Dictionaries with fold-specific data
    """
    # Filter labels for this fold
    train_labels = labels_df[labels_df['patient_id'].isin(train_patients)]
    test_labels = labels_df[labels_df['patient_id'].isin(test_patients)]
    
    # Create data dictionaries
    train_data = {
        'patient_ids': train_patients,
        'labels': train_labels,
        'node_id_maps': node_id_maps,
        'reverse_node_maps': reverse_node_maps,
        'node_features': full_data['node_features'] if 'node_features' in full_data else None,
    }
    
    test_data = {
        'patient_ids': test_patients,
        'labels': test_labels,
        'node_id_maps': node_id_maps,
        'reverse_node_maps': reverse_node_maps,
        'node_features': full_data['node_features'] if 'node_features' in full_data else None,
    }
    
    return train_data, test_data


def train_with_cross_validation(model_name, full_data, labels_df, graphs, n_folds=5, device=config.DEVICE):
    """
    Train and evaluate a model using stratified k-fold cross-validation
    
    Args:
        model_name: Name of model to train
        full_data: Full dataset dictionary
        labels_df: Full labels DataFrame
        graphs: Graph data
        n_folds: Number of folds
        device: torch device
    
    Returns:
        cv_results: Dictionary with cross-validation results
    """
    print(f"\n{'='*80}")
    print(f"K-FOLD CROSS-VALIDATION: {model_name}")
    print(f"{'='*80}")
    
    # Create folds
    folds = create_stratified_folds(labels_df, n_folds=n_folds)
    
    # Track metrics across folds
    fold_metrics = []
    
    # Train on each fold
    for fold_idx, (train_patients, test_patients) in enumerate(folds):
        print(f"\n{'='*80}")
        print(f"FOLD {fold_idx + 1}/{n_folds}")
        print(f"{'='*80}")
        
        # Prepare fold data
        train_data, test_data = prepare_fold_data(
            train_patients, test_patients,
            full_data, labels_df, graphs,
            full_data.get('node_id_maps', {}),
            full_data.get('reverse_node_maps', {})
        )
        
        # Split train into train/val (80/20)
        n_train = int(0.8 * len(train_patients))
        fold_train_patients = train_patients[:n_train]
        fold_val_patients = train_patients[n_train:]
        
        fold_train_data, fold_val_data = prepare_fold_data(
            fold_train_patients, fold_val_patients,
            full_data, labels_df, graphs,
            full_data.get('node_id_maps', {}),
            full_data.get('reverse_node_maps', {})
        )
        
        # Import model creation
        from models import create_model, count_parameters
        
        # Create model for this fold
        if model_name == 'TGN':
            num_nodes = len(full_data.get('node_id_maps', {}).get('patient', {}))
            model = create_model(model_name, num_nodes=num_nodes)
        else:
            model = create_model(model_name)
        
        print(f"\n  Model: {model_name}")
        print(f"  Parameters: {count_parameters(model):,}")
        
        # Train model
        model, metrics_tracker = train_model(
            model, fold_train_data, fold_val_data,
            model_name=f"{model_name}_fold{fold_idx+1}",
            device=device
        )
        
        # Evaluate on fold test set
        print(f"\n  Evaluating on fold {fold_idx + 1} test set...")
        
        # Dispatch to appropriate evaluation based on task type
        if config.TASK_TYPE == 'classification':
            # Classification evaluation
            predictions, true_labels, logits = validate_classification(model, test_data, device)
            fold_test_metrics = evaluate_model_classification(predictions, true_labels, logits)
        else:
            # Survival/regression evaluation
            test_labels_df = test_data['labels']
            patients_with_labels = test_labels_df['patient_id'].unique()
            node_features_full = test_data['node_features']['patient'].to(device)
            reverse_node_map = test_data.get('reverse_node_maps', {}).get('patient', {})
            pid_to_feat_idx = {feat_idx: pid for feat_idx, pid in reverse_node_map.items()}
            pid_to_feat_idx = {pid: feat_idx for feat_idx, pid in reverse_node_map.items()}
            
            patient_nodes_list = []
            patient_ids_test = []
            delta_targets = []
            binary_targets = []
            censored_flags = []
            
            for pid in patients_with_labels:
                if pid in pid_to_feat_idx:
                    feat_idx = pid_to_feat_idx[pid]
                    if feat_idx < len(node_features_full):
                        patient_nodes_list.append(feat_idx)
                        patient_ids_test.append(pid)
                        
                        patient_labels = test_labels_df[test_labels_df['patient_id'] == pid]
                        if len(patient_labels) > 0:
                            label = patient_labels.iloc[0]
                            delta_targets.append(max(0, label['days_to_next_ed']))
                            binary_targets.append(label['has_next_ed_30d'])
                            censored_flags.append(label['days_to_next_ed'] < 0)
                        else:
                            delta_targets.append(0.0)
                            binary_targets.append(0)
                            censored_flags.append(True)
            
            if len(patient_nodes_list) == 0:
                print(f"  ⚠️  Warning: No valid test patients found for fold {fold_idx + 1}")
                continue
            
            patient_nodes = torch.tensor(patient_nodes_list, dtype=torch.long).to(device)
            node_features = node_features_full[patient_nodes]
            patient_nodes_reindexed = torch.arange(len(patient_nodes), dtype=torch.long).to(device)
            
            delta_targets = torch.tensor(delta_targets, dtype=torch.float32)
            binary_targets = torch.tensor(binary_targets, dtype=torch.float32)
            censored_mask = torch.tensor(censored_flags, dtype=torch.bool)
            
            fold_test_metrics = evaluate_model(
                model, patient_nodes_reindexed, node_features,
                delta_targets, binary_targets, censored_mask, device
            )
        
        print(f"\n  Fold {fold_idx + 1} Test Metrics:")
        print_metrics(fold_test_metrics, prefix="    ")
        
        fold_metrics.append(fold_test_metrics)
    
    # Aggregate metrics across folds
    print(f"\n{'='*80}")
    print(f"CROSS-VALIDATION SUMMARY: {model_name}")
    print(f"{'='*80}")
    
    cv_results = aggregate_cv_metrics(fold_metrics)
    
    return cv_results


def aggregate_cv_metrics(fold_metrics):
    """
    Aggregate metrics across folds (mean and std)
    
    Args:
        fold_metrics: List of metric dictionaries (one per fold)
    
    Returns:
        cv_results: Dictionary with mean and std for each metric
    """
    # Get all metric names
    metric_names = fold_metrics[0].keys()
    
    cv_results = {}
    
    for metric_name in metric_names:
        values = [fold[metric_name] for fold in fold_metrics]
        values = [v for v in values if not np.isnan(v) and not np.isinf(v)]
        
        if len(values) > 0:
            cv_results[f"{metric_name}_mean"] = np.mean(values)
            cv_results[f"{metric_name}_std"] = np.std(values)
            cv_results[f"{metric_name}_min"] = np.min(values)
            cv_results[f"{metric_name}_max"] = np.max(values)
        else:
            cv_results[f"{metric_name}_mean"] = np.nan
            cv_results[f"{metric_name}_std"] = np.nan
            cv_results[f"{metric_name}_min"] = np.nan
            cv_results[f"{metric_name}_max"] = np.nan
    
    # Print summary
    print(f"\n  Aggregated Metrics (mean ± std):")
    print(f"  {'─'*60}")
    
    important_metrics = ['c_index', 'mae', 'rmse', 'auroc_7d', 'auroc_30d', 'auroc_90d', 'auroc_binary']
    
    for metric in important_metrics:
        if f"{metric}_mean" in cv_results:
            mean_val = cv_results[f"{metric}_mean"]
            std_val = cv_results[f"{metric}_std"]
            min_val = cv_results[f"{metric}_min"]
            max_val = cv_results[f"{metric}_max"]
            
            print(f"    {metric:20s}: {mean_val:.4f} ± {std_val:.4f}  (range: {min_val:.4f} - {max_val:.4f})")
    
    return cv_results


def save_cv_results(cv_results, model_name, output_dir=None):
    """
    Save cross-validation results to disk
    
    Args:
        cv_results: Dictionary with CV results
        model_name: Name of model
        output_dir: Directory to save results (default: config.RESULTS_DIR)
    """
    if output_dir is None:
        output_dir = Path(config.RESULTS_DIR)
    
    output_dir.mkdir(exist_ok=True)
    
    # Save as pickle
    cv_file = output_dir / f'{model_name}_cv_results.pkl'
    with open(cv_file, 'wb') as f:
        pickle.dump(cv_results, f)
    
    print(f"\n  ✓ Saved CV results to {cv_file}")
    
    # Save as CSV for easy viewing
    cv_df = pd.DataFrame([cv_results])
    csv_file = output_dir / f'{model_name}_cv_results.csv'
    cv_df.to_csv(csv_file, index=False)
    
    print(f"  ✓ Saved CV results to {csv_file}")


if __name__ == "__main__":
    print("Cross-validation module loaded successfully")
    print("Functions:")
    print("  - create_stratified_folds(labels_df, n_folds=5)")
    print("  - train_with_cross_validation(model_name, full_data, labels_df, graphs, n_folds=5)")
    print("  - aggregate_cv_metrics(fold_metrics)")

