"""
Main Pipeline for Heterogeneous Graph Transformer
ED Utilization Classification with Cross-Validation

This script runs the complete pipeline:
1. Load and preprocess data
2. Create patient-level labels
3. Build heterogeneous graph
4. Train HGT model with k-fold cross-validation
5. Evaluate and save results
"""

import torch
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Import modules
import config_hgt as config
from data_loader import DataLoader
from graph_builder import HeterogeneousGraphBuilder
from model_hgt import HGTClassifier, count_parameters
from cross_validation import CrossValidator, create_simple_split
from trainer import Trainer
from evaluator import Evaluator, aggregate_cv_results

# Set random seeds
config.set_seed()


def run_single_fold(
    fold_idx: int,
    fold_split: Dict,
    data_loader: DataLoader,
    patient_features: pd.DataFrame,
    patient_labels: pd.DataFrame,
    save_dir: str,
    verbose: bool = True
) -> Dict:
    """
    Run training and evaluation for a single fold
    
    Args:
        fold_idx: Fold index
        fold_split: Dictionary with train/val/test patient IDs
        data_loader: DataLoader object
        patient_features: Patient features DataFrame
        patient_labels: Patient labels DataFrame
        save_dir: Directory to save results
        verbose: Print progress
    
    Returns:
        Dictionary with fold results
    """
    if verbose:
        print("\n" + "=" * 80)
        print(f"FOLD {fold_idx + 1}")
        print("=" * 80)
    
    # Get patient IDs for this fold
    train_patients = fold_split['train']
    val_patients = fold_split['val']
    test_patients = fold_split['test']
    
    # All patients in this fold (train + val + test)
    all_fold_patients = np.concatenate([train_patients, val_patients, test_patients])
    
    # Build heterogeneous graph for this fold
    if verbose:
        print(f"\nBuilding graph for fold {fold_idx + 1}...")
    
    graph_builder = HeterogeneousGraphBuilder(data_loader, verbose=verbose)
    hetero_data = graph_builder.build_hetero_data(
        patient_ids=all_fold_patients,
        patient_features=patient_features,
        patient_labels=patient_labels
    )
    
    # Set train/val/test masks
    hetero_data = graph_builder.create_masks(
        hetero_data,
        graph_builder.patient_id_map,
        train_patients,
        val_patients,
        test_patients
    )
    
    # Verify masks
    assert hetero_data['patient'].train_mask.sum() == len(train_patients)
    assert hetero_data['patient'].val_mask.sum() == len(val_patients)
    assert hetero_data['patient'].test_mask.sum() == len(test_patients)
    
    if verbose:
        print(f"✓ Graph built successfully")
        print(f"  Train patients: {len(train_patients)}")
        print(f"  Val patients: {len(val_patients)}")
        print(f"  Test patients: {len(test_patients)}")
    
    # Create model
    node_feature_dims = {
        node_type: hetero_data[node_type].x.shape[1]
        for node_type in config.NODE_TYPES
    }
    
    model = HGTClassifier(
        node_types=config.NODE_TYPES,
        edge_types=config.EDGE_TYPES,
        node_feature_dims=node_feature_dims,
        hidden_dim=config.HGT_HIDDEN_DIM,
        num_layers=config.HGT_NUM_LAYERS,
        num_heads=config.HGT_NUM_HEADS,
        num_classes=config.NUM_CLASSES,
        dropout=config.HGT_DROPOUT,
        use_norm=config.HGT_USE_NORM
    )
    
    if verbose:
        print(f"\n✓ Model created")
        print(f"  Parameters: {count_parameters(model):,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        device=config.DEVICE,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        verbose=verbose
    )
    
    # Train model
    model_save_path = os.path.join(save_dir, f'fold_{fold_idx}_best_model.pt')
    history = trainer.train(
        hetero_data=hetero_data,
        num_epochs=config.NUM_EPOCHS,
        patience=config.EARLY_STOPPING_PATIENCE,
        min_delta=config.EARLY_STOPPING_MIN_DELTA,
        save_path=model_save_path if config.SAVE_BEST_MODEL else None
    )
    
    # Save training history
    history_save_path = os.path.join(save_dir, f'fold_{fold_idx}_history.json')
    trainer.save_history(history_save_path)
    
    # Evaluate on test set
    test_loss, test_metrics = trainer.evaluate(hetero_data, mask_name='test')
    
    # Get predictions
    test_preds, test_probs = trainer.predict(hetero_data, mask_name='test')
    test_labels = hetero_data['patient'].y[hetero_data['patient'].test_mask].cpu().numpy()
    
    # Create evaluator
    evaluator = Evaluator(class_names=['Low', 'Medium', 'High'], verbose=verbose)
    
    # Compute detailed metrics
    detailed_metrics = evaluator.compute_metrics(test_labels, test_preds, test_probs)
    evaluator.print_metrics(detailed_metrics, title=f"Fold {fold_idx + 1} Test Results")
    
    # Plot confusion matrix
    cm_save_path = os.path.join(save_dir, f'fold_{fold_idx}_confusion_matrix.png')
    evaluator.plot_confusion_matrix(test_labels, test_preds, save_path=cm_save_path)
    
    # Plot ROC curves
    roc_save_path = os.path.join(save_dir, f'fold_{fold_idx}_roc_curves.png')
    evaluator.plot_roc_curves(test_labels, test_probs, save_path=roc_save_path)
    
    # Plot training history
    history_plot_path = os.path.join(save_dir, f'fold_{fold_idx}_training_history.png')
    evaluator.plot_training_history(history, save_path=history_plot_path)
    
    # Save predictions
    if config.SAVE_PREDICTIONS:
        # Get test patient IDs
        test_patient_ids = []
        for patient_idx in range(len(graph_builder.patient_id_map)):
            if hetero_data['patient'].test_mask[patient_idx]:
                test_patient_ids.append(graph_builder.patient_idx_to_id[patient_idx])
        
        pred_save_path = os.path.join(save_dir, f'fold_{fold_idx}_predictions.csv')
        evaluator.save_predictions(
            np.array(test_patient_ids),
            test_labels,
            test_preds,
            test_probs,
            save_path=pred_save_path
        )
    
    # Generate classification report
    report_save_path = os.path.join(save_dir, f'fold_{fold_idx}_classification_report.txt')
    evaluator.generate_classification_report(
        test_labels,
        test_preds,
        save_path=report_save_path
    )
    
    # Return results
    fold_results = {
        'fold': fold_idx,
        'test_loss': test_loss,
        **detailed_metrics
    }
    
    # Remove confusion matrix from returned results (not serializable)
    if 'confusion_matrix' in fold_results:
        del fold_results['confusion_matrix']
    
    return fold_results


def run_cross_validation_pipeline(verbose: bool = True):
    """
    Run complete pipeline with k-fold cross-validation
    
    Args:
        verbose: Print progress
    """
    print("=" * 80)
    print("HETEROGENEOUS GRAPH TRANSFORMER PIPELINE")
    print("ED Utilization Classification with Cross-Validation")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {config.DEVICE}")
    print(f"Number of folds: {config.N_FOLDS}")
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(config.RESULTS_DIR, f'cv_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Results directory: {results_dir}")
    
    # Step 1: Load data
    print("\n" + "=" * 80)
    print("STEP 1: Loading Data")
    print("=" * 80)
    
    data_loader = DataLoader(verbose=verbose)
    data_loader.load_all_data()
    
    # Step 2: Create patient labels
    print("\n" + "=" * 80)
    print("STEP 2: Creating Patient Labels")
    print("=" * 80)
    
    patient_labels = data_loader.create_patient_labels()
    patient_features = data_loader.prepare_patient_features()
    
    # Save data summary
    summary = data_loader.get_summary()
    with open(os.path.join(results_dir, 'data_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Step 3: Create cross-validation folds
    print("\n" + "=" * 80)
    print("STEP 3: Creating Cross-Validation Folds")
    print("=" * 80)
    
    cv = CrossValidator(
        n_folds=config.N_FOLDS,
        stratified=config.STRATIFIED,
        random_state=config.CV_RANDOM_STATE,
        verbose=verbose
    )
    
    fold_splits = cv.create_folds(patient_labels)
    
    # Save fold splits
    cv_summary = cv.get_summary()
    with open(os.path.join(results_dir, 'cv_splits.json'), 'w') as f:
        json.dump(cv_summary, f, indent=2)
    
    # Step 4: Train and evaluate for each fold
    print("\n" + "=" * 80)
    print("STEP 4: Training and Evaluation")
    print("=" * 80)
    
    fold_results = []
    
    for fold_idx, fold_split in enumerate(fold_splits):
        fold_result = run_single_fold(
            fold_idx=fold_idx,
            fold_split=fold_split,
            data_loader=data_loader,
            patient_features=patient_features,
            patient_labels=patient_labels,
            save_dir=results_dir,
            verbose=verbose
        )
        
        fold_results.append(fold_result)
    
    # Step 5: Aggregate results
    print("\n" + "=" * 80)
    print("STEP 5: Aggregating Results")
    print("=" * 80)
    
    aggregated_results = aggregate_cv_results(fold_results, verbose=verbose)
    
    # Save fold results
    with open(os.path.join(results_dir, 'fold_results.json'), 'w') as f:
        json.dump(fold_results, f, indent=2)
    
    # Save aggregated results
    with open(os.path.join(results_dir, 'aggregated_results.json'), 'w') as f:
        json.dump(aggregated_results, f, indent=2)
    
    # Create summary report
    summary_path = os.path.join(results_dir, 'SUMMARY.txt')
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("HGT PIPELINE SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device: {config.DEVICE}\n")
        f.write(f"Number of folds: {config.N_FOLDS}\n\n")
        
        f.write("Aggregated Results (Mean ± Std):\n")
        f.write("-" * 80 + "\n")
        for metric in ['accuracy', 'balanced_accuracy', 'f1_macro', 'f1_weighted', 
                       'precision_macro', 'recall_macro', 'auroc_ovr']:
            if f'{metric}_mean' in aggregated_results:
                mean = aggregated_results[f'{metric}_mean']
                std = aggregated_results[f'{metric}_std']
                f.write(f"{metric}: {mean:.4f} ± {std:.4f}\n")
        
        f.write("\nPer-Fold Results:\n")
        f.write("-" * 80 + "\n")
        for fold_result in fold_results:
            f.write(f"\nFold {fold_result['fold'] + 1}:\n")
            for metric in ['accuracy', 'balanced_accuracy', 'f1_macro', 'f1_weighted']:
                if metric in fold_result:
                    f.write(f"  {metric}: {fold_result[metric]:.4f}\n")
    
    print(f"\n✓ Pipeline completed successfully!")
    print(f"Results saved to: {results_dir}")
    print(f"\nFinal Results:")
    print(f"  Accuracy: {aggregated_results['accuracy_mean']:.4f} ± {aggregated_results['accuracy_std']:.4f}")
    print(f"  F1 (Macro): {aggregated_results['f1_macro_mean']:.4f} ± {aggregated_results['f1_macro_std']:.4f}")
    print(f"  AUROC: {aggregated_results['auroc_ovr_mean']:.4f} ± {aggregated_results['auroc_ovr_std']:.4f}")
    
    return fold_results, aggregated_results


def run_simple_pipeline(verbose: bool = True):
    """
    Run pipeline with simple train/val/test split (no cross-validation)
    
    Args:
        verbose: Print progress
    """
    print("=" * 80)
    print("HETEROGENEOUS GRAPH TRANSFORMER PIPELINE")
    print("ED Utilization Classification (Single Split)")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {config.DEVICE}")
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(config.RESULTS_DIR, f'single_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)
    
    # Load data and create labels
    data_loader = DataLoader(verbose=verbose)
    data_loader.load_all_data()
    patient_labels = data_loader.create_patient_labels()
    patient_features = data_loader.prepare_patient_features()
    
    # Create simple split
    split = create_simple_split(
        patient_labels,
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VAL_RATIO,
        test_ratio=config.TEST_RATIO,
        stratified=config.STRATIFIED,
        verbose=verbose
    )
    
    # Run single fold (using the entire dataset)
    fold_split = {
        'fold': 0,
        'train': split['train'],
        'val': split['val'],
        'test': split['test']
    }
    
    results = run_single_fold(
        fold_idx=0,
        fold_split=fold_split,
        data_loader=data_loader,
        patient_features=patient_features,
        patient_labels=patient_labels,
        save_dir=results_dir,
        verbose=verbose
    )
    
    print(f"\n✓ Pipeline completed successfully!")
    print(f"Results saved to: {results_dir}")
    
    return results


if __name__ == "__main__":
    # Run cross-validation pipeline
    if config.USE_CROSS_VALIDATION:
        fold_results, aggregated_results = run_cross_validation_pipeline(verbose=True)
    else:
        results = run_simple_pipeline(verbose=True)
