"""
Main Execution Script for ED Utilization Prediction Pipeline
Orchestrates data preprocessing, feature extraction, graph construction,
training, and evaluation for TGN, TGAT, and HGT models
"""

import torch
import numpy as np
from pathlib import Path
import pickle
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import pipeline modules
import config
from data_preprocessing import preprocess_pipeline
try:
    from feature_engineering import engineer_all_features
    HAS_FEATURE_ENGINEERING = True
except ImportError:
    HAS_FEATURE_ENGINEERING = False
    print("⚠️  Warning: feature_engineering.py not found - will use basic features only")

from graph_construction import build_all_graphs
from models import create_model, count_parameters
from train import train_model
from evaluate import evaluate_model, print_metrics, MetricsTracker
from cross_validation import train_with_cross_validation, create_stratified_folds, save_cv_results

# Set random seeds for reproducibility
torch.manual_seed(config.SEED)
np.random.seed(config.SEED)
if config.DETERMINISTIC:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_directories():
    """Create necessary directories"""
    for dir_path in [config.OUTPUT_DIR, config.MODEL_SAVE_DIR, config.RESULTS_DIR]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def run_preprocessing():
    """Run data preprocessing if needed"""
    output_dir = Path(config.OUTPUT_DIR)
    
    # Check for force reprocess flag
    force_reprocess = getattr(config, 'FORCE_REPROCESS', False)
    
    if (output_dir / 'train_data.pkl').exists() and not force_reprocess:
        print("✓ Preprocessed data found, loading...")
        with open(output_dir / 'train_data.pkl', 'rb') as f:
            train_data, train_labels = pickle.load(f)
        with open(output_dir / 'val_data.pkl', 'rb') as f:
            val_data, val_labels = pickle.load(f)
        with open(output_dir / 'test_data.pkl', 'rb') as f:
            test_data, test_labels = pickle.load(f)
        
        # CRITICAL DIAGNOSTICS - Show what was loaded
        print("\n  📊 LOADED DATA SUMMARY:")
        print(f"    Train: {len(train_data['nyu_edu'])} ED visits, {len(train_labels)} patients")
        print(f"    Val:   {len(val_data['nyu_edu'])} ED visits, {len(val_labels)} patients")
        print(f"    Test:  {len(test_data['nyu_edu'])} ED visits, {len(test_labels)} patients")
        
        # Check for data quality issues
        train_uncensored = (train_labels['days_to_next_ed'] >= 0).sum()
        val_uncensored = (val_labels['days_to_next_ed'] >= 0).sum()
        
        print(f"\n    Label distribution:")
        print(f"    Train: {train_uncensored}/{len(train_labels)} uncensored samples")
        print(f"    Val:   {val_uncensored}/{len(val_labels)} uncensored samples")
        
        if train_uncensored == 0:
            print("\n    ⚠️  CRITICAL WARNING: NO UNCENSORED TRAINING SAMPLES!")
            print("    ⚠️  Model cannot learn without ED visit outcomes.")
            print("    ⚠️  This usually means:")
            print("        - Column names in data don't match expectations")
            print("        - Time cutoffs exclude all ED visits")
            print("        - Data was cached before fixes were applied")
            print("\n    💡 SOLUTION: Delete the outputs/ folder and rerun:")
            print("       rm -rf outputs/  # or rmdir /s outputs on Windows")
            print("       python main.py")
            print("\n    Or set FORCE_REPROCESS = True in config.py")
        
        return train_data, val_data, test_data, train_labels, val_labels, test_labels
    else:
        if force_reprocess:
            print("🔄 Force reprocessing data (FORCE_REPROCESS=True)...")
        else:
            print("Running preprocessing pipeline...")
        return preprocess_pipeline()


def run_feature_extraction(train_data, val_data, test_data):
    """Run feature extraction if needed"""
    output_dir = Path(config.OUTPUT_DIR)
    
    if (output_dir / 'features.pkl').exists():
        print("\n✓ Features found, loading...")
        with open(output_dir / 'features.pkl', 'rb') as f:
            features = pickle.load(f)
        return features
    else:
        print("\nRunning feature extraction pipeline...")
        
        if HAS_FEATURE_ENGINEERING:
            # Use engineered features
            print("  Using engineered temporal features...")
            try:
                # Engineer features from train data
                train_features = engineer_all_features(train_data)
                val_features = engineer_all_features(val_data)
                test_features = engineer_all_features(test_data)
                
                features = {
                    'train': train_features,
                    'val': val_features,
                    'test': test_features
                }
                
                # Save features
                output_dir.mkdir(exist_ok=True)
                with open(output_dir / 'features.pkl', 'wb') as f:
                    pickle.dump(features, f)
                
                return features
            except Exception as e:
                print(f"  ⚠️  Error in feature engineering: {e}")
                print(f"  Falling back to basic features...")
        
        # Fallback: return empty features (graphs will still work)
        print("  Using basic graph features only...")
        features = {
            'train': None,
            'val': None,
            'test': None
        }
        
        # Save empty features
        output_dir.mkdir(exist_ok=True)
        with open(output_dir / 'features.pkl', 'wb') as f:
            pickle.dump(features, f)
        
        return features


def run_graph_construction():
    """Run graph construction if needed"""
    output_dir = Path(config.OUTPUT_DIR)
    
    if (output_dir / 'graphs.pkl').exists():
        print("\n✓ Graphs found, loading...")
        with open(output_dir / 'graphs.pkl', 'rb') as f:
            graphs = pickle.load(f)
        return graphs
    else:
        print("\nRunning graph construction pipeline...")
        return build_all_graphs()


def prepare_training_data(graphs, features):
    """
    Prepare data dictionaries for training
    
    Returns:
        train_data_dict, val_data_dict, test_data_dict
    """
    print("\nPreparing training data...")
    
    # Reconstruct node features for each split
    train_data = {
        'graph': graphs['train_graph'],
        'labels': graphs['train_labels'],
        'node_id_maps': graphs['node_id_maps']['train'],
        'reverse_node_maps': graphs['reverse_node_maps']['train'],
        'node_features': {}  # Will be populated
    }
    
    val_data = {
        'graph': graphs['val_graph'],
        'labels': graphs['val_labels'],
        'node_id_maps': graphs['node_id_maps']['val'],
        'reverse_node_maps': graphs['reverse_node_maps']['val'],
        'node_features': {}
    }
    
    test_data = {
        'graph': graphs['test_graph'],
        'labels': graphs['test_labels'],
        'node_id_maps': graphs['node_id_maps']['test'],
        'reverse_node_maps': graphs['reverse_node_maps']['test'],
        'node_features': {}
    }
    
    # Add patient features (simplified - using cached features from graphs)
    if graphs['train_graph'] is not None:
        # PyG syntax: graph[node_type].x
        train_data['node_features']['patient'] = graphs['train_graph']['patient'].x
        val_data['node_features']['patient'] = graphs['val_graph']['patient'].x
        test_data['node_features']['patient'] = graphs['test_graph']['patient'].x
    else:
        # Fallback: create dummy features
        n_train_patients = len(train_data['node_id_maps']['patient'])
        n_val_patients = len(val_data['node_id_maps']['patient'])
        n_test_patients = len(test_data['node_id_maps']['patient'])
        
        train_data['node_features']['patient'] = torch.randn(n_train_patients, config.HIDDEN_DIM)
        val_data['node_features']['patient'] = torch.randn(n_val_patients, config.HIDDEN_DIM)
        test_data['node_features']['patient'] = torch.randn(n_test_patients, config.HIDDEN_DIM)
    
    print(f"  Train: {len(train_data['node_id_maps']['patient'])} patients, {len(train_data['labels'])} labels")
    print(f"  Val: {len(val_data['node_id_maps']['patient'])} patients, {len(val_data['labels'])} labels")
    print(f"  Test: {len(test_data['node_id_maps']['patient'])} patients, {len(test_data['labels'])} labels")
    
    return train_data, val_data, test_data


def train_all_models(train_data, val_data, test_data):
    """
    Train all specified models and compare results
    
    Returns:
        results: Dictionary with results for each model
    """
    results = {}
    
    for model_name in config.MODELS_TO_TRAIN:
        print(f"\n{'='*80}")
        print(f"MODEL: {model_name}")
        print(f"{'='*80}")
        
        try:
            # Create model
            if model_name == 'TGN':
                num_patients = len(train_data['node_id_maps']['patient'])
                model = create_model(model_name, num_nodes=num_patients)
            elif model_name == 'TGAT':
                model = create_model(model_name)
            elif model_name == 'HGT':
                model = create_model(model_name)
            else:
                print(f"  ✗ Unknown model: {model_name}")
                continue
            
            print(f"\n  Model parameters: {count_parameters(model):,}")
            
            # Train model
            trained_model, metrics_tracker = train_model(
                model, train_data, val_data, model_name, device=config.DEVICE
            )
            
            # Evaluate on test set (inductive)
            print(f"\n  Evaluating on test set (inductive)...")
            test_metrics = validate_test(trained_model, test_data)
            print_metrics(test_metrics, prefix="  TEST ")
            
            # Store results
            results[model_name] = {
                'model': trained_model,
                'metrics_tracker': metrics_tracker,
                'test_metrics': test_metrics,
                'num_parameters': count_parameters(trained_model)
            }
            
        except Exception as e:
            print(f"  ✗ Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return results


def validate_test(model, test_data):
    """Evaluate model on test set"""
    device = config.DEVICE
    model.eval()
    
    # Prepare test data
    patient_nodes = torch.tensor(
        list(test_data['node_id_maps']['patient'].values()),
        dtype=torch.long
    ).to(device)
    
    node_features = test_data['node_features']['patient'].to(device)
    
    # Get labels
    labels_df = test_data['labels']
    patient_ids = [test_data['reverse_node_maps']['patient'][i] for i in range(len(patient_nodes))]
    
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


def create_results_table(results):
    """
    Create comparison table of all models
    
    Returns:
        results_df: Pandas DataFrame with results
    """
    print(f"\n{'='*80}")
    print("RESULTS COMPARISON")
    print(f"{'='*80}\n")
    
    rows = []
    for model_name, result in results.items():
        test_metrics = result['test_metrics']
        row = {
            'Model': model_name,
            '#Params': f"{result['num_parameters']:,}",
            'C-index': f"{test_metrics.get('c_index', 0):.4f}",
            'MAE (days)': f"{test_metrics.get('mae', 0):.2f}",
            'RMSE (days)': f"{test_metrics.get('rmse', 0):.2f}",
            'AUROC@7d': f"{test_metrics.get('auroc_7d', 0):.4f}",
            'AUROC@30d': f"{test_metrics.get('auroc_30d', 0):.4f}",
            'AUROC@90d': f"{test_metrics.get('auroc_90d', 0):.4f}",
        }
        rows.append(row)
    
    results_df = pd.DataFrame(rows)
    print(results_df.to_string(index=False))
    
    # Save results
    results_dir = Path(config.RESULTS_DIR)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(results_dir / f'results_{timestamp}.csv', index=False)
    print(f"\n  ✓ Results saved to {results_dir / f'results_{timestamp}.csv'}")
    
    return results_df


def train_all_models_cv(full_data, labels_df, graphs):
    """
    Train all models using k-fold cross-validation
    
    Args:
        full_data: Full dataset dictionary
        labels_df: All labels (train + val + test combined)
        graphs: Graph data
    
    Returns:
        cv_results: Dictionary with CV results for each model
    """
    print(f"\n{'='*80}")
    print(f"K-FOLD CROSS-VALIDATION MODE")
    print(f"  Number of folds: {config.K_FOLDS}")
    print(f"{'='*80}")
    
    cv_results = {}
    
    for model_name in config.MODELS_TO_TRAIN:
        try:
            print(f"\n{'='*80}")
            print(f"MODEL: {model_name} (K-Fold CV)")
            print(f"{'='*80}")
            
            # Run k-fold CV
            model_cv_results = train_with_cross_validation(
                model_name=model_name,
                full_data=full_data,
                labels_df=labels_df,
                graphs=graphs,
                n_folds=config.K_FOLDS,
                device=config.DEVICE
            )
            
            cv_results[model_name] = model_cv_results
            
            # Save results
            save_cv_results(model_cv_results, model_name)
            
        except Exception as e:
            print(f"\n  ✗ Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create comparison table
    if cv_results:
        print(f"\n{'='*80}")
        print("CROSS-VALIDATION RESULTS COMPARISON")
        print(f"{'='*80}")
        
        # Create DataFrame for comparison
        comparison_data = []
        for model_name, results in cv_results.items():
            row = {'Model': model_name}
            # Add mean metrics
            for key, value in results.items():
                if key.endswith('_mean'):
                    metric_name = key.replace('_mean', '')
                    std_value = results.get(f"{metric_name}_std", 0)
                    row[metric_name] = f"{value:.4f} ± {std_value:.4f}"
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        print(f"\n{comparison_df.to_string(index=False)}")
        
        # Save comparison
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = Path(config.RESULTS_DIR) / f'cv_comparison_{timestamp}.csv'
        comparison_df.to_csv(results_file, index=False)
        print(f"\n  ✓ Saved comparison to {results_file}")
    
    return cv_results


def main():
    """Main execution pipeline"""
    print("\n" + "="*80)
    print("ED UTILIZATION PREDICTION PIPELINE")
    print("Temporal Heterogeneous Graph Neural Networks")
    print("="*80)
    
    # Setup
    setup_directories()
    
    eval_mode = "K-Fold CV" if config.USE_CROSS_VALIDATION else "Temporal Split"
    
    print(f"\nConfiguration:")
    print(f"  Device: {config.DEVICE}")
    print(f"  Evaluation: {eval_mode}")
    if config.USE_CROSS_VALIDATION:
        print(f"  K-Folds: {config.K_FOLDS}")
    print(f"  Models: {', '.join(config.MODELS_TO_TRAIN)}")
    print(f"  Hidden dim: {config.HIDDEN_DIM}")
    print(f"  Num epochs: {config.NUM_EPOCHS}")
    print(f"  Learning rate: {config.LEARNING_RATE}")
    
    # Step 1: Data Preprocessing
    train_data, val_data, test_data, train_labels, val_labels, test_labels = run_preprocessing()
    
    # Step 2: Feature Extraction
    features = run_feature_extraction(train_data, val_data, test_data)
    
    # Step 3: Graph Construction
    graphs = run_graph_construction()
    
    # Step 4: Choose evaluation strategy
    if config.USE_CROSS_VALIDATION:
        # Combine all data for k-fold CV
        print("\n  Using K-Fold Cross-Validation...")
        
        # Combine labels from all splits
        all_labels = pd.concat([train_labels, val_labels, test_labels], ignore_index=True)
        
        # Prepare full data dictionary
        full_data = {
            'node_id_maps': graphs['node_id_maps']['train'],  # Use train maps as base
            'reverse_node_maps': graphs['reverse_node_maps']['train'],
            'node_features': {}
        }
        
        # Combine node features from all splits
        for node_type in ['patient', 'diagnosis', 'procedure', 'provider']:
            try:
                train_feat = graphs['train_graph'][node_type].x if hasattr(graphs['train_graph'], '__getitem__') else None
                if train_feat is not None:
                    full_data['node_features'][node_type] = train_feat
            except:
                pass
        
        # Train with k-fold CV
        results = train_all_models_cv(full_data, all_labels, graphs)
        
    else:
        # Use temporal split
        print("\n  Using Temporal Split Evaluation...")
        
        # Step 4: Prepare Training Data
        train_dict, val_dict, test_dict = prepare_training_data(graphs, features)
        
        # Step 5: Train Models
        results = train_all_models(train_dict, val_dict, test_dict)
        
        # Step 6: Create Results Table
        if results:
            results_df = create_results_table(results)
        else:
            print("\n  ✗ No results to display (all models failed)")
    
    print(f"\n{'='*80}")
    print("PIPELINE COMPLETE")
    print(f"{'='*80}\n")
    
    return results


if __name__ == "__main__":
    results = main()


