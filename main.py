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
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Import pipeline modules
import config
from data_preprocessing import preprocess_pipeline
from data_preprocessing_enhanced import (
    stratified_split_with_minimum_positives,
    create_enriched_labels_with_history,
    create_utilization_labels,
    create_patient_features,
    create_patient_features_time_aware
)
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
    """Run data preprocessing with enhanced stratified splitting"""
    output_dir = Path(config.OUTPUT_DIR)
    
    # Check for force reprocess flag
    force_reprocess = getattr(config, 'FORCE_REPROCESS', False)
    use_enhanced = getattr(config, 'USE_ENHANCED_PREPROCESSING', True)
    
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
        print(f"    Train: {len(train_data['nyu_edu'])} ED visits, {len(train_labels)} samples")
        print(f"    Val:   {len(val_data['nyu_edu'])} ED visits, {len(val_labels)} samples")
        print(f"    Test:  {len(test_data['nyu_edu'])} ED visits, {len(test_labels)} samples")
        
        # Check for data quality issues
        train_positives = train_labels['has_next_ed_30d'].sum()
        val_positives = val_labels['has_next_ed_30d'].sum()
        
        print(f"\n    Label distribution:")
        print(f"    Train: {train_positives}/{len(train_labels)} positive samples ({100*train_positives/len(train_labels):.1f}%)")
        print(f"    Val:   {val_positives}/{len(val_labels)} positive samples ({100*val_positives/len(val_labels):.1f}%)")
        
        if train_positives < 100:
            print("\n    ⚠️⚠️⚠️  CRITICAL WARNING: VERY FEW POSITIVE TRAINING SAMPLES!")
            print(f"    ⚠️  Only {train_positives} ED-within-30d samples found!")
            print("    ⚠️  Model likely to collapse to 'always negative' solution.")
            print("\n    💡 SOLUTION: Set FORCE_REPROCESS = True in config.py")
            print("       This will use enhanced preprocessing with balanced splits.")
        
        return train_data, val_data, test_data, train_labels, val_labels, test_labels
    else:
        if force_reprocess:
            print("🔄 Force reprocessing data (FORCE_REPROCESS=True)...")
        
        if use_enhanced:
            print("\n" + "="*80)
            print("USING ENHANCED PREPROCESSING WITH BALANCED SPLITS")
            print("="*80)
            
            # Load raw data
            from data_preprocessing import load_raw_data, create_timestamps, standardize_column_names
            data = load_raw_data()
            
            # Preprocess data: add timestamps
            print("\n  Preprocessing data (adding timestamps)...")
            data = standardize_column_names(data)
            data = create_timestamps(data)
            
            # Create stratified patient-level splits
            train_data, val_data, test_data = stratified_split_with_minimum_positives(
                data, min_positives_per_split=300
            )
            
            # Create enriched labels with historical observation points
            # Create labels based on task type
            if config.TASK_TYPE == 'classification':
                use_percentile = getattr(config, 'USE_PERCENTILE_BINNING', True)
                # Compute thresholds on train set, then apply to val/test for consistency
                train_labels, thresholds = create_utilization_labels(
                    train_data, train_data['patient_ids'], 'train', 
                    observation_times='latest', use_percentile_binning=use_percentile
                )
                val_labels, _ = create_utilization_labels(
                    val_data, val_data['patient_ids'], 'val', 
                    observation_times='latest', use_percentile_binning=use_percentile,
                    predefined_thresholds=thresholds
                )
                test_labels, _ = create_utilization_labels(
                    test_data, test_data['patient_ids'], 'test', 
                    observation_times='latest', use_percentile_binning=use_percentile,
                    predefined_thresholds=thresholds
                )
            else:
                # Survival/regression task
                train_labels = create_enriched_labels_with_history(train_data, train_data['patient_ids'], 'train')
                val_labels = create_enriched_labels_with_history(val_data, val_data['patient_ids'], 'val')
                test_labels = create_enriched_labels_with_history(test_data, test_data['patient_ids'], 'test')
            
            # Save processed data
            output_dir.mkdir(exist_ok=True)
            with open(output_dir / 'train_data.pkl', 'wb') as f:
                pickle.dump((train_data, train_labels), f)
            with open(output_dir / 'val_data.pkl', 'wb') as f:
                pickle.dump((val_data, val_labels), f)
            with open(output_dir / 'test_data.pkl', 'wb') as f:
                pickle.dump((test_data, test_labels), f)
            
            print("\n✓ Enhanced preprocessing complete")
            
            return train_data, val_data, test_data, train_labels, val_labels, test_labels
        else:
            print("Using original preprocessing pipeline...")
            return preprocess_pipeline()


def run_feature_extraction(train_data, val_data, test_data):
    """Run feature extraction with real patient features"""
    output_dir = Path(config.OUTPUT_DIR)
    force_reprocess = getattr(config, 'FORCE_REPROCESS', False)
    
    if (output_dir / 'features.pkl').exists() and not force_reprocess:
        print("\n✓ Features found, loading...")
        with open(output_dir / 'features.pkl', 'rb') as f:
            features = pickle.load(f)
        
        # Diagnostic: show feature info
        if 'train_patient_features' in features:
            n_patients = len(features['train_patient_features'])
            if n_patients > 0:
                sample_feat = next(iter(features['train_patient_features'].values()))
                feat_dim = len(sample_feat) if hasattr(sample_feat, '__len__') else sample_feat.shape[0]
                print(f"  Train: {n_patients} patients × {feat_dim} features")
        
        return features
    else:
        print("\nRunning feature extraction pipeline...")
        print("  Creating REAL patient features (not zero vectors)...")
        
        # Use the enhanced feature extraction
        # CRITICAL: Use time-aware features if labels have observation_time (prevents leakage)
        try:
            # Check if labels have observation_time (enriched labels)
            use_time_aware = 'observation_time' in train_labels.columns if hasattr(train_labels, 'columns') else False
            
            if use_time_aware:
                print("  Using time-aware features (NO LEAKAGE - features computed per observation time)")
                # Create time-aware features per label
                train_feat_dict, feat_names = create_patient_features_time_aware(train_data, train_labels)
                val_feat_dict, _ = create_patient_features_time_aware(val_data, val_labels)
                test_feat_dict, _ = create_patient_features_time_aware(test_data, test_labels)
                
                # Aggregate to per-patient (use most recent observation_time's features)
                # This maintains compatibility with graph construction while preventing leakage
                def aggregate_to_patient(feat_dict, labels_df):
                    patient_features = {}
                    for patient_id in labels_df['patient_id'].unique():
                        patient_labels = labels_df[labels_df['patient_id'] == patient_id].sort_values('observation_time')
                        # Use most recent observation's features
                        latest_idx = patient_labels.index[-1]
                        if latest_idx in feat_dict:
                            feat_tensor = feat_dict[latest_idx]
                            # Convert tensor to numpy array for DataFrame
                            if torch.is_tensor(feat_tensor):
                                patient_features[patient_id] = feat_tensor.cpu().numpy()
                            else:
                                patient_features[patient_id] = feat_tensor
                    return patient_features
                
                train_features_dict = aggregate_to_patient(train_feat_dict, train_labels)
                val_features_dict = aggregate_to_patient(val_feat_dict, val_labels)
                test_features_dict = aggregate_to_patient(test_feat_dict, test_labels)
                
                # Convert to DataFrame for compatibility
                # Create DataFrame with patient_id as index
                train_features = pd.DataFrame.from_dict(train_features_dict, orient='index')
                val_features = pd.DataFrame.from_dict(val_features_dict, orient='index')
                test_features = pd.DataFrame.from_dict(test_features_dict, orient='index')
                
                # Rename columns to feature names if available
                if feat_names and len(feat_names) == train_features.shape[1]:
                    train_features.columns = feat_names
                    val_features.columns = feat_names
                    test_features.columns = feat_names
                
                print(f"    Aggregated to {len(train_features)} patients (using latest observation_time per patient)")
            else:
                print("  Using static features (one per patient)")
                train_features = create_patient_features(train_data, train_data['patient_ids'])
                val_features = create_patient_features(val_data, val_data['patient_ids'])
                test_features = create_patient_features(test_data, test_data['patient_ids'])
            
            features = {
                'train': train_features,
                'val': val_features,
                'test': test_features
            }
            
            # Show feature summary
            print(f"\n  ✓ Feature extraction complete:")
            print(f"    Train: {train_features.shape[0]} patients × {train_features.shape[1]} features")
            print(f"    Val:   {val_features.shape[0]} patients × {val_features.shape[1]} features")
            print(f"    Test:  {test_features.shape[0]} patients × {test_features.shape[1]} features")
            print(f"    Top features: {list(train_features.columns[:10])}")
            
            # Convert DataFrames to dict-of-tensors format expected by graph builder
            print("\n  Converting features to tensor format for graph construction...")
            
            # Get actual feature dimension from the DataFrame
            actual_feat_dim = train_features.shape[1]
            print(f"    Actual feature dimension: {actual_feat_dim}")
            print(f"    Target dimension (PROJECTED_DIM): {config.PROJECTED_DIM}")
            
            def df_to_feature_dict(df, target_dim=config.PROJECTED_DIM):
                """Convert DataFrame to {patient_id: tensor} dict with padding/truncation"""
                feature_dict = {}
                for patient_id in df.index:
                    feat_values = df.loc[patient_id].values
                    feat_tensor = torch.tensor(feat_values, dtype=torch.float32)
                    
                    # Pad or truncate to target dimension
                    if len(feat_tensor) < target_dim:
                        # Pad with zeros
                        padding = torch.zeros(target_dim - len(feat_tensor))
                        feat_tensor = torch.cat([feat_tensor, padding])
                    elif len(feat_tensor) > target_dim:
                        # Truncate (shouldn't happen but handle it)
                        feat_tensor = feat_tensor[:target_dim]
                    
                    feature_dict[patient_id] = feat_tensor
                return feature_dict
            
            features_for_graph = {
                'train_patient_features': df_to_feature_dict(train_features),
                'val_patient_features': df_to_feature_dict(val_features),
                'test_patient_features': df_to_feature_dict(test_features),
                # Note: code_embeddings, provider_embeddings, etc. would come from LLM embeddings
                # For now, graph builder will use zero vectors for those (which is fine)
            }
            
            # Save both formats
            output_dir.mkdir(exist_ok=True)
            with open(output_dir / 'features.pkl', 'wb') as f:
                pickle.dump(features_for_graph, f)
            
            # Also save original DataFrames for later analysis
            with open(output_dir / 'features_df.pkl', 'wb') as f:
                pickle.dump(features, f)
            
            return features_for_graph
        except Exception as e:
            print(f"  ⚠️  Error in feature engineering: {e}")
            print(f"  Falling back to basic features...")
            
            # Fallback: return empty features dict (graph builder will use zeros)
            print("  Using zero-initialized features (no extracted features available)...")
            features = {
                # Empty dicts - graph builder will create zero vectors
                'train_patient_features': {},
                'val_patient_features': {},
                'test_patient_features': {},
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
        
        # DIAGNOSTIC: Show label distribution for CV
        print(f"\n  📊 Combined Labels for Cross-Validation:")
        print(f"    Total samples: {len(all_labels)}")
        print(f"    Unique patients: {all_labels['patient_id'].nunique()}")
        print(f"    Uncensored samples: {(all_labels['days_to_next_ed'] >= 0).sum()} ({100*(all_labels['days_to_next_ed'] >= 0).sum()/len(all_labels):.1f}%)")
        print(f"    Positive samples (ED within 30d): {all_labels['has_next_ed_30d'].sum()} ({100*all_labels['has_next_ed_30d'].sum()/len(all_labels):.1f}%)")
        print(f"    Censored: {(all_labels['days_to_next_ed'] < 0).sum()}")
        
        # Prepare full data dictionary with MERGED node ID maps from all splits
        # This ensures ALL diagnosis/procedure codes are in the vocabulary,
        # allowing embeddings to transfer across CV folds
        print("\n  Merging node ID maps from all splits for CV...")
        
        merged_node_id_maps = defaultdict(dict)
        merged_reverse_node_maps = defaultdict(dict)
        
        # Merge node IDs from train, val, test
        for split_name in ['train', 'val', 'test']:
            if split_name not in graphs['node_id_maps']:
                continue
            for node_type in graphs['node_id_maps'][split_name]:
                for node_id, node_idx in graphs['node_id_maps'][split_name][node_type].items():
                    if node_id not in merged_node_id_maps[node_type]:
                        new_idx = len(merged_node_id_maps[node_type])
                        merged_node_id_maps[node_type][node_id] = new_idx
                        merged_reverse_node_maps[node_type][new_idx] = node_id
        
        print(f"    Merged node types:")
        for ntype in merged_node_id_maps:
            print(f"      {ntype}: {len(merged_node_id_maps[ntype])} unique nodes")
        
        full_data = {
            'node_id_maps': dict(merged_node_id_maps),
            'reverse_node_maps': dict(merged_reverse_node_maps),
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


