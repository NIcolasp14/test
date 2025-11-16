"""
Time-Aware Feature Engineering - NO LEAKAGE
Computes features at each observation time using only historical data
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Dict, List
import torch


def create_time_aware_features(data: Dict[str, pd.DataFrame], 
                                labels_df: pd.DataFrame,
                                split_name: str) -> Dict[int, torch.Tensor]:
    """
    Create features for each (patient, observation_time) pair in labels.
    
    CRITICAL: Features computed using ONLY events BEFORE observation_time.
    No information leakage.
    
    Args:
        data: Split data with 'nyu_edu', 'diagnosis', 'procedures', 'demographics'
        labels_df: DataFrame with 'patient_id' and 'observation_time' columns
        split_name: Name of split (for logging)
    
    Returns:
        Dict mapping (patient_id, observation_time_index) to feature tensor
    """
    print(f"\n  Creating time-aware features for {split_name} split...")
    print(f"    {len(labels_df)} prediction instances")
    
    ed_visits = data['nyu_edu']
    diagnosis = data['diagnosis']
    procedures = data['procedures']
    demographics = data['demographics']
    
    feature_dict = {}
    
    for idx, row in labels_df.iterrows():
        patient_id = row['patient_id']
        obs_time = row['observation_time']
        
        # Filter to ONLY events BEFORE observation time (NO LEAKAGE)
        patient_ed_before = ed_visits[
            (ed_visits['sys_mbr_sk'] == patient_id) & 
            (ed_visits['timestamp'] < obs_time)
        ].sort_values('timestamp')
        
        patient_dx_before = diagnosis[
            (diagnosis['clm_sys_mbr_sk'] == patient_id) & 
            (diagnosis['timestamp'] < obs_time)
        ].sort_values('timestamp')
        
        patient_proc_before = procedures[
            (procedures['sys_mbr_sk'] == patient_id) & 
            (procedures['timestamp'] < obs_time)
        ].sort_values('timestamp')
        
        # === COMPUTE FEATURES AT THIS OBSERVATION TIME ===
        feat = {}
        
        # === DEMOGRAPHICS (static) ===
        patient_demo = demographics[demographics['sys_mbr_sk'] == patient_id]
        if len(patient_demo) > 0:
            demo = patient_demo.iloc[0]
            if 'dob' in demo.index and pd.notna(demo['dob']):
                feat['age'] = (obs_time - demo['dob']).days / 365.25
            else:
                feat['age'] = 50.0
            feat['is_male'] = 1.0 if demo.get('mbr_gender_cd') == 'M' else 0.0
        else:
            feat['age'] = 50.0
            feat['is_male'] = 0.0
        
        # === ED RECENCY & FREQUENCY ===
        if len(patient_ed_before) > 0:
            last_ed_time = patient_ed_before['timestamp'].max()
            feat['days_since_last_ed'] = (obs_time - last_ed_time).days
            feat['had_any_ed'] = 1.0
            
            # Count in time windows
            feat['ed_count_7d'] = len(patient_ed_before[patient_ed_before['timestamp'] > (obs_time - timedelta(days=7))])
            feat['ed_count_30d'] = len(patient_ed_before[patient_ed_before['timestamp'] > (obs_time - timedelta(days=30))])
            feat['ed_count_90d'] = len(patient_ed_before[patient_ed_before['timestamp'] > (obs_time - timedelta(days=90))])
            feat['ed_count_365d'] = len(patient_ed_before[patient_ed_before['timestamp'] > (obs_time - timedelta(days=365))])
            feat['ed_count_all_time'] = len(patient_ed_before)
            
            # Trend: recent vs historical
            if feat['ed_count_365d'] > 0:
                feat['ed_trend_30d_vs_365d'] = feat['ed_count_30d'] / feat['ed_count_365d']
            else:
                feat['ed_trend_30d_vs_365d'] = 0.0
        else:
            # No ED history before this point
            feat['days_since_last_ed'] = -1.0  # Sentinel for "never had ED"
            feat['had_any_ed'] = 0.0
            feat['ed_count_7d'] = 0.0
            feat['ed_count_30d'] = 0.0
            feat['ed_count_90d'] = 0.0
            feat['ed_count_365d'] = 0.0
            feat['ed_count_all_time'] = 0.0
            feat['ed_trend_30d_vs_365d'] = 0.0
        
        # === DIAGNOSIS HISTORY ===
        if len(patient_dx_before) > 0:
            last_dx_time = patient_dx_before['timestamp'].max()
            feat['days_since_last_dx'] = (obs_time - last_dx_time).days
            feat['dx_count_30d'] = len(patient_dx_before[patient_dx_before['timestamp'] > (obs_time - timedelta(days=30))])
            feat['dx_count_365d'] = len(patient_dx_before[patient_dx_before['timestamp'] > (obs_time - timedelta(days=365))])
            feat['dx_count_all_time'] = len(patient_dx_before)
            feat['unique_dx_codes'] = patient_dx_before['icd9_diagnosis_cd'].nunique()
        else:
            feat['days_since_last_dx'] = -1.0
            feat['dx_count_30d'] = 0.0
            feat['dx_count_365d'] = 0.0
            feat['dx_count_all_time'] = 0.0
            feat['unique_dx_codes'] = 0.0
        
        # === PROCEDURE HISTORY ===
        if len(patient_proc_before) > 0:
            last_proc_time = patient_proc_before['timestamp'].max()
            feat['days_since_last_proc'] = (obs_time - last_proc_time).days
            feat['proc_count_30d'] = len(patient_proc_before[patient_proc_before['timestamp'] > (obs_time - timedelta(days=30))])
            feat['proc_count_365d'] = len(patient_proc_before[patient_proc_before['timestamp'] > (obs_time - timedelta(days=365))])
            feat['proc_count_all_time'] = len(patient_proc_before)
        else:
            feat['days_since_last_proc'] = -1.0
            feat['proc_count_30d'] = 0.0
            feat['proc_count_365d'] = 0.0
            feat['proc_count_all_time'] = 0.0
        
        # === SEASONALITY ===
        feat['month'] = obs_time.month
        feat['is_winter'] = 1.0 if obs_time.month in [12, 1, 2] else 0.0
        feat['is_weekend'] = 1.0 if obs_time.weekday() >= 5 else 0.0
        feat['day_of_week'] = obs_time.weekday() / 6.0
        
        # === NORMALIZE / SCALE ===
        # Days: cap at 365 and normalize to [0, 1]
        for col in ['days_since_last_ed', 'days_since_last_dx', 'days_since_last_proc']:
            if feat[col] >= 0:  # Only normalize non-sentinel values
                feat[col] = min(feat[col], 365) / 365.0
            # else: keep -1.0 as "never happened" sentinel
        
        # Counts: log1p transform for heavy-tailed distributions
        for col in ['ed_count_all_time', 'dx_count_all_time', 'proc_count_all_time', 'unique_dx_codes']:
            feat[col] = np.log1p(feat[col])  # log(1 + x)
        
        # Age: normalize to roughly [0, 1]
        feat['age'] = feat['age'] / 100.0  # Assume max age ~100
        
        # Convert to tensor
        feature_vector = torch.tensor(list(feat.values()), dtype=torch.float32)
        feature_dict[idx] = feature_vector
    
    print(f"    ✓ Created {len(feature_dict)} time-aware feature vectors")
    print(f"    Feature dimension: {len(feature_vector)}")
    
    return feature_dict, list(feat.keys())  # Return feature names too


def integrate_with_graph(time_aware_features: Dict, 
                         labels_df: pd.DataFrame,
                         target_dim: int = 128) -> Dict[int, torch.Tensor]:
    """
    Pad time-aware features to target dimension for graph construction.
    
    Args:
        time_aware_features: Dict from create_time_aware_features
        labels_df: Labels DataFrame
        target_dim: Target feature dimension (e.g., 128)
    
    Returns:
        Dict mapping patient_id to padded feature tensor
    """
    # Group by patient_id and use most recent observation's features
    # (or could average, or keep all - depends on model design)
    
    patient_features = {}
    
    for patient_id in labels_df['patient_id'].unique():
        # Get all feature vectors for this patient
        patient_rows = labels_df[labels_df['patient_id'] == patient_id].index
        patient_vecs = [time_aware_features[idx] for idx in patient_rows if idx in time_aware_features]
        
        if patient_vecs:
            # Use most recent (last) observation's features
            feat_vec = patient_vecs[-1]
            
            # Pad to target dimension
            if len(feat_vec) < target_dim:
                padding = torch.zeros(target_dim - len(feat_vec))
                feat_vec = torch.cat([feat_vec, padding])
            elif len(feat_vec) > target_dim:
                feat_vec = feat_vec[:target_dim]
            
            patient_features[patient_id] = feat_vec
        else:
            # Fallback
            patient_features[patient_id] = torch.zeros(target_dim)
    
    return patient_features

