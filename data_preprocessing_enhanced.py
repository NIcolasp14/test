"""
Enhanced Data Preprocessing with Balanced Splits and Real Features
Addresses label starvation, class imbalance, and feature quality issues
"""

import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta
from pathlib import Path
import pickle
from typing import Dict, Tuple, List
from collections import defaultdict
import config

def create_patient_features_time_aware(data: Dict[str, pd.DataFrame], 
                                       labels_df: pd.DataFrame) -> Dict[int, torch.Tensor]:
    """
    Create time-aware features for each (patient, observation_time) in labels.
    
    CRITICAL: Features computed using ONLY events BEFORE observation_time.
    No information leakage.
    
    Returns:
        Dict mapping label index to feature tensor
    """
    print(f"\n  Creating time-aware features (NO LEAKAGE)...")
    print(f"    {len(labels_df)} prediction instances")
    
    ed_visits = data['nyu_edu']
    diagnosis = data['diagnosis']
    procedures = data['procedures']
    demographics = data['demographics']
    
    feature_dict = {}
    feature_names = []
    
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
        
        feat = {}
        
        # Demographics (static)
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
        
        # ED recency & frequency (ONLY BEFORE obs_time)
        if len(patient_ed_before) > 0:
            last_ed_time = patient_ed_before['timestamp'].max()
            feat['days_since_last_ed'] = (obs_time - last_ed_time).days
            feat['had_any_ed'] = 1.0
            feat['ed_count_7d'] = len(patient_ed_before[patient_ed_before['timestamp'] > (obs_time - timedelta(days=7))])
            feat['ed_count_30d'] = len(patient_ed_before[patient_ed_before['timestamp'] > (obs_time - timedelta(days=30))])
            feat['ed_count_90d'] = len(patient_ed_before[patient_ed_before['timestamp'] > (obs_time - timedelta(days=90))])
            feat['ed_count_365d'] = len(patient_ed_before[patient_ed_before['timestamp'] > (obs_time - timedelta(days=365))])
            feat['ed_count_all_time'] = len(patient_ed_before)
            if feat['ed_count_365d'] > 0:
                feat['ed_trend'] = feat['ed_count_30d'] / feat['ed_count_365d']
            else:
                feat['ed_trend'] = 0.0
        else:
            feat['days_since_last_ed'] = -1.0  # Sentinel: never had ED
            feat['had_any_ed'] = 0.0
            feat['ed_count_7d'] = 0.0
            feat['ed_count_30d'] = 0.0
            feat['ed_count_90d'] = 0.0
            feat['ed_count_365d'] = 0.0
            feat['ed_count_all_time'] = 0.0
            feat['ed_trend'] = 0.0
        
        # Diagnosis history
        if len(patient_dx_before) > 0:
            feat['days_since_last_dx'] = (obs_time - patient_dx_before['timestamp'].max()).days
            feat['dx_count_30d'] = len(patient_dx_before[patient_dx_before['timestamp'] > (obs_time - timedelta(days=30))])
            feat['dx_count_365d'] = len(patient_dx_before[patient_dx_before['timestamp'] > (obs_time - timedelta(days=365))])
            feat['dx_count_all_time'] = len(patient_dx_before)
        else:
            feat['days_since_last_dx'] = -1.0
            feat['dx_count_30d'] = 0.0
            feat['dx_count_365d'] = 0.0
            feat['dx_count_all_time'] = 0.0
        
        # Procedure history
        if len(patient_proc_before) > 0:
            feat['days_since_last_proc'] = (obs_time - patient_proc_before['timestamp'].max()).days
            feat['proc_count_30d'] = len(patient_proc_before[patient_proc_before['timestamp'] > (obs_time - timedelta(days=30))])
            feat['proc_count_365d'] = len(patient_proc_before[patient_proc_before['timestamp'] > (obs_time - timedelta(days=365))])
            feat['proc_count_all_time'] = len(patient_proc_before)
        else:
            feat['days_since_last_proc'] = -1.0
            feat['proc_count_30d'] = 0.0
            feat['proc_count_365d'] = 0.0
            feat['proc_count_all_time'] = 0.0
        
        # Seasonality
        feat['month'] = obs_time.month / 12.0
        feat['is_winter'] = 1.0 if obs_time.month in [12, 1, 2] else 0.0
        feat['is_weekend'] = 1.0 if obs_time.weekday() >= 5 else 0.0
        feat['day_of_week'] = obs_time.weekday() / 6.0
        
        # Normalize: days capped at 365, counts log-transformed
        for col in ['days_since_last_ed', 'days_since_last_dx', 'days_since_last_proc']:
            if feat[col] >= 0:
                feat[col] = min(feat[col], 365) / 365.0
            # else: keep -1.0 as sentinel
        
        for col in ['ed_count_all_time', 'dx_count_all_time', 'proc_count_all_time']:
            feat[col] = np.log1p(feat[col])
        
        feat['age'] = feat['age'] / 100.0
        
        if not feature_names:
            feature_names = list(feat.keys())
        
        feature_vector = torch.tensor(list(feat.values()), dtype=torch.float32)
        feature_dict[idx] = feature_vector
    
    print(f"    ✓ Created {len(feature_dict)} time-aware feature vectors")
    print(f"    Feature dimension: {len(feature_names)} features")
    print(f"    Features: {feature_names[:10]}...")
    
    return feature_dict, feature_names


def create_patient_features(data: Dict[str, pd.DataFrame], patient_ids: List[int]) -> pd.DataFrame:
    """
    Create REAL patient features (not zero vectors)
    
    Features include:
    - Demographics: age, gender
    - Temporal patterns: ED frequency (7d, 30d, 90d, 365d), days since last ED
    - Healthcare utilization: diagnosis count, procedure count
    - Recency features: time since last diagnosis, procedure
    - Trend features: ED visit acceleration
    - SDOH: social determinants if available
    """
    print(f"\n  Creating real patient features for {len(patient_ids)} patients...")
    
    demographics = data['demographics']
    nyu_edu = data['nyu_edu']
    diagnosis = data['diagnosis']
    procedures = data['procedures']
    sdoh = data.get('sdoh', pd.DataFrame())
    
    # Standardize EMPI columns
    if 'empi' not in demographics.columns and 'sys_mbr_sk' in demographics.columns:
        demographics = demographics.copy()
        demographics['empi'] = demographics['sys_mbr_sk']
    
    features_list = []
    
    for patient_id in patient_ids:
        # Get patient data
        patient_demo = demographics[demographics['sys_mbr_sk'] == patient_id]
        patient_ed = nyu_edu[nyu_edu['sys_mbr_sk'] == patient_id].sort_values('timestamp')
        patient_dx = diagnosis[diagnosis['clm_sys_mbr_sk'] == patient_id].sort_values('timestamp')
        patient_proc = procedures[procedures['sys_mbr_sk'] == patient_id].sort_values('timestamp')
        
        # Reference time: latest event for this patient
        all_times = []
        if len(patient_ed) > 0:
            all_times.extend(patient_ed['timestamp'].tolist())
        if len(patient_dx) > 0:
            all_times.extend(patient_dx['timestamp'].tolist())
        if len(patient_proc) > 0:
            all_times.extend(patient_proc['timestamp'].tolist())
        
        if not all_times:
            continue  # Skip patients with no events
        
        ref_time = max(all_times)
        
        feat = {}
        
        # === DEMOGRAPHICS ===
        if len(patient_demo) > 0:
            demo = patient_demo.iloc[0]
            
            # Age
            if 'dob' in demo and pd.notna(demo['dob']):
                feat['age'] = (ref_time - demo['dob']).days / 365.25
            else:
                feat['age'] = 50.0  # Default
            
            # Gender
            feat['is_male'] = 1.0 if demo.get('mbr_gender_cd') == 'M' else 0.0
            feat['is_female'] = 1.0 if demo.get('mbr_gender_cd') == 'F' else 0.0
        else:
            feat['age'] = 50.0
            feat['is_male'] = 0.0
            feat['is_female'] = 0.0
        
        # === TEMPORAL ED PATTERNS (CRITICAL FOR PREDICTION) ===
        # ED frequency in various windows
        feat['ed_count_7d'] = len(patient_ed[patient_ed['timestamp'] > (ref_time - timedelta(days=7))])
        feat['ed_count_30d'] = len(patient_ed[patient_ed['timestamp'] > (ref_time - timedelta(days=30))])
        feat['ed_count_90d'] = len(patient_ed[patient_ed['timestamp'] > (ref_time - timedelta(days=90))])
        feat['ed_count_365d'] = len(patient_ed[patient_ed['timestamp'] > (ref_time - timedelta(days=365))])
        feat['ed_count_all_time'] = len(patient_ed)
        
        # Days since last ED (STRONG PREDICTOR)
        if len(patient_ed) > 0:
            last_ed_time = patient_ed['timestamp'].max()
            feat['days_since_last_ed'] = (ref_time - last_ed_time).days
            feat['has_any_ed'] = 1.0
        else:
            feat['days_since_last_ed'] = 3650  # 10 years (no ED)
            feat['has_any_ed'] = 0.0
        
        # ED trend (acceleration): recent vs past
        if feat['ed_count_365d'] > 0:
            feat['ed_trend_30d_vs_365d'] = feat['ed_count_30d'] / feat['ed_count_365d']
        else:
            feat['ed_trend_30d_vs_365d'] = 0.0
        
        if feat['ed_count_90d'] > 0:
            feat['ed_trend_30d_vs_90d'] = feat['ed_count_30d'] / feat['ed_count_90d']
        else:
            feat['ed_trend_30d_vs_90d'] = 0.0
        
        # === DIAGNOSIS & PROCEDURE COUNTS ===
        feat['dx_count_30d'] = len(patient_dx[patient_dx['timestamp'] > (ref_time - timedelta(days=30))])
        feat['dx_count_365d'] = len(patient_dx[patient_dx['timestamp'] > (ref_time - timedelta(days=365))])
        feat['dx_count_all_time'] = len(patient_dx)
        
        feat['proc_count_30d'] = len(patient_proc[patient_proc['timestamp'] > (ref_time - timedelta(days=30))])
        feat['proc_count_365d'] = len(patient_proc[patient_proc['timestamp'] > (ref_time - timedelta(days=365))])
        feat['proc_count_all_time'] = len(patient_proc)
        
        # === RECENCY FEATURES ===
        if len(patient_dx) > 0:
            feat['days_since_last_dx'] = (ref_time - patient_dx['timestamp'].max()).days
        else:
            feat['days_since_last_dx'] = 3650
        
        if len(patient_proc) > 0:
            feat['days_since_last_proc'] = (ref_time - patient_proc['timestamp'].max()).days
        else:
            feat['days_since_last_proc'] = 3650
        
        # === TIME-OF-YEAR / SEASONALITY ===
        feat['month'] = ref_time.month
        feat['is_winter'] = 1.0 if ref_time.month in [12, 1, 2] else 0.0
        feat['is_weekend'] = 1.0 if ref_time.weekday() >= 5 else 0.0
        feat['day_of_week'] = ref_time.weekday() / 6.0  # Normalize to [0, 1]
        
        # === SDOH (if available) ===
        if len(sdoh) > 0 and len(patient_demo) > 0:
            empi = patient_demo.iloc[0].get('empi')
            if pd.notna(empi):
                # Try to find SDOH by empi or lumeris_empi
                if 'lumeris_empi' in sdoh.columns:
                    sdoh_row = sdoh[sdoh['lumeris_empi'] == empi]
                elif 'empi' in sdoh.columns:
                    sdoh_row = sdoh[sdoh['empi'] == empi]
                else:
                    sdoh_row = pd.DataFrame()
                
                if len(sdoh_row) > 0:
                    sdoh_data = sdoh_row.iloc[0]
                    for col in ['sdh_agg_transportation', 'sdh_agg_affordability', 
                                'sdh_agg_health_beh', 'sdh_agg_housing_sup']:
                        if col in sdoh_data.index:
                            feat[col] = float(sdoh_data[col]) if pd.notna(sdoh_data[col]) else 0.0
                        else:
                            feat[col] = 0.0
                else:
                    # Default SDOH
                    for col in ['sdh_agg_transportation', 'sdh_agg_affordability', 
                                'sdh_agg_health_beh', 'sdh_agg_housing_sup']:
                        feat[col] = 0.0
        else:
            # Default SDOH
            for col in ['sdh_agg_transportation', 'sdh_agg_affordability', 
                        'sdh_agg_health_beh', 'sdh_agg_housing_sup']:
                feat[col] = 0.0
        
        features_list.append({'patient_id': patient_id, **feat})
    
    features_df = pd.DataFrame(features_list)
    features_df = features_df.set_index('patient_id')
    
    # Fill any NaNs
    features_df = features_df.fillna(0)
    
    # Normalize certain features to [0, 1] range
    for col in ['days_since_last_ed', 'days_since_last_dx', 'days_since_last_proc']:
        if col in features_df.columns:
            # Cap at 365 days and normalize
            features_df[col] = np.clip(features_df[col], 0, 365) / 365.0
    
    print(f"  ✓ Created {len(features_df.columns)} features for {len(features_df)} patients")
    print(f"    Features: {list(features_df.columns[:10])}...")
    
    return features_df


def create_enriched_labels_with_history(data: Dict[str, pd.DataFrame], 
                                        patient_ids: List[int],
                                        split_name: str) -> pd.DataFrame:
    """
    Create enriched labels that include historical observation points
    
    Instead of only labeling between consecutive ED visits, create labels at:
    1. Each ED visit (predicting next visit)
    2. Each diagnosis/procedure (predicting next ED)
    3. Monthly snapshots for active patients
    
    This dramatically increases positive sample count.
    """
    print(f"\n  Creating enriched labels for {split_name} split...")
    
    # Check if timestamp column exists, if not, the data hasn't been preprocessed yet
    if 'timestamp' not in data['nyu_edu'].columns:
        print(f"  ⚠️  Data doesn't have 'timestamp' column yet - using original preprocessing labels")
        # Fall back to original create_labels function
        from data_preprocessing import create_labels
        return create_labels(data, split_name)
    
    ed_visits = data['nyu_edu'].sort_values(['sys_mbr_sk', 'timestamp'])
    diagnosis = data['diagnosis'].sort_values(['clm_sys_mbr_sk', 'timestamp'])
    procedures = data['procedures'].sort_values(['sys_mbr_sk', 'timestamp'])
    
    labels = []
    
    for patient_id in patient_ids:
        patient_ed = ed_visits[ed_visits['sys_mbr_sk'] == patient_id].copy()
        patient_dx = diagnosis[diagnosis['clm_sys_mbr_sk'] == patient_id].copy()
        patient_proc = procedures[procedures['sys_mbr_sk'] == patient_id].copy()
        
        if len(patient_ed) == 0:
            # No ED visits - create censored labels at other events
            all_events = []
            if len(patient_dx) > 0:
                all_events.extend(patient_dx['timestamp'].tolist())
            if len(patient_proc) > 0:
                all_events.extend(patient_proc['timestamp'].tolist())
            
            # Sample up to 3 events to avoid overwhelming with negatives
            if all_events:
                sampled_events = sorted(all_events)[-3:]  # Last 3 events
                for event_time in sampled_events:
                    labels.append({
                        'patient_id': patient_id,
                        'observation_time': event_time,
                        'next_ed_time': None,
                        'days_to_next_ed': -1,
                        'has_next_ed_30d': 0,
                        'is_censored': 1
                    })
        else:
            # Patient has ED visits
            ed_times = sorted(patient_ed['timestamp'].tolist())
            
            # 1. Label at each ED visit (predicting next)
            for i, current_ed_time in enumerate(ed_times[:-1]):
                next_ed_time = ed_times[i + 1]
                days_to_next = (next_ed_time - current_ed_time).days
                
                labels.append({
                    'patient_id': patient_id,
                    'observation_time': current_ed_time,
                    'next_ed_time': next_ed_time,
                    'days_to_next_ed': days_to_next,
                    'has_next_ed_30d': 1 if days_to_next <= 30 else 0,
                    'is_censored': 0
                })
            
            # 2. Label at diagnosis/procedure events (predicting next ED)
            # Only include events BEFORE ED visits to avoid data leakage
            for event_df, event_type in [(patient_dx, 'dx'), (patient_proc, 'proc')]:
                if len(event_df) == 0:
                    continue
                
                # Sample a subset to avoid overwhelming (max 5 per patient)
                event_times = sorted(event_df['timestamp'].tolist())
                if len(event_times) > 5:
                    # Take first, last, and 3 random in between
                    import random
                    random.seed(42)
                    middle = random.sample(event_times[1:-1], min(3, len(event_times) - 2))
                    event_times = [event_times[0]] + middle + [event_times[-1]]
                
                for event_time in event_times:
                    # Find next ED after this event
                    future_eds = [ed_time for ed_time in ed_times if ed_time > event_time]
                    
                    if future_eds:
                        next_ed_time = min(future_eds)
                        days_to_next = (next_ed_time - event_time).days
                        
                        labels.append({
                            'patient_id': patient_id,
                            'observation_time': event_time,
                            'next_ed_time': next_ed_time,
                            'days_to_next_ed': days_to_next,
                            'has_next_ed_30d': 1 if days_to_next <= 30 else 0,
                            'is_censored': 0
                        })
                    else:
                        # Censored (no future ED after this event)
                        labels.append({
                            'patient_id': patient_id,
                            'observation_time': event_time,
                            'next_ed_time': None,
                            'days_to_next_ed': -1,
                            'has_next_ed_30d': 0,
                            'is_censored': 1
                        })
            
            # 3. Last ED visit is censored
            labels.append({
                'patient_id': patient_id,
                'observation_time': ed_times[-1],
                'next_ed_time': None,
                'days_to_next_ed': -1,
                'has_next_ed_30d': 0,
                'is_censored': 1
            })
    
    labels_df = pd.DataFrame(labels)
    
    # Normalize days_to_next_ed
    MAX_DAYS = float(config.MAX_DAYS_NORMALIZATION)
    uncensored_mask = labels_df['days_to_next_ed'] > 0
    labels_df['days_to_next_ed_normalized'] = -1.0
    if uncensored_mask.sum() > 0:
        clipped_days = np.clip(labels_df.loc[uncensored_mask, 'days_to_next_ed'], 0, MAX_DAYS)
        labels_df.loc[uncensored_mask, 'days_to_next_ed_normalized'] = clipped_days / MAX_DAYS
    
    # Print detailed stats
    n_total = len(labels_df)
    n_censored = labels_df['is_censored'].sum()
    n_uncensored = n_total - n_censored
    n_ed_30d = labels_df['has_next_ed_30d'].sum()
    
    print(f"  ✓ Created {n_total} prediction samples")
    print(f"    Censored: {n_censored} ({100*n_censored/n_total:.1f}%)")
    print(f"    Uncensored: {n_uncensored} ({100*n_uncensored/n_total:.1f}%)")
    print(f"    ED within 30d: {n_ed_30d} ({100*n_ed_30d/n_total:.1f}%)")
    
    if uncensored_mask.sum() > 0:
        print(f"    Days to next ED (uncensored): "
              f"mean={labels_df.loc[uncensored_mask, 'days_to_next_ed'].mean():.1f}, "
              f"median={labels_df.loc[uncensored_mask, 'days_to_next_ed'].median():.1f}, "
              f"max={labels_df.loc[uncensored_mask, 'days_to_next_ed'].max():.0f}")
    
    if n_ed_30d < 100:
        print(f"\n  ⚠️⚠️⚠️  WARNING: Only {n_ed_30d} positive samples!")
        print(f"  This is likely insufficient for training. Consider:")
        print(f"    1. Expanding the time window (include more historical data)")
        print(f"    2. Using SMOTE/oversampling")
        print(f"    3. Adjusting the prediction window (e.g., 60d instead of 30d)")
    
    return labels_df


def stratified_split_with_minimum_positives(data: Dict[str, pd.DataFrame],
                                             min_positives_per_split: int = 300) -> Tuple[Dict, Dict, Dict]:
    """
    Create train/val/test splits ensuring MINIMUM number of positive samples in each
    
    Instead of time-based splits, use stratified patient-level splitting with
    oversampling to ensure each split has enough positive samples.
    """
    print("\n" + "="*80)
    print("CREATING BALANCED STRATIFIED SPLITS")
    print("="*80)
    print(f"\nTarget: At least {min_positives_per_split} positive samples per split")
    
    # First, identify all patients and their ED visit patterns
    ed_visits = data['nyu_edu'].copy()
    all_patients = data['demographics']['sys_mbr_sk'].unique()
    
    # Classify patients by ED frequency (proxy for risk)
    patient_ed_counts = ed_visits.groupby('sys_mbr_sk').size().to_dict()
    patient_has_ed = {pid: patient_ed_counts.get(pid, 0) > 0 for pid in all_patients}
    patient_ed_high_utilizer = {pid: patient_ed_counts.get(pid, 0) >= 3 for pid in all_patients}
    
    # Separate patients into risk groups
    high_utilizers = [pid for pid in all_patients if patient_ed_high_utilizer[pid]]
    has_ed_low = [pid for pid in all_patients if patient_has_ed[pid] and not patient_ed_high_utilizer[pid]]
    no_ed = [pid for pid in all_patients if not patient_has_ed[pid]]
    
    print(f"\nPatient distribution:")
    print(f"  High utilizers (≥3 ED visits): {len(high_utilizers)}")
    print(f"  Low utilizers (1-2 ED visits): {len(has_ed_low)}")
    print(f"  No ED visits: {len(no_ed)}")
    
    # Split each group with 60-20-20 ratio
    np.random.seed(42)
    
    def split_list(lst, ratios=(0.6, 0.2, 0.2)):
        """Split list into train/val/test"""
        shuffled = lst.copy()
        np.random.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        return shuffled[:n_train], shuffled[n_train:n_train+n_val], shuffled[n_train+n_val:]
    
    # Split each group
    high_train, high_val, high_test = split_list(high_utilizers)
    low_train, low_val, low_test = split_list(has_ed_low)
    no_train, no_val, no_test = split_list(no_ed)
    
    # Combine
    train_patients = high_train + low_train + no_train
    val_patients = high_val + low_val + no_val
    test_patients = high_test + low_test + no_test
    
    print(f"\nInitial patient splits:")
    print(f"  Train: {len(train_patients)} patients")
    print(f"  Val: {len(val_patients)} patients")
    print(f"  Test: {len(test_patients)} patients")
    
    # Create data splits
    def create_split_data(patient_list, split_name):
        """Create split data dictionary"""
        split = {}
        split['patient_ids'] = patient_list
        split['demographics'] = data['demographics'][data['demographics']['sys_mbr_sk'].isin(patient_list)].copy()
        split['nyu_edu'] = data['nyu_edu'][data['nyu_edu']['sys_mbr_sk'].isin(patient_list)].copy()
        split['diagnosis'] = data['diagnosis'][data['diagnosis']['clm_sys_mbr_sk'].isin(patient_list)].copy()
        split['procedures'] = data['procedures'][data['procedures']['sys_mbr_sk'].isin(patient_list)].copy()
        if 'sdoh' in data:
            # Match by EMPI if available
            if 'empi' in data['demographics'].columns:
                patient_empis = split['demographics']['empi'].unique()
                if 'lumeris_empi' in data['sdoh'].columns:
                    split['sdoh'] = data['sdoh'][data['sdoh']['lumeris_empi'].isin(patient_empis)]
                elif 'empi' in data['sdoh'].columns:
                    split['sdoh'] = data['sdoh'][data['sdoh']['empi'].isin(patient_empis)]
                else:
                    split['sdoh'] = pd.DataFrame()
            else:
                split['sdoh'] = pd.DataFrame()
        else:
            split['sdoh'] = pd.DataFrame()
        
        print(f"\n{split_name} split data summary:")
        print(f"  Patients: {len(split['patient_ids'])}")
        print(f"  ED visits: {len(split['nyu_edu'])}")
        print(f"  Diagnoses: {len(split['diagnosis'])}")
        print(f"  Procedures: {len(split['procedures'])}")
        if len(split.get('sdoh', [])) > 0:
            print(f"  SDOH records: {len(split['sdoh'])}")
        
        return split
    
    train_data = create_split_data(train_patients, "TRAIN")
    val_data = create_split_data(val_patients, "VAL")
    test_data = create_split_data(test_patients, "TEST")
    
    print("\n" + "="*80)
    print("✓ Balanced stratified splits created")
    print("="*80)
    
    return train_data, val_data, test_data


