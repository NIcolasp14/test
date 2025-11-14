"""
Advanced Feature Engineering for ED Utilization Prediction
Adds temporal, recency, and frequency features to improve model performance
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import config


def compute_ed_recency_features(data_dict):
    """
    Compute recency features from ED visit history
    
    Features:
    - days_since_last_ed: Days since most recent ED visit
    - ed_count_30d: Number of ED visits in last 30 days
    - ed_count_90d: Number of ED visits in last 90 days
    - ed_count_365d: Number of ED visits in last 365 days
    - ed_frequency: Average ED visits per month
    - time_to_first_ed: Days from DOB to first ED visit
    - ed_trend: Slope of ED visits over time (increasing/decreasing)
    
    Args:
        data_dict: Dictionary containing 'nyu_edu' DataFrame with ED visits
    
    Returns:
        ed_features: DataFrame with patient-level ED recency features
    """
    print("\n  Computing ED recency features...")
    
    ed_visits = data_dict['nyu_edu'].copy()
    
    if len(ed_visits) == 0:
        print("    Warning: No ED visits found!")
        return pd.DataFrame()
    
    ed_visits = ed_visits.sort_values(['sys_mbr_sk', 'timestamp'])
    
    features = []
    
    for patient_id in ed_visits['sys_mbr_sk'].unique():
        patient_ed = ed_visits[ed_visits['sys_mbr_sk'] == patient_id].sort_values('timestamp')
        
        if len(patient_ed) == 0:
            continue
        
        # Get time range
        first_ed = patient_ed['timestamp'].min()
        last_ed = patient_ed['timestamp'].max()
        observation_end = data_dict.get('observation_end', last_ed)
        
        # Days since last ED (as of observation end)
        days_since_last_ed = (observation_end - last_ed).days
        
        # ED counts in different time windows (looking back from last ED)
        ed_count_30d = len(patient_ed[patient_ed['timestamp'] >= last_ed - timedelta(days=30)])
        ed_count_90d = len(patient_ed[patient_ed['timestamp'] >= last_ed - timedelta(days=90)])
        ed_count_365d = len(patient_ed[patient_ed['timestamp'] >= last_ed - timedelta(days=365)])
        
        # ED frequency (visits per month)
        observation_days = (last_ed - first_ed).days + 1
        observation_months = max(observation_days / 30.0, 1.0)
        ed_frequency = len(patient_ed) / observation_months
        
        # Time to first ED from birth/enrollment
        patient_demo = data_dict['demographics'][data_dict['demographics']['sys_mbr_sk'] == patient_id]
        if len(patient_demo) > 0 and 'dob' in patient_demo.columns:
            dob = patient_demo.iloc[0]['dob']
            time_to_first_ed = (first_ed - dob).days if pd.notna(dob) else -1
        else:
            time_to_first_ed = -1
        
        # ED trend (slope of visits over time)
        # Simple linear trend: count visits in first half vs second half
        midpoint = first_ed + (last_ed - first_ed) / 2
        ed_first_half = len(patient_ed[patient_ed['timestamp'] < midpoint])
        ed_second_half = len(patient_ed[patient_ed['timestamp'] >= midpoint])
        ed_trend = (ed_second_half - ed_first_half) / max(observation_months, 1.0)
        
        # Seasonality: month of most recent ED
        month_of_last_ed = last_ed.month
        
        features.append({
            'patient_id': patient_id,
            'days_since_last_ed': days_since_last_ed,
            'ed_count_30d': ed_count_30d,
            'ed_count_90d': ed_count_90d,
            'ed_count_365d': ed_count_365d,
            'ed_frequency': ed_frequency,
            'time_to_first_ed': time_to_first_ed,
            'ed_trend': ed_trend,
            'month_of_last_ed': month_of_last_ed,
            'total_ed_visits': len(patient_ed)
        })
    
    ed_features_df = pd.DataFrame(features)
    
    print(f"    ✓ Created ED recency features for {len(ed_features_df)} patients")
    print(f"      Mean days since last ED: {ed_features_df['days_since_last_ed'].mean():.1f}")
    print(f"      Mean ED frequency (per month): {ed_features_df['ed_frequency'].mean():.2f}")
    
    return ed_features_df


def compute_diagnosis_features(data_dict):
    """
    Compute features from diagnosis history
    
    Features:
    - unique_dx_count: Number of unique diagnosis codes
    - dx_count_30d: Number of diagnoses in last 30 days
    - days_since_last_dx: Days since most recent diagnosis
    - charlson_index: Charlson comorbidity index (if available)
    - has_chronic_condition: Binary flag for chronic conditions
    
    Args:
        data_dict: Dictionary containing 'diagnosis' DataFrame
    
    Returns:
        dx_features: DataFrame with patient-level diagnosis features
    """
    print("\n  Computing diagnosis features...")
    
    diagnosis = data_dict['diagnosis'].copy()
    
    if len(diagnosis) == 0:
        print("    Warning: No diagnoses found!")
        return pd.DataFrame()
    
    features = []
    
    for patient_id in diagnosis['clm_sys_mbr_sk'].unique():
        patient_dx = diagnosis[diagnosis['clm_sys_mbr_sk'] == patient_id]
        
        # Unique diagnosis codes
        unique_dx = patient_dx['dx_code'].nunique() if 'dx_code' in patient_dx.columns else 0
        
        # Recent diagnoses
        last_dx = patient_dx['timestamp'].max()
        observation_end = data_dict.get('observation_end', last_dx)
        days_since_last_dx = (observation_end - last_dx).days
        
        dx_count_30d = len(patient_dx[patient_dx['timestamp'] >= last_dx - timedelta(days=30)])
        
        features.append({
            'patient_id': patient_id,
            'unique_dx_count': unique_dx,
            'dx_count_30d': dx_count_30d,
            'days_since_last_dx': days_since_last_dx,
            'total_dx_count': len(patient_dx)
        })
    
    dx_features_df = pd.DataFrame(features)
    
    print(f"    ✓ Created diagnosis features for {len(dx_features_df)} patients")
    print(f"      Mean unique diagnoses: {dx_features_df['unique_dx_count'].mean():.1f}")
    
    return dx_features_df


def compute_procedure_features(data_dict):
    """
    Compute features from procedure history
    
    Features:
    - unique_proc_count: Number of unique procedure codes
    - proc_count_30d: Number of procedures in last 30 days
    - days_since_last_proc: Days since most recent procedure
    
    Args:
        data_dict: Dictionary containing 'procedures' DataFrame
    
    Returns:
        proc_features: DataFrame with patient-level procedure features
    """
    print("\n  Computing procedure features...")
    
    procedures = data_dict['procedures'].copy()
    
    if len(procedures) == 0:
        print("    Warning: No procedures found!")
        return pd.DataFrame()
    
    features = []
    
    for patient_id in procedures['sys_mbr_sk'].unique():
        patient_proc = procedures[procedures['sys_mbr_sk'] == patient_id]
        
        # Unique procedure codes
        unique_proc = patient_proc['proc_code'].nunique() if 'proc_code' in patient_proc.columns else 0
        
        # Recent procedures
        last_proc = patient_proc['timestamp'].max()
        observation_end = data_dict.get('observation_end', last_proc)
        days_since_last_proc = (observation_end - last_proc).days
        
        proc_count_30d = len(patient_proc[patient_proc['timestamp'] >= last_proc - timedelta(days=30)])
        
        features.append({
            'patient_id': patient_id,
            'unique_proc_count': unique_proc,
            'proc_count_30d': proc_count_30d,
            'days_since_last_proc': days_since_last_proc,
            'total_proc_count': len(patient_proc)
        })
    
    proc_features_df = pd.DataFrame(features)
    
    print(f"    ✓ Created procedure features for {len(proc_features_df)} patients")
    print(f"      Mean unique procedures: {proc_features_df['unique_proc_count'].mean():.1f}")
    
    return proc_features_df


def compute_temporal_features(data_dict):
    """
    Compute temporal/seasonal features
    
    Features:
    - is_winter: ED visit in winter months (Dec, Jan, Feb)
    - is_summer: ED visit in summer months (Jun, Jul, Aug)
    - day_of_week: Day of week of most recent event
    - days_enrolled: Days since enrollment/first event
    
    Args:
        data_dict: Dictionary with patient data
    
    Returns:
        temporal_features: DataFrame with temporal features
    """
    print("\n  Computing temporal features...")
    
    features = []
    
    # Use demographics for patient list
    demographics = data_dict['demographics']
    
    for _, patient in demographics.iterrows():
        patient_id = patient['sys_mbr_sk']
        
        # Get most recent event timestamp
        all_timestamps = []
        
        if 'nyu_edu' in data_dict:
            patient_ed = data_dict['nyu_edu'][data_dict['nyu_edu']['sys_mbr_sk'] == patient_id]
            if len(patient_ed) > 0:
                all_timestamps.extend(patient_ed['timestamp'].tolist())
        
        if 'diagnosis' in data_dict:
            patient_dx = data_dict['diagnosis'][data_dict['diagnosis']['clm_sys_mbr_sk'] == patient_id]
            if len(patient_dx) > 0:
                all_timestamps.extend(patient_dx['timestamp'].tolist())
        
        if len(all_timestamps) == 0:
            continue
        
        most_recent = max(all_timestamps)
        first_event = min(all_timestamps)
        
        # Seasonal features
        month = most_recent.month
        is_winter = 1 if month in [12, 1, 2] else 0
        is_summer = 1 if month in [6, 7, 8] else 0
        is_spring = 1 if month in [3, 4, 5] else 0
        is_fall = 1 if month in [9, 10, 11] else 0
        
        # Day of week (0 = Monday, 6 = Sunday)
        day_of_week = most_recent.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0
        
        # Days enrolled
        days_enrolled = (most_recent - first_event).days
        
        features.append({
            'patient_id': patient_id,
            'is_winter': is_winter,
            'is_summer': is_summer,
            'is_spring': is_spring,
            'is_fall': is_fall,
            'day_of_week': day_of_week,
            'is_weekend': is_weekend,
            'days_enrolled': days_enrolled
        })
    
    temporal_features_df = pd.DataFrame(features)
    
    print(f"    ✓ Created temporal features for {len(temporal_features_df)} patients")
    
    return temporal_features_df


def engineer_all_features(data_dict):
    """
    Compute all engineered features and merge them
    
    Args:
        data_dict: Dictionary with raw patient data
    
    Returns:
        features_df: DataFrame with all engineered features
    """
    print("\n" + "="*80)
    print("FEATURE ENGINEERING")
    print("="*80)
    
    # Compute different feature sets
    ed_features = compute_ed_recency_features(data_dict)
    dx_features = compute_diagnosis_features(data_dict)
    proc_features = compute_procedure_features(data_dict)
    temporal_features = compute_temporal_features(data_dict)
    
    # Merge all features
    print("\n  Merging feature sets...")
    
    # Start with demographics as base
    features_df = data_dict['demographics'][['sys_mbr_sk']].copy()
    features_df = features_df.rename(columns={'sys_mbr_sk': 'patient_id'})
    
    # Merge ED features
    if len(ed_features) > 0:
        features_df = features_df.merge(ed_features, on='patient_id', how='left')
    
    # Merge diagnosis features
    if len(dx_features) > 0:
        features_df = features_df.merge(dx_features, on='patient_id', how='left')
    
    # Merge procedure features
    if len(proc_features) > 0:
        features_df = features_df.merge(proc_features, on='patient_id', how='left')
    
    # Merge temporal features
    if len(temporal_features) > 0:
        features_df = features_df.merge(temporal_features, on='patient_id', how='left')
    
    # Fill missing values
    features_df = features_df.fillna(0)
    
    print(f"\n  ✓ Final feature set: {len(features_df)} patients, {len(features_df.columns)-1} features")
    print(f"    Features: {', '.join([c for c in features_df.columns if c != 'patient_id'][:10])}...")
    
    return features_df


if __name__ == "__main__":
    print("Feature engineering module loaded successfully")
    print("Functions:")
    print("  - engineer_all_features(data_dict)")
    print("  - compute_ed_recency_features(data_dict)")
    print("  - compute_diagnosis_features(data_dict)")
    print("  - compute_procedure_features(data_dict)")
    print("  - compute_temporal_features(data_dict)")
