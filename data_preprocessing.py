"""
Data Preprocessing and Time-Based Splitting
Handles loading augmented data and creating train/val/test splits with no leakage
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import pickle
from typing import Dict, Tuple, List
import config

def parse_date(date_str):
    """Parse date string with multiple format support, handle ### and empty"""
    if pd.isna(date_str) or date_str == '' or date_str == '#######':
        return None
    
    # Try multiple formats
    for fmt in ['%m/%d/%Y', '%m/%d/%y', '%Y-%m-%d', '%m-%d-%Y']:
        try:
            return datetime.strptime(str(date_str).strip(), fmt)
        except:
            continue
    return None


def load_raw_data() -> Dict[str, pd.DataFrame]:
    """Load all augmented CSV files"""
    print("Loading raw data files...")
    
    data_path = Path(config.DATA_DIR)
    
    data = {
        'demographics': pd.read_csv(data_path / 'demographics.csv', index_col=0),
        'diagnosis': pd.read_csv(data_path / 'diagnosis.csv', index_col=0),
        'procedures': pd.read_csv(data_path / 'procedures.csv', index_col=0),
        'nyu_edu': pd.read_csv(data_path / 'nyu_edu.csv', index_col=0),
        'sdoh': pd.read_csv(data_path / 'sdoh.csv', sep=';', index_col=0),
        'procMapping': pd.read_csv(data_path / 'procMapping.csv', sep=';', index_col=0),
    }
    
    print(f"  demographics: {len(data['demographics'])} rows")
    print(f"  diagnosis: {len(data['diagnosis'])} rows")
    print(f"  procedures: {len(data['procedures'])} rows")
    print(f"  nyu_edu: {len(data['nyu_edu'])} rows")
    print(f"  sdoh: {len(data['sdoh'])} rows")
    print(f"  procMapping: {len(data['procMapping'])} rows")
    
    return data


def create_timestamps(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Parse and create timestamps for all temporal events
    Adds 'timestamp' column to relevant dataframes
    """
    print("Creating timestamps...")
    
    # Diagnosis timestamps
    data['diagnosis']['timestamp'] = data['diagnosis']['clm_claim_beg_svc_dt'].apply(parse_date)
    data['diagnosis'] = data['diagnosis'][data['diagnosis']['timestamp'].notna()].copy()
    print(f"  diagnosis: {len(data['diagnosis'])} with valid timestamps")
    
    # Procedure timestamps
    data['procedures']['timestamp'] = data['procedures']['svc_from_dt'].apply(parse_date)
    data['procedures'] = data['procedures'][data['procedures']['timestamp'].notna()].copy()
    print(f"  procedures: {len(data['procedures'])} with valid timestamps")
    
    # ED visit timestamps (nyu_edu)
    data['nyu_edu']['timestamp'] = data['nyu_edu']['hosp_adm_dt'].apply(parse_date)
    data['nyu_edu'] = data['nyu_edu'][data['nyu_edu']['timestamp'].notna()].copy()
    print(f"  nyu_edu: {len(data['nyu_edu'])} with valid timestamps")
    
    # Demographics - use DOB for age calculation
    data['demographics']['dob'] = data['demographics']['mbr_dob'].apply(parse_date)
    
    return data


def time_based_split(data: Dict[str, pd.DataFrame]) -> Tuple[Dict, Dict, Dict]:
    """
    Create train/val/test splits based on time cutoffs
    Ensures no patient ID leakage between splits
    """
    print("\nCreating time-based splits...")
    print(f"  Train cutoff: {config.T_CUT_TRAIN}")
    print(f"  Val cutoff: {config.T_CUT_VAL}")
    print(f"  Test cutoff: {config.T_CUT_TEST}")
    
    t_train = datetime.strptime(config.T_CUT_TRAIN, '%Y-%m-%d')
    t_val = datetime.strptime(config.T_CUT_VAL, '%Y-%m-%d')
    t_test = datetime.strptime(config.T_CUT_TEST, '%Y-%m-%d')
    
    # Collect all events with timestamps and patient IDs
    all_events = []
    
    # Diagnosis events
    for _, row in data['diagnosis'].iterrows():
        all_events.append({
            'patient_id': row['clm_sys_mbr_sk'],
            'timestamp': row['timestamp'],
            'source': 'diagnosis'
        })
    
    # Procedure events
    for _, row in data['procedures'].iterrows():
        all_events.append({
            'patient_id': row['sys_mbr_sk'],
            'timestamp': row['timestamp'],
            'source': 'procedure'
        })
    
    # ED visit events
    for _, row in data['nyu_edu'].iterrows():
        all_events.append({
            'patient_id': row['sys_mbr_sk'],
            'timestamp': row['timestamp'],
            'source': 'ed_visit'
        })
    
    events_df = pd.DataFrame(all_events)
    
    # Assign each patient to ONE split based on their FIRST event
    patient_first_event = events_df.groupby('patient_id')['timestamp'].min()
    
    train_patients = set(patient_first_event[patient_first_event < t_train].index)
    val_patients = set(patient_first_event[
        (patient_first_event >= t_train) & (patient_first_event < t_val)
    ].index)
    test_patients = set(patient_first_event[patient_first_event >= t_val].index)
    
    print(f"\n  Patient distribution:")
    print(f"    Train: {len(train_patients)} patients")
    print(f"    Val: {len(val_patients)} patients")
    print(f"    Test: {len(test_patients)} patients")
    
    # Verify no leakage
    assert len(train_patients & val_patients) == 0, "Patient leakage: train-val"
    assert len(train_patients & test_patients) == 0, "Patient leakage: train-test"
    assert len(val_patients & test_patients) == 0, "Patient leakage: val-test"
    print("  ✓ No patient ID leakage between splits")
    
    # Split data
    def split_by_patients_and_time(df, patient_col, time_col, patients, t_max):
        """Filter dataframe by patient set and time"""
        return df[
            (df[patient_col].isin(patients)) & 
            (df[time_col] < t_max)
        ].copy()
    
    train_data = {
        'diagnosis': split_by_patients_and_time(
            data['diagnosis'], 'clm_sys_mbr_sk', 'timestamp', train_patients, t_train
        ),
        'procedures': split_by_patients_and_time(
            data['procedures'], 'sys_mbr_sk', 'timestamp', train_patients, t_train
        ),
        'nyu_edu': split_by_patients_and_time(
            data['nyu_edu'], 'sys_mbr_sk', 'timestamp', train_patients, t_train
        ),
        'demographics': data['demographics'][
            data['demographics']['sys_mbr_sk'].isin(train_patients)
        ].copy(),
        'sdoh': data['sdoh'][
            data['sdoh']['lumeris_empi'].isin(
                data['demographics'][data['demographics']['sys_mbr_sk'].isin(train_patients)]['empi']
            )
        ].copy(),
        'procMapping': data['procMapping'].copy(),
        'patient_ids': train_patients
    }
    
    val_data = {
        'diagnosis': split_by_patients_and_time(
            data['diagnosis'], 'clm_sys_mbr_sk', 'timestamp', val_patients, t_val
        ),
        'procedures': split_by_patients_and_time(
            data['procedures'], 'sys_mbr_sk', 'timestamp', val_patients, t_val
        ),
        'nyu_edu': split_by_patients_and_time(
            data['nyu_edu'], 'sys_mbr_sk', 'timestamp', val_patients, t_val
        ),
        'demographics': data['demographics'][
            data['demographics']['sys_mbr_sk'].isin(val_patients)
        ].copy(),
        'sdoh': data['sdoh'][
            data['sdoh']['lumeris_empi'].isin(
                data['demographics'][data['demographics']['sys_mbr_sk'].isin(val_patients)]['empi']
            )
        ].copy(),
        'procMapping': data['procMapping'].copy(),
        'patient_ids': val_patients
    }
    
    test_data = {
        'diagnosis': data['diagnosis'][
            (data['diagnosis']['clm_sys_mbr_sk'].isin(test_patients)) & 
            (data['diagnosis']['timestamp'] >= t_val)
        ].copy(),
        'procedures': data['procedures'][
            (data['procedures']['sys_mbr_sk'].isin(test_patients)) & 
            (data['procedures']['timestamp'] >= t_val)
        ].copy(),
        'nyu_edu': data['nyu_edu'][
            (data['nyu_edu']['sys_mbr_sk'].isin(test_patients)) & 
            (data['nyu_edu']['timestamp'] >= t_val)
        ].copy(),
        'demographics': data['demographics'][
            data['demographics']['sys_mbr_sk'].isin(test_patients)
        ].copy(),
        'sdoh': data['sdoh'][
            data['sdoh']['lumeris_empi'].isin(
                data['demographics'][data['demographics']['sys_mbr_sk'].isin(test_patients)]['empi']
            )
        ].copy(),
        'procMapping': data['procMapping'].copy(),
        'patient_ids': test_patients
    }
    
    print(f"\n  Event distribution:")
    print(f"    Train: {len(train_data['diagnosis'])} dx, {len(train_data['procedures'])} proc, {len(train_data['nyu_edu'])} ED")
    print(f"    Val: {len(val_data['diagnosis'])} dx, {len(val_data['procedures'])} proc, {len(val_data['nyu_edu'])} ED")
    print(f"    Test: {len(test_data['diagnosis'])} dx, {len(test_data['procedures'])} proc, {len(test_data['nyu_edu'])} ED")
    
    return train_data, val_data, test_data


def create_labels(split_data: Dict[str, pd.DataFrame], split_name: str) -> pd.DataFrame:
    """
    Create prediction labels: time to next ED visit for each patient
    
    Returns DataFrame with columns:
    - patient_id
    - current_timestamp (observation time)
    - next_ed_timestamp (next ED visit time, or None)
    - days_to_next_ed (time delta in days, or -1 for censored)
    - has_next_ed_30d (binary: ED within 30 days)
    """
    print(f"\nCreating labels for {split_name} split...")
    
    ed_visits = split_data['nyu_edu'].copy()
    ed_visits = ed_visits.sort_values(['sys_mbr_sk', 'timestamp'])
    
    labels = []
    
    for patient_id in split_data['patient_ids']:
        patient_ed = ed_visits[ed_visits['sys_mbr_sk'] == patient_id].sort_values('timestamp')
        
        if len(patient_ed) == 0:
            # No ED visits for this patient - censored
            # Use last known event as observation time
            all_events = []
            
            diag = split_data['diagnosis'][split_data['diagnosis']['clm_sys_mbr_sk'] == patient_id]
            if len(diag) > 0:
                all_events.extend(diag['timestamp'].tolist())
            
            proc = split_data['procedures'][split_data['procedures']['sys_mbr_sk'] == patient_id]
            if len(proc) > 0:
                all_events.extend(proc['timestamp'].tolist())
            
            if all_events:
                last_event = max(all_events)
                labels.append({
                    'patient_id': patient_id,
                    'current_timestamp': last_event,
                    'next_ed_timestamp': None,
                    'days_to_next_ed': -1,  # Censored
                    'has_next_ed_30d': 0
                })
        else:
            # Create label for each ED visit except the last one
            for i in range(len(patient_ed) - 1):
                current_ed = patient_ed.iloc[i]
                next_ed = patient_ed.iloc[i + 1]
                
                days_to_next = (next_ed['timestamp'] - current_ed['timestamp']).days
                
                labels.append({
                    'patient_id': patient_id,
                    'current_timestamp': current_ed['timestamp'],
                    'next_ed_timestamp': next_ed['timestamp'],
                    'days_to_next_ed': days_to_next,
                    'has_next_ed_30d': 1 if days_to_next <= 30 else 0
                })
            
            # Last ED visit is censored
            last_ed = patient_ed.iloc[-1]
            labels.append({
                'patient_id': patient_id,
                'current_timestamp': last_ed['timestamp'],
                'next_ed_timestamp': None,
                'days_to_next_ed': -1,  # Censored
                'has_next_ed_30d': 0
            })
    
    labels_df = pd.DataFrame(labels)
    
    print(f"  Created {len(labels_df)} prediction samples")
    print(f"  Censored: {(labels_df['days_to_next_ed'] == -1).sum()}")
    print(f"  Uncensored: {(labels_df['days_to_next_ed'] > 0).sum()}")
    print(f"  ED within 30d: {labels_df['has_next_ed_30d'].sum()}")
    
    return labels_df


def preprocess_pipeline():
    """Main preprocessing pipeline"""
    print("="*80)
    print("DATA PREPROCESSING PIPELINE")
    print("="*80)
    
    # Load data
    data = load_raw_data()
    
    # Create timestamps
    data = create_timestamps(data)
    
    # Time-based splits
    train_data, val_data, test_data = time_based_split(data)
    
    # Create labels
    train_labels = create_labels(train_data, 'train')
    val_labels = create_labels(val_data, 'val')
    test_labels = create_labels(test_data, 'test')
    
    # Save processed data
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    
    print("\nSaving processed data...")
    with open(output_dir / 'train_data.pkl', 'wb') as f:
        pickle.dump((train_data, train_labels), f)
    with open(output_dir / 'val_data.pkl', 'wb') as f:
        pickle.dump((val_data, val_labels), f)
    with open(output_dir / 'test_data.pkl', 'wb') as f:
        pickle.dump((test_data, test_labels), f)
    
    print("  ✓ Saved to", output_dir)
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE")
    print("="*80)
    
    return train_data, val_data, test_data, train_labels, val_labels, test_labels


if __name__ == "__main__":
    preprocess_pipeline()

