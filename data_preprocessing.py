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
    
    # Helper function to read CSV with fallback options
    def read_csv_robust(filepath, **kwargs):
        """Try reading CSV with multiple fallback options"""
        def validate_dataframe(df, filename):
            """Check if dataframe was read correctly"""
            if len(df.columns) == 0:
                print(f"  ⚠️  Warning: {filename} has no columns after parsing!")
                return False
            if len(df) == 0:
                print(f"  ⚠️  Warning: {filename} has no rows after parsing!")
                return False
            return True
        
        try:
            df = pd.read_csv(filepath, **kwargs)
            if validate_dataframe(df, filepath.name):
                return df
            else:
                raise ValueError(f"DataFrame validation failed for {filepath.name}")
        except Exception as e:
            print(f"  Warning: Error reading {filepath.name} with initial params: {e}")
            
            # Diagnose the file
            try:
                print(f"  Diagnosing file format...")
                with open(filepath, 'r', encoding='utf-8') as f:
                    first_lines = [f.readline().strip() for _ in range(5)]
                print(f"  First 5 lines of {filepath.name}:")
                for i, line in enumerate(first_lines, 1):
                    print(f"    Line {i}: {line[:100]}")  # Show first 100 chars
            except Exception as diag_error:
                print(f"  Could not diagnose file: {diag_error}")
            
            # Try without index_col
            try:
                print(f"  Attempting: without index_col...")
                kwargs_copy = kwargs.copy()
                kwargs_copy.pop('index_col', None)
                df = pd.read_csv(filepath, **kwargs_copy)
                if validate_dataframe(df, filepath.name):
                    return df
                else:
                    raise ValueError("No columns after removing index_col")
            except Exception as e2:
                print(f"  Failed: {e2}")
                
                # Try with different separator
                try:
                    print(f"  Attempting: comma separator instead of semicolon...")
                    kwargs_copy = kwargs.copy()
                    kwargs_copy['sep'] = ','
                    kwargs_copy.pop('index_col', None)  # Also remove index_col
                    df = pd.read_csv(filepath, **kwargs_copy)
                    if validate_dataframe(df, filepath.name):
                        return df
                    else:
                        raise ValueError("No columns with comma separator")
                except Exception as e3:
                    print(f"  Failed: {e3}")
                    
                    # Try with on_bad_lines='skip' (pandas 1.3+)
                    try:
                        print(f"  Attempting: skip bad lines...")
                        kwargs_copy = kwargs.copy()
                        kwargs_copy['on_bad_lines'] = 'skip'
                        kwargs_copy['sep'] = kwargs.get('sep', ',')
                        kwargs_copy.pop('index_col', None)
                        df = pd.read_csv(filepath, **kwargs_copy)
                        if validate_dataframe(df, filepath.name):
                            return df
                        else:
                            raise ValueError("No columns after skipping bad lines")
                    except:
                        # Final fallback: try with error_bad_lines=False for older pandas
                        try:
                            print(f"  Attempting: error_bad_lines=False (older pandas)...")
                            kwargs_copy = kwargs.copy()
                            kwargs_copy['error_bad_lines'] = False
                            kwargs_copy['warn_bad_lines'] = True
                            kwargs_copy.pop('index_col', None)
                            df = pd.read_csv(filepath, **kwargs_copy)
                            if validate_dataframe(df, filepath.name):
                                return df
                            else:
                                raise ValueError("No columns with error_bad_lines=False")
                        except Exception as final_error:
                            print(f"  All attempts failed. Last error: {final_error}")
                            raise
    
    data = {
        'demographics': read_csv_robust(data_path / 'demographics.csv', index_col=0),
        'diagnosis': read_csv_robust(data_path / 'diagnosis.csv', index_col=0),
        'procedures': read_csv_robust(data_path / 'procedures.csv', index_col=0),
        'nyu_edu': read_csv_robust(data_path / 'nyu_edu.csv', index_col=0),
        'sdoh': read_csv_robust(data_path / 'sdoh.csv', sep=';', index_col=0),
        'procMapping': read_csv_robust(data_path / 'procMapping.csv', sep=';', index_col=0),
    }
    
    print(f"  demographics: {len(data['demographics'])} rows, {len(data['demographics'].columns)} columns")
    print(f"  diagnosis: {len(data['diagnosis'])} rows, {len(data['diagnosis'].columns)} columns")
    print(f"  procedures: {len(data['procedures'])} rows, {len(data['procedures'].columns)} columns")
    print(f"  nyu_edu: {len(data['nyu_edu'])} rows, {len(data['nyu_edu'].columns)} columns")
    print(f"  sdoh: {len(data['sdoh'])} rows, {len(data['sdoh'].columns)} columns")
    if len(data['sdoh'].columns) > 0:
        print(f"    SDOH columns (first 5): {list(data['sdoh'].columns[:5])}")
    print(f"  procMapping: {len(data['procMapping'])} rows, {len(data['procMapping'].columns)} columns")
    
    # Detect and standardize column names
    data = standardize_column_names(data)
    
    return data


def find_column(df: pd.DataFrame, possible_names: list, df_name: str = "dataframe") -> str:
    """
    Find a column in dataframe by trying multiple possible names
    Returns the actual column name if found, None otherwise
    """
    cols = df.columns.tolist()
    
    # Try exact match first
    for name in possible_names:
        if name in cols:
            return name
    
    # Try case-insensitive match
    cols_lower = {col.lower(): col for col in cols}
    for name in possible_names:
        if name.lower() in cols_lower:
            return cols_lower[name.lower()]
    
    # Try partial match
    for name in possible_names:
        name_lower = name.lower()
        for col in cols:
            if name_lower in col.lower():
                return col
    
    return None


def standardize_column_names(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Detect actual column names and create standardized aliases
    This handles cases where the actual data has different column names
    """
    print("\nStandardizing column names...")
    
    # SDOH: Look for EMPI column
    empi_col = find_column(data['sdoh'], ['lumeris_empi', 'empi', 'EMPI', 'patient_id', 'patient_empi'], 'SDOH')
    
    if empi_col:
        if empi_col != 'lumeris_empi':
            print(f"  SDOH: Using '{empi_col}' as 'lumeris_empi'")
            data['sdoh']['lumeris_empi'] = data['sdoh'][empi_col]
        else:
            print(f"  SDOH: Found 'lumeris_empi' column ✓")
    else:
        sdoh_cols = data['sdoh'].columns.tolist()
        print(f"  Warning: No EMPI column found in SDOH.")
        print(f"    Available columns: {sdoh_cols[:10]}")
        # Try to use the first column that looks like an ID
        id_col = None
        for col in sdoh_cols:
            if any(x in str(col).lower() for x in ['id', 'key', 'sk']):
                id_col = col
                break
        
        if id_col:
            print(f"    Using '{id_col}' as fallback for 'lumeris_empi'")
            data['sdoh']['lumeris_empi'] = data['sdoh'][id_col]
        else:
            print(f"    Creating sequential IDs for 'lumeris_empi'")
            data['sdoh']['lumeris_empi'] = range(len(data['sdoh']))
    
    # Demographics: Ensure 'empi' column exists
    empi_col_demo = find_column(data['demographics'], ['empi', 'EMPI', 'lumeris_empi', 'patient_empi'], 'demographics')
    
    if empi_col_demo:
        if empi_col_demo != 'empi':
            print(f"  Demographics: Using '{empi_col_demo}' as 'empi'")
            data['demographics']['empi'] = data['demographics'][empi_col_demo]
        else:
            print(f"  Demographics: Found 'empi' column ✓")
    else:
        demo_cols = data['demographics'].columns.tolist()
        print(f"  Warning: No EMPI column found in demographics.")
        print(f"    Available columns: {demo_cols[:10]}")
        # Try to use sys_mbr_sk as a fallback
        if 'sys_mbr_sk' in demo_cols:
            print(f"    Using 'sys_mbr_sk' as 'empi' fallback")
            data['demographics']['empi'] = data['demographics']['sys_mbr_sk']
        else:
            print(f"    Creating index-based 'empi' column")
            data['demographics']['empi'] = data['demographics'].index
    
    print("  ✓ Column names standardized")
    return data


def create_timestamps(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Parse and create timestamps for all temporal events
    Adds 'timestamp' column to relevant dataframes
    """
    print("\nCreating timestamps...")
    
    # Diagnosis timestamps
    diag_date_col = find_column(data['diagnosis'], 
                                 ['clm_claim_beg_svc_dt', 'claim_date', 'service_date', 'date', 'timestamp'], 
                                 'diagnosis')
    if diag_date_col:
        print(f"  Using '{diag_date_col}' for diagnosis timestamps")
        data['diagnosis']['timestamp'] = data['diagnosis'][diag_date_col].apply(parse_date)
        data['diagnosis'] = data['diagnosis'][data['diagnosis']['timestamp'].notna()].copy()
        print(f"  diagnosis: {len(data['diagnosis'])} with valid timestamps")
    else:
        print(f"  Warning: No date column found in diagnosis. Creating dummy timestamps.")
        data['diagnosis']['timestamp'] = pd.Timestamp.now()
    
    # Procedure timestamps
    proc_date_col = find_column(data['procedures'], 
                                ['svc_from_dt', 'service_date', 'procedure_date', 'date', 'timestamp'], 
                                'procedures')
    if proc_date_col:
        print(f"  Using '{proc_date_col}' for procedure timestamps")
        data['procedures']['timestamp'] = data['procedures'][proc_date_col].apply(parse_date)
        data['procedures'] = data['procedures'][data['procedures']['timestamp'].notna()].copy()
        print(f"  procedures: {len(data['procedures'])} with valid timestamps")
    else:
        print(f"  Warning: No date column found in procedures. Creating dummy timestamps.")
        data['procedures']['timestamp'] = pd.Timestamp.now()
    
    # ED visit timestamps (nyu_edu)
    ed_date_col = find_column(data['nyu_edu'], 
                              ['month_date', 'hosp_adm_dt', 'admission_date', 'visit_date', 'date', 'timestamp'], 
                              'nyu_edu')
    if ed_date_col:
        print(f"  Using '{ed_date_col}' for ED visit timestamps")
        data['nyu_edu']['timestamp'] = data['nyu_edu'][ed_date_col].apply(parse_date)
        data['nyu_edu'] = data['nyu_edu'][data['nyu_edu']['timestamp'].notna()].copy()
        
        # Filter out future dates (data quality issue)
        now = pd.Timestamp.now()
        future_dates = data['nyu_edu']['timestamp'] > now
        if future_dates.any():
            print(f"  ⚠ Warning: Found {future_dates.sum()} ED visits with future dates. Removing them.")
            print(f"    Future date range: {data['nyu_edu'][future_dates]['timestamp'].min()} to {data['nyu_edu'][future_dates]['timestamp'].max()}")
            data['nyu_edu'] = data['nyu_edu'][~future_dates].copy()
        
        print(f"  nyu_edu: {len(data['nyu_edu'])} with valid timestamps")
    else:
        print(f"  Warning: No date column found in nyu_edu. Creating dummy timestamps.")
        data['nyu_edu']['timestamp'] = pd.Timestamp.now()
    
    # Demographics - use DOB for age calculation
    dob_col = find_column(data['demographics'], 
                         ['mbr_dob', 'dob', 'date_of_birth', 'birth_date'], 
                         'demographics')
    if dob_col:
        print(f"  Using '{dob_col}' for date of birth")
        data['demographics']['dob'] = data['demographics'][dob_col].apply(parse_date)
    else:
        print(f"  Warning: No DOB column found in demographics. Ages will be approximate.")
        # Create a default DOB (e.g., 50 years ago)
        data['demographics']['dob'] = pd.Timestamp.now() - pd.DateOffset(years=50)
    
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
    
    # Diagnostic: Show date ranges for each event type
    print("\n  Date ranges by event type:")
    for source in ['diagnosis', 'procedure', 'ed_visit']:
        source_events = events_df[events_df['source'] == source]
        if len(source_events) > 0:
            min_date = source_events['timestamp'].min()
            max_date = source_events['timestamp'].max()
            print(f"    {source}: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')} ({len(source_events)} events)")
        else:
            print(f"    {source}: No events found")
    
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
    
    # Warn about insufficient ED visits
    if len(train_data['nyu_edu']) == 0:
        print(f"\n  ⚠ WARNING: No ED visits in training data!")
        print(f"    This is likely because all ED visits occur after {config.T_CUT_TRAIN}")
        print(f"    The model may not learn effectively without training ED visits.")
        print(f"    Consider adjusting T_CUT_TRAIN in config.py to include more ED visits.")
    
    if len(val_data['nyu_edu']) == 0:
        print(f"\n  ⚠ WARNING: No ED visits in validation data!")
        print(f"    This will make it difficult to tune hyperparameters.")
        print(f"    Consider adjusting T_CUT_VAL in config.py.")
    
    if len(train_data['nyu_edu']) < 10:
        print(f"\n  ⚠ WARNING: Very few ED visits in training data ({len(train_data['nyu_edu'])})!")
        print(f"    Model training may be unstable with so few positive examples.")
    
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


