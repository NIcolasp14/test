"""
Data Loading and Preprocessing Module
Loads CSV files and creates patient-level labels for ED utilization classification
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

import config_hgt as config


class DataLoader:
    """Load and preprocess healthcare data for HGT pipeline"""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.demographics = None
        self.diagnosis = None
        self.procedures = None
        self.sdoh = None
        self.nyu_edu = None
        self.patient_labels = None
        
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """Load all CSV files"""
        if self.verbose:
            print("=" * 80)
            print("Loading data files...")
            print("=" * 80)
        
        # Load demographics
        try:
            self.demographics = pd.read_csv(config.DEMOGRAPHICS_FILE)
            if self.verbose:
                print(f"✓ Demographics: {len(self.demographics)} rows")
        except Exception as e:
            print(f"✗ Error loading demographics: {e}")
            self.demographics = pd.DataFrame()
        
        # Load diagnosis
        try:
            self.diagnosis = pd.read_csv(config.DIAGNOSIS_FILE)
            if self.verbose:
                print(f"✓ Diagnosis: {len(self.diagnosis)} rows")
        except Exception as e:
            print(f"✗ Error loading diagnosis: {e}")
            self.diagnosis = pd.DataFrame()
        
        # Load procedures
        try:
            self.procedures = pd.read_csv(config.PROCEDURES_FILE)
            if self.verbose:
                print(f"✓ Procedures: {len(self.procedures)} rows")
        except Exception as e:
            print(f"✗ Error loading procedures: {e}")
            self.procedures = pd.DataFrame()
        
        # Load SDOH
        try:
            self.sdoh = pd.read_csv(config.SDOH_FILE, sep=';')
            if self.verbose:
                print(f"✓ SDOH: {len(self.sdoh)} rows")
        except Exception as e:
            print(f"✗ Error loading SDOH: {e}")
            self.sdoh = pd.DataFrame()
        
        # Load NYU EDU (target labels)
        try:
            self.nyu_edu = pd.read_csv(config.NYU_EDU_FILE)
            if self.verbose:
                print(f"✓ NYU EDU: {len(self.nyu_edu)} rows")
        except Exception as e:
            print(f"✗ Error loading NYU EDU: {e}")
            self.nyu_edu = pd.DataFrame()
        
        return {
            'demographics': self.demographics,
            'diagnosis': self.diagnosis,
            'procedures': self.procedures,
            'sdoh': self.sdoh,
            'nyu_edu': self.nyu_edu
        }
    
    def create_patient_labels(self) -> pd.DataFrame:
        """
        Create patient-level labels for ED utilization classification
        
        Returns:
            DataFrame with columns: [patient_id, ed_count, ed_utilization_class]
            Classes: 0=Low, 1=Medium, 2=High
        """
        if self.verbose:
            print("\n" + "=" * 80)
            print("Creating patient-level ED utilization labels...")
            print("=" * 80)
        
        # Check if nyu_edu exists and has required columns
        if self.nyu_edu is None or len(self.nyu_edu) == 0:
            raise ValueError("NYU EDU data not loaded")
        
        # Aggregate ED counts per patient
        # Use 'ed_count' column which represents total ED visits
        patient_ed_counts = self.nyu_edu.groupby('sys_mbr_sk').agg({
            'ed_count': 'sum',
            'outpat_ed_count': 'sum',
            'ed_to_obs_count': 'sum'
        }).reset_index()
        
        # Total ED utilization = all ED-related counts
        patient_ed_counts['total_ed_visits'] = (
            patient_ed_counts['ed_count'].fillna(0) +
            patient_ed_counts['outpat_ed_count'].fillna(0) +
            patient_ed_counts['ed_to_obs_count'].fillna(0)
        )
        
        # Create classification labels
        if config.UTILIZATION_STRATEGY == "percentile":
            # Percentile-based binning (balanced classes)
            low_threshold = np.percentile(
                patient_ed_counts['total_ed_visits'], 
                config.LOW_PERCENTILE
            )
            high_threshold = np.percentile(
                patient_ed_counts['total_ed_visits'], 
                config.HIGH_PERCENTILE
            )
            
            patient_ed_counts['ed_utilization_class'] = pd.cut(
                patient_ed_counts['total_ed_visits'],
                bins=[-np.inf, low_threshold, high_threshold, np.inf],
                labels=[0, 1, 2]  # Low, Medium, High
            ).astype(int)
            
            if self.verbose:
                print(f"Percentile-based binning:")
                print(f"  Low: 0 to {low_threshold:.1f} visits")
                print(f"  Medium: {low_threshold:.1f} to {high_threshold:.1f} visits")
                print(f"  High: {high_threshold:.1f}+ visits")
        
        else:  # Fixed thresholds
            def classify_utilization(count):
                if count <= config.FIXED_THRESHOLDS['low'][1]:
                    return 0  # Low
                elif count <= config.FIXED_THRESHOLDS['medium'][1]:
                    return 1  # Medium
                else:
                    return 2  # High
            
            patient_ed_counts['ed_utilization_class'] = patient_ed_counts[
                'total_ed_visits'
            ].apply(classify_utilization)
            
            if self.verbose:
                print(f"Fixed threshold binning:")
                for cls_name, (low, high) in config.FIXED_THRESHOLDS.items():
                    print(f"  {cls_name.capitalize()}: {low}-{high} visits")
        
        # Class distribution
        class_dist = patient_ed_counts['ed_utilization_class'].value_counts().sort_index()
        if self.verbose:
            print(f"\nClass distribution:")
            class_names = ['Low', 'Medium', 'High']
            for cls_idx, count in class_dist.items():
                pct = 100 * count / len(patient_ed_counts)
                print(f"  Class {cls_idx} ({class_names[cls_idx]}): {count} ({pct:.1f}%)")
        
        # Rename column for clarity
        patient_labels = patient_ed_counts[[
            'sys_mbr_sk', 'total_ed_visits', 'ed_utilization_class'
        ]].rename(columns={'sys_mbr_sk': 'patient_id'})
        
        self.patient_labels = patient_labels
        
        return patient_labels
    
    def prepare_patient_features(self) -> pd.DataFrame:
        """
        Prepare patient-level features
        
        Returns:
            DataFrame with patient features
        """
        if self.verbose:
            print("\n" + "=" * 80)
            print("Preparing patient features...")
            print("=" * 80)
        
        # Start with demographics
        patient_features = self.demographics.copy()
        
        # Process gender
        gender_map = {'M': 1, 'F': 0, 'Male': 1, 'Female': 0}
        patient_features['gender_encoded'] = patient_features['mbr_gender_cd'].map(
            gender_map
        ).fillna(0.5)  # Unknown = 0.5
        
        # Calculate age from DOB
        def calculate_age(dob_str):
            try:
                # Try multiple date formats
                for fmt in ['%m/%d/%y', '%m/%d/%Y', '%Y-%m-%d', '%d/%m/%Y']:
                    try:
                        dob = datetime.strptime(str(dob_str), fmt)
                        age = (datetime.now() - dob).days / 365.25
                        return max(0, min(120, age))  # Clamp to reasonable range
                    except:
                        continue
                return np.nan
            except:
                return np.nan
        
        patient_features['age'] = patient_features['mbr_dob'].apply(calculate_age)
        patient_features['age'] = patient_features['age'].fillna(
            patient_features['age'].median()
        )
        
        # Merge with SDOH if available
        if config.USE_SDOH and self.sdoh is not None and len(self.sdoh) > 0:
            # Use lumeris_empi as the key
            if 'empi' in patient_features.columns and 'lumeris_empi' in self.sdoh.columns:
                # Select SDOH features
                sdoh_cols = ['lumeris_empi'] + [
                    col for col in config.SDOH_FEATURES 
                    if col in self.sdoh.columns
                ]
                sdoh_subset = self.sdoh[sdoh_cols].copy()
                
                # Convert to numeric and fill NaNs
                for col in sdoh_subset.columns:
                    if col != 'lumeris_empi':
                        sdoh_subset[col] = pd.to_numeric(
                            sdoh_subset[col], errors='coerce'
                        ).fillna(0)
                
                # Merge
                patient_features = patient_features.merge(
                    sdoh_subset,
                    left_on='empi',
                    right_on='lumeris_empi',
                    how='left'
                )
                
                if self.verbose:
                    print(f"  Merged {len(sdoh_cols)-1} SDOH features")
        
        # Select final feature columns
        feature_cols = ['sys_mbr_sk', 'age', 'gender_encoded']
        
        # Add SDOH features if they exist
        for col in config.SDOH_FEATURES:
            if col in patient_features.columns:
                feature_cols.append(col)
                patient_features[col] = patient_features[col].fillna(0)
        
        patient_features = patient_features[feature_cols].rename(
            columns={'sys_mbr_sk': 'patient_id'}
        )
        
        # Fill any remaining NaNs
        patient_features = patient_features.fillna(0)
        
        if self.verbose:
            print(f"  Final patient features: {patient_features.shape[1]-1} features")
            print(f"  Features: {feature_cols[1:]}")
        
        return patient_features
    
    def get_diagnosis_data(self) -> pd.DataFrame:
        """Get diagnosis data with patient linkage"""
        if self.diagnosis is None or len(self.diagnosis) == 0:
            return pd.DataFrame(columns=['patient_id', 'diagnosis_code'])
        
        dx_data = self.diagnosis[['clm_sys_mbr_sk', 'icd9_diagnosis_cd']].copy()
        dx_data.columns = ['patient_id', 'diagnosis_code']
        dx_data = dx_data.dropna()
        
        return dx_data
    
    def get_procedure_data(self) -> pd.DataFrame:
        """Get procedure data with patient and provider linkage"""
        if self.procedures is None or len(self.procedures) == 0:
            return pd.DataFrame(columns=[
                'patient_id', 'procedure_code', 'provider_npi', 'hospital_id'
            ])
        
        proc_data = self.procedures[[
            'sys_mbr_sk', 'cpt_cd', 'prov_npi_full_nm', 'hosp_adm_id'
        ]].copy()
        proc_data.columns = [
            'patient_id', 'procedure_code', 'provider_npi', 'hospital_id'
        ]
        proc_data = proc_data.dropna(subset=['patient_id', 'procedure_code'])
        
        return proc_data
    
    def get_summary(self) -> Dict:
        """Get summary statistics of loaded data"""
        summary = {
            'n_patients': len(self.demographics) if self.demographics is not None else 0,
            'n_diagnosis_records': len(self.diagnosis) if self.diagnosis is not None else 0,
            'n_procedure_records': len(self.procedures) if self.procedures is not None else 0,
            'n_sdoh_records': len(self.sdoh) if self.sdoh is not None else 0,
            'n_ed_records': len(self.nyu_edu) if self.nyu_edu is not None else 0,
        }
        
        if self.patient_labels is not None:
            summary['n_labeled_patients'] = len(self.patient_labels)
            summary['class_distribution'] = self.patient_labels[
                'ed_utilization_class'
            ].value_counts().to_dict()
        
        return summary


def test_data_loader():
    """Test the data loader"""
    print("Testing DataLoader...")
    
    loader = DataLoader(verbose=True)
    
    # Load all data
    data_dict = loader.load_all_data()
    
    # Create patient labels
    patient_labels = loader.create_patient_labels()
    
    # Prepare patient features
    patient_features = loader.prepare_patient_features()
    
    # Get diagnosis and procedure data
    dx_data = loader.get_diagnosis_data()
    proc_data = loader.get_procedure_data()
    
    # Print summary
    print("\n" + "=" * 80)
    print("Data Summary")
    print("=" * 80)
    summary = loader.get_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    print("\nDataLoader test completed successfully!")
    
    return loader


if __name__ == "__main__":
    test_data_loader()
