"""
Cross-Validation Module with Data Leakage Prevention

Key principles to prevent data leakage:
1. Split at the patient level (not visit/event level)
2. Ensure no patient appears in multiple splits
3. Stratify by target class for balanced folds
4. Build separate graphs for each fold
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from typing import List, Tuple, Dict
import torch

import config_hgt as config


class CrossValidator:
    """
    Cross-validation with proper data leakage prevention
    """
    
    def __init__(
        self,
        n_folds: int = config.N_FOLDS,
        stratified: bool = config.STRATIFIED,
        random_state: int = config.CV_RANDOM_STATE,
        verbose: bool = True
    ):
        self.n_folds = n_folds
        self.stratified = stratified
        self.random_state = random_state
        self.verbose = verbose
        
        self.fold_splits = []
    
    def create_folds(
        self,
        patient_labels: pd.DataFrame
    ) -> List[Dict[str, np.ndarray]]:
        """
        Create k-fold cross-validation splits at patient level
        
        Args:
            patient_labels: DataFrame with columns [patient_id, ed_utilization_class]
        
        Returns:
            List of dictionaries with train/val/test patient IDs for each fold
        """
        if self.verbose:
            print("\n" + "=" * 80)
            print(f"Creating {self.n_folds}-fold cross-validation splits...")
            print("=" * 80)
        
        # Extract patient IDs and labels
        patient_ids = patient_labels['patient_id'].values
        labels = patient_labels['ed_utilization_class'].values
        
        # Remove any patients with missing labels
        valid_mask = labels >= 0
        patient_ids = patient_ids[valid_mask]
        labels = labels[valid_mask]
        
        if self.verbose:
            print(f"Total patients with valid labels: {len(patient_ids)}")
            print(f"Class distribution:")
            for cls in range(config.NUM_CLASSES):
                count = (labels == cls).sum()
                pct = 100 * count / len(labels)
                print(f"  Class {cls}: {count} ({pct:.1f}%)")
        
        # Create stratified k-fold splits
        if self.stratified:
            skf = StratifiedKFold(
                n_splits=self.n_folds,
                shuffle=True,
                random_state=self.random_state
            )
            splits = list(skf.split(patient_ids, labels))
        else:
            from sklearn.model_selection import KFold
            kf = KFold(
                n_splits=self.n_folds,
                shuffle=True,
                random_state=self.random_state
            )
            splits = list(kf.split(patient_ids))
        
        # Create fold splits
        self.fold_splits = []
        for fold_idx, (train_val_idx, test_idx) in enumerate(splits):
            # Get train+val and test patient IDs
            train_val_patients = patient_ids[train_val_idx]
            test_patients = patient_ids[test_idx]
            
            # Further split train+val into train and val (80/20)
            train_val_labels = labels[train_val_idx]
            
            if self.stratified:
                train_idx, val_idx = train_test_split(
                    np.arange(len(train_val_patients)),
                    test_size=0.2,
                    stratify=train_val_labels,
                    random_state=self.random_state + fold_idx
                )
            else:
                train_idx, val_idx = train_test_split(
                    np.arange(len(train_val_patients)),
                    test_size=0.2,
                    random_state=self.random_state + fold_idx
                )
            
            train_patients = train_val_patients[train_idx]
            val_patients = train_val_patients[val_idx]
            
            # Verify no overlap (critical for preventing data leakage)
            assert len(set(train_patients) & set(val_patients)) == 0, \
                f"Fold {fold_idx}: Train and val sets overlap!"
            assert len(set(train_patients) & set(test_patients)) == 0, \
                f"Fold {fold_idx}: Train and test sets overlap!"
            assert len(set(val_patients) & set(test_patients)) == 0, \
                f"Fold {fold_idx}: Val and test sets overlap!"
            
            fold_split = {
                'fold': fold_idx,
                'train': train_patients,
                'val': val_patients,
                'test': test_patients
            }
            
            self.fold_splits.append(fold_split)
            
            if self.verbose:
                print(f"\nFold {fold_idx + 1}:")
                print(f"  Train: {len(train_patients)} patients")
                print(f"  Val: {len(val_patients)} patients")
                print(f"  Test: {len(test_patients)} patients")
                
                # Check class distribution in this fold
                train_labels = labels[train_val_idx][train_idx]
                val_labels = labels[train_val_idx][val_idx]
                test_labels = labels[test_idx]
                
                print(f"  Train class distribution: {np.bincount(train_labels)}")
                print(f"  Val class distribution: {np.bincount(val_labels)}")
                print(f"  Test class distribution: {np.bincount(test_labels)}")
        
        if self.verbose:
            print("\n✓ Cross-validation folds created successfully")
            print("  No patient appears in multiple splits (data leakage prevented)")
        
        return self.fold_splits
    
    def get_fold(self, fold_idx: int) -> Dict[str, np.ndarray]:
        """Get a specific fold"""
        if fold_idx >= len(self.fold_splits):
            raise ValueError(f"Fold {fold_idx} does not exist")
        return self.fold_splits[fold_idx]
    
    def create_masks(
        self,
        hetero_data,
        patient_id_map: Dict,
        train_patients: np.ndarray,
        val_patients: np.ndarray,
        test_patients: np.ndarray
    ):
        """
        Create train/val/test masks for HeteroData object
        
        Args:
            hetero_data: HeteroData object
            patient_id_map: Dictionary mapping patient_id to node index
            train_patients: Array of training patient IDs
            val_patients: Array of validation patient IDs
            test_patients: Array of test patient IDs
        
        Returns:
            hetero_data with updated masks
        """
        n_patients = hetero_data['patient'].x.shape[0]
        
        # Initialize all masks to False
        train_mask = torch.zeros(n_patients, dtype=torch.bool)
        val_mask = torch.zeros(n_patients, dtype=torch.bool)
        test_mask = torch.zeros(n_patients, dtype=torch.bool)
        
        # Set masks for each split
        for patient_id in train_patients:
            if patient_id in patient_id_map:
                patient_idx = patient_id_map[patient_id]
                train_mask[patient_idx] = True
        
        for patient_id in val_patients:
            if patient_id in patient_id_map:
                patient_idx = patient_id_map[patient_id]
                val_mask[patient_idx] = True
        
        for patient_id in test_patients:
            if patient_id in patient_id_map:
                patient_idx = patient_id_map[patient_id]
                test_mask[patient_idx] = True
        
        # Update hetero_data
        hetero_data['patient'].train_mask = train_mask
        hetero_data['patient'].val_mask = val_mask
        hetero_data['patient'].test_mask = test_mask
        
        # Verify no overlap
        assert (train_mask & val_mask).sum() == 0, "Train and val masks overlap!"
        assert (train_mask & test_mask).sum() == 0, "Train and test masks overlap!"
        assert (val_mask & test_mask).sum() == 0, "Val and test masks overlap!"
        
        return hetero_data
    
    def get_summary(self) -> Dict:
        """Get summary of cross-validation splits"""
        if len(self.fold_splits) == 0:
            return {}
        
        summary = {
            'n_folds': self.n_folds,
            'stratified': self.stratified,
            'folds': []
        }
        
        for fold in self.fold_splits:
            fold_summary = {
                'fold': fold['fold'],
                'n_train': len(fold['train']),
                'n_val': len(fold['val']),
                'n_test': len(fold['test'])
            }
            summary['folds'].append(fold_summary)
        
        return summary


def create_simple_split(
    patient_labels: pd.DataFrame,
    train_ratio: float = config.TRAIN_RATIO,
    val_ratio: float = config.VAL_RATIO,
    test_ratio: float = config.TEST_RATIO,
    stratified: bool = True,
    random_state: int = config.CV_RANDOM_STATE,
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """
    Create a simple train/val/test split (no cross-validation)
    
    Args:
        patient_labels: DataFrame with patient labels
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        stratified: Whether to stratify by class
        random_state: Random seed
        verbose: Print split statistics
    
    Returns:
        Dictionary with train/val/test patient IDs
    """
    if verbose:
        print("\n" + "=" * 80)
        print("Creating train/val/test split...")
        print("=" * 80)
    
    # Verify ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Train/val/test ratios must sum to 1.0"
    
    patient_ids = patient_labels['patient_id'].values
    labels = patient_labels['ed_utilization_class'].values
    
    # First split: separate test set
    if stratified:
        train_val_patients, test_patients = train_test_split(
            patient_ids,
            test_size=test_ratio,
            stratify=labels,
            random_state=random_state
        )
        
        # Get labels for train_val
        train_val_labels = labels[np.isin(patient_ids, train_val_patients)]
        
        # Second split: separate train and val
        val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
        train_patients, val_patients = train_test_split(
            train_val_patients,
            test_size=val_ratio_adjusted,
            stratify=train_val_labels,
            random_state=random_state
        )
    else:
        train_val_patients, test_patients = train_test_split(
            patient_ids,
            test_size=test_ratio,
            random_state=random_state
        )
        
        val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
        train_patients, val_patients = train_test_split(
            train_val_patients,
            test_size=val_ratio_adjusted,
            random_state=random_state
        )
    
    # Verify no overlap
    assert len(set(train_patients) & set(val_patients)) == 0, \
        "Train and val sets overlap!"
    assert len(set(train_patients) & set(test_patients)) == 0, \
        "Train and test sets overlap!"
    assert len(set(val_patients) & set(test_patients)) == 0, \
        "Val and test sets overlap!"
    
    split = {
        'train': train_patients,
        'val': val_patients,
        'test': test_patients
    }
    
    if verbose:
        print(f"Train: {len(train_patients)} patients ({100*train_ratio:.1f}%)")
        print(f"Val: {len(val_patients)} patients ({100*val_ratio:.1f}%)")
        print(f"Test: {len(test_patients)} patients ({100*test_ratio:.1f}%)")
        
        # Check class distribution
        train_labels = labels[np.isin(patient_ids, train_patients)]
        val_labels = labels[np.isin(patient_ids, val_patients)]
        test_labels = labels[np.isin(patient_ids, test_patients)]
        
        print(f"\nClass distribution:")
        print(f"  Train: {np.bincount(train_labels)}")
        print(f"  Val: {np.bincount(val_labels)}")
        print(f"  Test: {np.bincount(test_labels)}")
        
        print("\n✓ Split created successfully (no data leakage)")
    
    return split


def test_cross_validation():
    """Test cross-validation module"""
    print("Testing CrossValidator...")
    
    # Create dummy patient labels
    np.random.seed(42)
    n_patients = 1000
    patient_ids = np.arange(n_patients)
    labels = np.random.choice([0, 1, 2], size=n_patients, p=[0.3, 0.4, 0.3])
    
    patient_labels = pd.DataFrame({
        'patient_id': patient_ids,
        'ed_utilization_class': labels
    })
    
    # Test cross-validation
    cv = CrossValidator(n_folds=5, stratified=True, verbose=True)
    fold_splits = cv.create_folds(patient_labels)
    
    # Test simple split
    split = create_simple_split(
        patient_labels,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        stratified=True,
        verbose=True
    )
    
    print("\nCross-validation test completed successfully!")
    
    return cv


if __name__ == "__main__":
    test_cross_validation()
