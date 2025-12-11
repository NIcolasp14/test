"""
Evaluation Module
Comprehensive evaluation metrics and visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix,
    classification_report, roc_curve, auc
)
from typing import Dict, List, Tuple, Optional
import os

import config_hgt as config


class Evaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self, class_names: List[str] = None, verbose: bool = True):
        self.class_names = class_names or ['Low', 'Medium', 'High']
        self.verbose = verbose
        
    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_probs: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Compute comprehensive classification metrics
        
        Args:
            y_true: True labels (N,)
            y_pred: Predicted labels (N,)
            y_probs: Predicted probabilities (N, num_classes)
        
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        
        # F1, Precision, Recall (macro and weighted)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class F1
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        for i, class_name in enumerate(self.class_names):
            metrics[f'f1_{class_name.lower()}'] = f1_per_class[i]
        
        # AUROC (if probabilities provided)
        if y_probs is not None:
            try:
                metrics['auroc_ovr'] = roc_auc_score(
                    y_true, y_probs, multi_class='ovr', average='macro'
                )
                metrics['auroc_ovo'] = roc_auc_score(
                    y_true, y_probs, multi_class='ovo', average='macro'
                )
            except:
                metrics['auroc_ovr'] = 0.0
                metrics['auroc_ovo'] = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        
        return metrics
    
    def print_metrics(self, metrics: Dict, title: str = "Evaluation Metrics"):
        """Print metrics in a formatted way"""
        if not self.verbose:
            return
        
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)
        
        # Overall metrics
        print("\nOverall Metrics:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"  F1 (Macro): {metrics['f1_macro']:.4f}")
        print(f"  F1 (Weighted): {metrics['f1_weighted']:.4f}")
        print(f"  Precision (Macro): {metrics['precision_macro']:.4f}")
        print(f"  Recall (Macro): {metrics['recall_macro']:.4f}")
        
        if 'auroc_ovr' in metrics:
            print(f"  AUROC (OvR): {metrics['auroc_ovr']:.4f}")
            print(f"  AUROC (OvO): {metrics['auroc_ovo']:.4f}")
        
        # Per-class metrics
        print("\nPer-Class F1 Scores:")
        for class_name in self.class_names:
            key = f'f1_{class_name.lower()}'
            if key in metrics:
                print(f"  {class_name}: {metrics[key]:.4f}")
        
        # Confusion matrix
        if 'confusion_matrix' in metrics:
            print("\nConfusion Matrix:")
            cm = metrics['confusion_matrix']
            
            # Print header
            header = "         " + "  ".join([f"{name:>8}" for name in self.class_names])
            print(header)
            
            # Print rows
            for i, row in enumerate(cm):
                row_str = f"{self.class_names[i]:>8}" + "  ".join([f"{val:>8}" for val in row])
                print(row_str)
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[str] = None,
        normalize: bool = False
    ):
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save figure
            normalize: Whether to normalize the confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
        )
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_roc_curves(
        self,
        y_true: np.ndarray,
        y_probs: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Plot ROC curves for each class
        
        Args:
            y_true: True labels
            y_probs: Predicted probabilities (N, num_classes)
            save_path: Path to save figure
        """
        from sklearn.preprocessing import label_binarize
        
        # Binarize labels
        y_true_bin = label_binarize(y_true, classes=range(len(self.class_names)))
        
        plt.figure(figsize=(10, 8))
        
        # Plot ROC curve for each class
        for i, class_name in enumerate(self.class_names):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(
                fpr, tpr,
                label=f'{class_name} (AUC = {roc_auc:.2f})',
                linewidth=2
            )
        
        # Plot diagonal
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves (One-vs-Rest)')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"ROC curves saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_training_history(
        self,
        history: Dict,
        save_path: Optional[str] = None
    ):
        """
        Plot training history (loss and metrics over epochs)
        
        Args:
            history: Training history dictionary
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot loss
        axes[0, 0].plot(history['train_loss'], label='Train', linewidth=2)
        axes[0, 0].plot(history['val_loss'], label='Val', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Plot F1 score
        train_f1 = [m['f1_macro'] for m in history['train_metrics']]
        val_f1 = [m['f1_macro'] for m in history['val_metrics']]
        axes[0, 1].plot(train_f1, label='Train', linewidth=2)
        axes[0, 1].plot(val_f1, label='Val', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('F1 Score (Macro)')
        axes[0, 1].set_title('F1 Score')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # Plot accuracy
        train_acc = [m['accuracy'] for m in history['train_metrics']]
        val_acc = [m['accuracy'] for m in history['val_metrics']]
        axes[1, 0].plot(train_acc, label='Train', linewidth=2)
        axes[1, 0].plot(val_acc, label='Val', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # Plot balanced accuracy
        train_bal_acc = [m['balanced_accuracy'] for m in history['train_metrics']]
        val_bal_acc = [m['balanced_accuracy'] for m in history['val_metrics']]
        axes[1, 1].plot(train_bal_acc, label='Train', linewidth=2)
        axes[1, 1].plot(val_bal_acc, label='Val', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Balanced Accuracy')
        axes[1, 1].set_title('Balanced Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"Training history plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[str] = None
    ) -> str:
        """
        Generate detailed classification report
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save report
        
        Returns:
            Classification report string
        """
        report = classification_report(
            y_true,
            y_pred,
            target_names=self.class_names,
            digits=4
        )
        
        if self.verbose:
            print("\n" + "=" * 80)
            print("Classification Report")
            print("=" * 80)
            print(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write("Classification Report\n")
                f.write("=" * 80 + "\n")
                f.write(report)
            
            if self.verbose:
                print(f"Classification report saved to {save_path}")
        
        return report
    
    def save_predictions(
        self,
        patient_ids: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_probs: np.ndarray,
        save_path: str
    ):
        """
        Save predictions to CSV
        
        Args:
            patient_ids: Patient IDs
            y_true: True labels
            y_pred: Predicted labels
            y_probs: Predicted probabilities
            save_path: Path to save CSV
        """
        # Create DataFrame
        df = pd.DataFrame({
            'patient_id': patient_ids,
            'true_label': y_true,
            'predicted_label': y_pred,
            'true_class': [self.class_names[i] for i in y_true],
            'predicted_class': [self.class_names[i] for i in y_pred]
        })
        
        # Add probability columns
        for i, class_name in enumerate(self.class_names):
            df[f'prob_{class_name.lower()}'] = y_probs[:, i]
        
        # Add correctness flag
        df['correct'] = y_true == y_pred
        
        # Save to CSV
        df.to_csv(save_path, index=False)
        
        if self.verbose:
            print(f"Predictions saved to {save_path}")


def aggregate_cv_results(
    fold_results: List[Dict],
    verbose: bool = True
) -> Dict:
    """
    Aggregate results across CV folds
    
    Args:
        fold_results: List of result dictionaries from each fold
        verbose: Print aggregated results
    
    Returns:
        Dictionary of aggregated metrics
    """
    # Collect metrics across folds
    metric_names = [k for k in fold_results[0].keys() if k != 'confusion_matrix']
    
    aggregated = {}
    for metric_name in metric_names:
        values = [fold[metric_name] for fold in fold_results]
        aggregated[f'{metric_name}_mean'] = np.mean(values)
        aggregated[f'{metric_name}_std'] = np.std(values)
        aggregated[f'{metric_name}_min'] = np.min(values)
        aggregated[f'{metric_name}_max'] = np.max(values)
    
    if verbose:
        print("\n" + "=" * 80)
        print("Aggregated Cross-Validation Results")
        print("=" * 80)
        
        for metric_name in ['accuracy', 'balanced_accuracy', 'f1_macro', 'f1_weighted']:
            if f'{metric_name}_mean' in aggregated:
                mean = aggregated[f'{metric_name}_mean']
                std = aggregated[f'{metric_name}_std']
                print(f"{metric_name}: {mean:.4f} ± {std:.4f}")
    
    return aggregated


def test_evaluator():
    """Test evaluator module"""
    print("Testing Evaluator...")
    
    # Create dummy predictions
    np.random.seed(42)
    n_samples = 100
    y_true = np.random.choice([0, 1, 2], size=n_samples)
    y_pred = np.random.choice([0, 1, 2], size=n_samples)
    y_probs = np.random.dirichlet(np.ones(3), size=n_samples)
    
    # Create evaluator
    evaluator = Evaluator(class_names=['Low', 'Medium', 'High'], verbose=True)
    
    # Compute metrics
    metrics = evaluator.compute_metrics(y_true, y_pred, y_probs)
    evaluator.print_metrics(metrics)
    
    # Generate report
    report = evaluator.generate_classification_report(y_true, y_pred)
    
    print("\nEvaluator test completed successfully!")
    
    return evaluator


if __name__ == "__main__":
    test_evaluator()
