"""
Configuration for Heterogeneous Graph Transformer (HGT) Pipeline
ED Utilization Classification: Low, Medium, High
"""

import torch
import os

# ============================================================================
# PATHS
# ============================================================================
# Data paths
DATA_DIR = "../Files V2/augmented"
DEMOGRAPHICS_FILE = os.path.join(DATA_DIR, "demographics.csv")
DIAGNOSIS_FILE = os.path.join(DATA_DIR, "diagnosis.csv")
PROCEDURES_FILE = os.path.join(DATA_DIR, "procedures.csv")
SDOH_FILE = os.path.join(DATA_DIR, "sdoh.csv")
NYU_EDU_FILE = os.path.join(DATA_DIR, "nyu_edu.csv")

# Output paths
OUTPUT_DIR = "outputs"
MODEL_SAVE_DIR = "saved_models"
RESULTS_DIR = "results"
CACHE_DIR = "cache"

# ============================================================================
# TASK CONFIGURATION
# ============================================================================
TASK_TYPE = "classification"  # ED utilization classification
NUM_CLASSES = 3  # Low, Medium, High utilization

# ED Utilization binning strategy
# We'll aggregate ED counts per patient and bin into 3 classes
UTILIZATION_STRATEGY = "percentile"  # Options: "percentile", "fixed"

# Percentile-based binning (recommended for balanced classes)
LOW_PERCENTILE = 33  # 0-33rd percentile = Low
HIGH_PERCENTILE = 67  # 67-100th percentile = High
# 33-67th percentile = Medium

# Fixed thresholds (only used if UTILIZATION_STRATEGY = "fixed")
FIXED_THRESHOLDS = {
    'low': (0, 2),      # 0-2 ED visits
    'medium': (3, 5),   # 3-5 ED visits
    'high': (6, 999)    # 6+ ED visits
}

# ============================================================================
# HETEROGENEOUS GRAPH CONFIGURATION
# ============================================================================
# Node types in the heterogeneous graph
NODE_TYPES = [
    'patient',      # Patient nodes
    'diagnosis',    # Diagnosis code nodes (ICD9)
    'procedure',    # Procedure code nodes (CPT)
    'provider',     # Provider nodes (NPI)
    'hospital'      # Hospital nodes
]

# Edge types (source, relation, destination)
EDGE_TYPES = [
    ('patient', 'has_diagnosis', 'diagnosis'),
    ('diagnosis', 'diagnosed_in', 'patient'),  # Reverse edge
    ('patient', 'has_procedure', 'procedure'),
    ('procedure', 'performed_on', 'patient'),  # Reverse edge
    ('procedure', 'performed_by', 'provider'),
    ('provider', 'performs', 'procedure'),  # Reverse edge
    ('provider', 'works_at', 'hospital'),
    ('hospital', 'employs', 'provider'),  # Reverse edge
    ('patient', 'visits', 'hospital'),
    ('hospital', 'visited_by', 'patient')  # Reverse edge
]

# Metadata types for PyG HeteroData
METADATA = (NODE_TYPES, EDGE_TYPES)

# Cold-start handling for rare codes
MIN_CODE_FREQUENCY = 2  # Codes appearing < 2 times become UNK
UNK_TOKEN = "<UNK>"

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
# Patient features
PATIENT_NUMERICAL_FEATURES = ['age', 'gender_encoded']
PATIENT_FEATURE_DIM = 16  # After processing

# Diagnosis/Procedure features
USE_CODE_EMBEDDINGS = True  # Use learnable embeddings for codes
CODE_EMBEDDING_DIM = 32

# Provider/Hospital features
PROVIDER_FEATURE_DIM = 8
HOSPITAL_FEATURE_DIM = 8

# SDOH features
USE_SDOH = True
SDOH_FEATURES = [
    'sdh_agg_transportation', 'sdh_agg_communication_sup', 
    'sdh_agg_affordability', 'sdh_agg_employment_sup',
    'sdh_agg_insurance_cov', 'sdh_agg_caregiver_sup',
    'sdh_agg_geography', 'sdh_agg_housing_sup', 'sdh_agg_health_beh'
]

# Normalization
NORMALIZE_FEATURES = True

# ============================================================================
# MODEL ARCHITECTURE - Heterogeneous Graph Transformer
# ============================================================================
# HGT Configuration
HGT_HIDDEN_DIM = 128
HGT_NUM_HEADS = 4
HGT_NUM_LAYERS = 3
HGT_DROPOUT = 0.3
HGT_USE_NORM = True

# Prediction head
PREDICTION_HEAD_HIDDEN_DIM = 64
PREDICTION_HEAD_DROPOUT = 0.3

# ============================================================================
# CROSS-VALIDATION CONFIGURATION
# ============================================================================
USE_CROSS_VALIDATION = True
N_FOLDS = 5
STRATIFIED = True  # Stratify by target class
CV_RANDOM_STATE = 42

# Train/Val/Test split (if not using CV)
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Data leakage prevention
# - Split at patient level (not visit level)
# - Ensure no patient appears in multiple splits
PATIENT_LEVEL_SPLIT = True

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
# Optimization
BATCH_SIZE = 32  # Number of patients per batch
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
OPTIMIZER = "adam"  # Options: "adam", "adamw"

# Gradient clipping
GRAD_CLIP = 1.0

# Loss function
USE_CLASS_WEIGHTS = True  # Weight classes inversely to their frequency
LABEL_SMOOTHING = 0.1  # Smooth labels for regularization

# Learning rate scheduler
USE_LR_SCHEDULER = True
SCHEDULER_TYPE = "cosine"  # Options: "cosine", "step", "plateau"
SCHEDULER_PATIENCE = 10  # For plateau
SCHEDULER_FACTOR = 0.5  # For step/plateau
SCHEDULER_STEP_SIZE = 20  # For step
T_MAX = 50  # For cosine

# Early stopping
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 15
EARLY_STOPPING_MIN_DELTA = 0.001
EARLY_STOPPING_METRIC = "val_f1_macro"  # Metric to monitor

# ============================================================================
# NEIGHBOR SAMPLING (for mini-batch training)
# ============================================================================
USE_NEIGHBOR_SAMPLING = True  # Use neighbor sampling for large graphs
NUM_NEIGHBORS = [10, 10, 10]  # Number of neighbors per layer per edge type
NUM_WORKERS = 4

# ============================================================================
# EVALUATION METRICS
# ============================================================================
METRICS = [
    'accuracy',
    'balanced_accuracy',
    'f1_macro',
    'f1_weighted',
    'precision_macro',
    'recall_macro',
    'auroc_ovr',  # One-vs-Rest AUROC
    'confusion_matrix'
]

PRIMARY_METRIC = 'f1_macro'  # For model selection

# ============================================================================
# COMPUTATIONAL SETTINGS
# ============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
DETERMINISTIC = True  # For reproducibility

# Memory optimization
USE_MIXED_PRECISION = False  # FP16 training
GRADIENT_ACCUMULATION_STEPS = 1

# ============================================================================
# LOGGING
# ============================================================================
VERBOSE = True
LOG_INTERVAL = 10  # Log every N batches
SAVE_BEST_MODEL = True
SAVE_PREDICTIONS = True

# Wandb (optional)
USE_WANDB = False
WANDB_PROJECT = "hgt-ed-utilization"
WANDB_ENTITY = None

# ============================================================================
# REPRODUCIBILITY
# ============================================================================
def set_seed(seed=SEED):
    """Set random seeds for reproducibility"""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if DETERMINISTIC:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Create directories
for dir_path in [OUTPUT_DIR, MODEL_SAVE_DIR, RESULTS_DIR, CACHE_DIR]:
    os.makedirs(dir_path, exist_ok=True)

print(f"Configuration loaded. Device: {DEVICE}")
print(f"Task: {NUM_CLASSES}-class ED Utilization Classification")
print(f"Cross-validation: {N_FOLDS}-fold" if USE_CROSS_VALIDATION else "Train/Val/Test split")
