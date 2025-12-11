"""
Hyperparameter Configuration for ED Utilization Prediction
All hyperparameters in one place for easy tuning
"""

import torch

# ============================================================================
# PATHS
# ============================================================================
DATA_DIR = "Files V2/augmented"  # Change to "Files V2" if using non-augmented data
OUTPUT_DIR = "outputs"
MODEL_SAVE_DIR = "models"
RESULTS_DIR = "results"

# ============================================================================
# DATA PREPROCESSING STRATEGY
# ============================================================================
FORCE_REPROCESS = True           # Force reprocessing even if cached data exists
USE_ENHANCED_PREPROCESSING = True  # Use enhanced preprocessing with balanced splits
MIN_POSITIVES_PER_SPLIT = 300    # Minimum positive samples per split

# ============================================================================
# EVALUATION STRATEGY
# ============================================================================
USE_CROSS_VALIDATION = True   # Use k-fold CV instead of temporal split
                             # Set to False to use temporal split below
K_FOLDS = 5                  # Number of folds for cross-validation
CV_RANDOM_STATE = 42         # Random seed for CV fold creation

# ============================================================================
# TIME-BASED SPLITS (YYYY-MM-DD) - Only used if USE_CROSS_VALIDATION = False
# ============================================================================
T_CUT_TRAIN = "2020-12-31"  # Training data up to this date
T_CUT_VAL = "2021-12-31"    # Validation data up to this date
T_CUT_TEST = "2022-12-31"   # Test data up to this date

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
# LLM for code embeddings
LLM_MODEL = "microsoft/BioGPT-Large"  # or "microsoft/BioGPT" for smaller
LLM_EMBEDDING_DIM = 1024  # BioGPT-Large hidden size
PROJECTED_DIM = 128       # Project down to this dimension
USE_CACHED_EMBEDDINGS = True  # Cache LLM embeddings to disk

# Numerical feature normalization
NORMALIZE_NUMERICAL = True
NUMERICAL_FEATURES = ['age', 'cost', 'charlson_index', 'los']

# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================
NODE_TYPES = ['patient', 'visit', 'dx_code', 'proc_code', 'provider', 'hospital', 'sdoh']

EDGE_TYPES = [
    ('patient', 'has_visit', 'visit'),
    ('visit', 'has_diagnosis', 'dx_code'),
    ('visit', 'has_procedure', 'proc_code'),
    ('proc_code', 'performed_by', 'provider'),
    ('provider', 'works_at', 'hospital'),
    ('patient', 'has_sdoh', 'sdoh'),
    ('patient', 'ed_visit', 'visit'),  # Target label edges
]

# Cold-start handling
UNK_TOKEN = "<UNK>"
MIN_CODE_FREQUENCY = 2  # Codes appearing less than this become UNK

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================
HIDDEN_DIM = 128
NUM_LAYERS = 3
DROPOUT = 0.3  # Increased from 0.1 for better regularization

# TGN specific
TGN_MEMORY_DIM = 128
TGN_TIME_DIM = 16        # Fourier time encoding dimension
TGN_MESSAGE_DIM = 128
TGN_AGGREGATOR = "mean"  # Options: "mean", "last", "sum"

# TGAT specific
TGAT_NUM_HEADS = 8
TGAT_DROPOUT = 0.1

# HGT specific
HGT_NUM_HEADS = 4
HGT_USE_NORM = True

# LSTM/Time-series specific
LSTM_HIDDEN_DIM = 128
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.3
LSTM_BIDIRECTIONAL = True
MAX_SEQUENCE_LENGTH = 50  # Maximum number of events per patient to use

# Cox Proportional Hazards specific
COX_ALPHA = 0.1          # L2 regularization for Cox model
COX_MAX_ITER = 1000      # Maximum iterations

# DeepSurv specific
DEEPSURV_LAYERS = [128, 64, 32]  # Hidden layer sizes
DEEPSURV_DROPOUT = 0.3
DEEPSURV_BATCH_NORM = True

# ============================================================================
# TASK CONFIGURATION
# ============================================================================
TASK_TYPE = "classification"  # Options: "classification", "survival", "regression"
NUM_CLASSES = 3               # Low, Medium, High utilization

# ED Utilization Classification Thresholds
# Based on # of ED visits in lookback window
UTILIZATION_LOOKBACK_DAYS = 365  # Look at past 12 months

# Binning strategy
USE_PERCENTILE_BINNING = True  # If True, use percentile-based binning (33/67 percentiles for balanced classes)
                               # If False, use fixed thresholds below

# Fixed thresholds (only used if USE_PERCENTILE_BINNING = False)
UTILIZATION_THRESHOLDS = {
    'low': (0, 1),      # 0-1 visits = low utilizer
    'medium': (2, 4),   # 2-4 visits = medium utilizer
    'high': (5, 999)    # 5+ visits = high utilizer
}

# Class weights for imbalanced data (will be computed from data if None)
CLASS_WEIGHTS = None  # Or manually set e.g. [1.0, 2.0, 5.0] to upweight rare classes

# Legacy parameters (kept for backward compatibility, not used in classification)
USE_MULTI_TASK = False    # Not used in classification mode
BINARY_THRESHOLD_DAYS = 30
MAX_DAYS_NORMALIZATION = 365

# ============================================================================
# TRAINING
# ============================================================================
BATCH_SIZE = 32
NUM_EPOCHS = 150  # Increased from 100
LEARNING_RATE = 3e-4  # Decreased from 1e-3 for stability
WEIGHT_DECAY = 5e-3  # Increased from 1e-4 for regularization
GRAD_CLIP = 1.0

# Loss configuration for classification
USE_FOCAL_LOSS = True     # Use focal loss for class imbalance
FOCAL_ALPHA = None        # Per-class weights (computed from data if None)
FOCAL_GAMMA = 2.0         # Focusing parameter for focal loss
LABEL_SMOOTHING = 0.1     # Label smoothing for regularization

# Legacy loss weights (not used in classification)
LAMBDA_BCE = 1.0
LAMBDA_MAE = 0.0

# Optimizer
OPTIMIZER = "adamw"       # Options: "adam", "adamw", "sgd"

# LR Scheduler
USE_SCHEDULER = True
SCHEDULER_TYPE = "cosine" # Options: "cosine", "step", "plateau"
T_MAX = 20                # For cosine annealing
SCHEDULER_PATIENCE = 10   # For plateau scheduler

# Early stopping (adjusted for AUROC metric)
EARLY_STOPPING = True
PATIENCE = 20  # Increased from 10 - more patience for AUROC improvements
MIN_DELTA = 0.01  # Increased from 1e-4 - require meaningful improvements

# ============================================================================
# SAMPLING (for mini-batch training)
# ============================================================================
# PyG neighbor sampling
FANOUT = [15, 10, 10]     # Neighbor samples per layer
NUM_WORKERS = 4

# Negative sampling for link prediction
NEG_SAMPLE_RATIO = 0.0    # 0 = no negative sampling (we're doing regression)

# ============================================================================
# EVALUATION
# ============================================================================
EVAL_BATCH_SIZE = 64

# Metrics to track for classification
TRACK_METRICS = [
    'accuracy',           # Overall accuracy
    'balanced_accuracy',  # Balanced accuracy (accounts for class imbalance)
    'f1_macro',          # Macro-averaged F1 (equal weight per class)
    'f1_weighted',       # Weighted F1 (weighted by support)
    'precision_macro',   # Macro precision
    'recall_macro',      # Macro recall
    'auroc_ovr',         # One-vs-Rest AUROC
    'auroc_ovo',         # One-vs-One AUROC
]

PRIMARY_METRIC = 'f1_macro'  # For early stopping and model selection

# Bootstrap confidence intervals
BOOTSTRAP_SAMPLES = 1000
CONFIDENCE_LEVEL = 0.95

# ============================================================================
# COMPUTATIONAL
# ============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
NUM_THREADS = 4

# Memory optimization
USE_GRADIENT_CHECKPOINTING = True
USE_MIXED_PRECISION = False  # AMP for faster training (if GPU available)

# ============================================================================
# LOGGING & CHECKPOINTING
# ============================================================================
LOG_INTERVAL = 10         # Log every N batches
SAVE_INTERVAL = 5         # Save checkpoint every N epochs
VERBOSE = True

# Weights & Biases (optional)
USE_WANDB = False
WANDB_PROJECT = "ed-utilization"
WANDB_ENTITY = None       # Your W&B username

# ============================================================================
# ABLATION STUDIES
# ============================================================================
RUN_ABLATIONS = False
ABLATIONS = [
    'no_llm_embeddings',   # Use random embeddings instead of LLM
    'shuffle_timestamps',  # Shuffle to verify temporal signal
    'no_sdoh',            # Remove SDOH features
]

# ============================================================================
# REPRODUCIBILITY
# ============================================================================
DETERMINISTIC = True      # Makes training deterministic (slower)

# ============================================================================
# MODEL SELECTION
# ============================================================================
MODELS_TO_TRAIN = ['TGN', 'TGAT', 'LSTM']  # Which models to train
# Options for classification: 
#   Graph models: 'TGN', 'TGAT', 'HGT'
#   Time-series: 'LSTM'
#   Classical: 'LogisticRegression', 'RandomForest', 'XGBoost'
# Note: CoxPH and DeepSurv are survival models, not used for classification

# ============================================================================
# SANITY CHECKS
# ============================================================================
RUN_SANITY_CHECKS = True
MAX_SANITY_SAMPLES = 1000  # For quick sanity checks

print(f"Configuration loaded. Device: {DEVICE}")

