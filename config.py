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
# TIME-BASED SPLITS (YYYY-MM-DD)
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
DROPOUT = 0.1

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

# Prediction head
USE_MULTI_TASK = True     # Both time-to-event and binary classification
BINARY_THRESHOLD_DAYS = 30  # Predict if ED visit within N days

# ============================================================================
# TRAINING
# ============================================================================
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

# Loss weights
LAMBDA_BCE = 0.5          # Weight for binary classification loss
LAMBDA_MAE = 1.0          # Weight for MAE loss

# Optimizer
OPTIMIZER = "adamw"       # Options: "adam", "adamw", "sgd"

# LR Scheduler
USE_SCHEDULER = True
SCHEDULER_TYPE = "cosine" # Options: "cosine", "step", "plateau"
T_MAX = 20                # For cosine annealing
SCHEDULER_PATIENCE = 10   # For plateau scheduler

# Early stopping
EARLY_STOPPING = True
PATIENCE = 10
MIN_DELTA = 1e-4          # Minimum improvement to count as progress

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
EVAL_TIME_WINDOWS = [7, 30, 90]  # Days for AUROC evaluation

# Metrics to track
TRACK_METRICS = [
    'c_index',        # Concordance index (Harrell)
    'mae',            # Mean absolute error on time-to-event
    'rmse',           # Root mean squared error
    'auroc_7d',       # AUROC for 7-day window
    'auroc_30d',      # AUROC for 30-day window
    'auroc_90d',      # AUROC for 90-day window
]

PRIMARY_METRIC = 'c_index'  # For early stopping and model selection

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
MODELS_TO_TRAIN = ['TGN', 'TGAT', 'HGT']  # Which models to train
# Options: 'TGN', 'TGAT', 'HGT' or ['TGN'] for single model

# ============================================================================
# SANITY CHECKS
# ============================================================================
RUN_SANITY_CHECKS = True
MAX_SANITY_SAMPLES = 1000  # For quick sanity checks

print(f"Configuration loaded. Device: {DEVICE}")

