"""
config.py — shared configuration for all Member 3 models.
Edit this file to change dataset paths, hyperparameters, or training settings.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR  = os.path.join(OUTPUT_DIR, "checkpoints")
PLOT_DIR   = os.path.join(OUTPUT_DIR, "plots")

for d in [OUTPUT_DIR, MODEL_DIR, PLOT_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Dataset ────────────────────────────────────────────────────────────────────
# Place the VicRoads CSV here; column names are normalised in data/loader.py.
DATA_FILE     = os.path.join(DATA_DIR, "boroondara_traffic.csv")
# Which SCATS site(s) to train on. None = use all.
TARGET_SITES  = None
# Each sample is SEQ_LEN time-steps (15-min intervals) → predict PRED_LEN steps ahead
SEQ_LEN       = 12      # 3 hours of history
PRED_LEN      = 1       # predict next 15-min volume
TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.15
# TEST_RATIO  = 0.15  (remainder)

# ── Training ───────────────────────────────────────────────────────────────────
BATCH_SIZE    = 64
EPOCHS        = 50
LEARNING_RATE = 1e-3
PATIENCE      = 10      # early-stopping patience (epochs)
SEED          = 42

# ── Model shared ──────────────────────────────────────────────────────────────
INPUT_SIZE    = 1       # univariate: traffic volume only
HIDDEN_SIZE   = 64
NUM_LAYERS    = 2
DROPOUT       = 0.2

# ── Transformer-specific ──────────────────────────────────────────────────────
NHEAD         = 4       # attention heads (must divide HIDDEN_SIZE)
DIM_FEEDFORWARD = 128

# ── Evaluation ────────────────────────────────────────────────────────────────
METRICS = ["MAE", "RMSE", "MAPE"]
