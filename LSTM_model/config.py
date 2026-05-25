"""
config.py — shared configuration for Member 2 (LSTM) model.
Edit this file to change dataset paths, hyperparameters, or training settings.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
# Re-use the pre-processed CSV from Member 3's data folder (no copy needed)
DATA_DIR   = os.path.join(os.path.dirname(BASE_DIR), "GRU + 3rd Algorithm", "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR  = os.path.join(OUTPUT_DIR, "checkpoints")
PLOT_DIR   = os.path.join(OUTPUT_DIR, "plots")

for d in [OUTPUT_DIR, MODEL_DIR, PLOT_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Dataset ────────────────────────────────────────────────────────────────────
DATA_FILE    = os.path.join(DATA_DIR, "boroondara_traffic.csv")
TARGET_SITES = None          # None = aggregate all sites
SEQ_LEN      = 12            # 3 hours of history (12 × 15-min intervals)
PRED_LEN     = 1             # predict next 15-min volume
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
# TEST_RATIO = 0.15 (remainder)

# ── Training ───────────────────────────────────────────────────────────────────
BATCH_SIZE    = 64           # match Member 3 for fair comparison
EPOCHS        = 50
LEARNING_RATE = 1e-3
PATIENCE      = 10           # early-stopping patience
SEED          = 42

# ── Model ──────────────────────────────────────────────────────────────────────
INPUT_SIZE  = 1              # univariate: traffic volume only
HIDDEN_SIZE = 64
NUM_LAYERS  = 2
DROPOUT     = 0.2

# ── Evaluation ────────────────────────────────────────────────────────────────
METRICS = ["MAE", "RMSE", "MAPE"]
