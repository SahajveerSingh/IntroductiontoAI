"""
config.py — shared configuration for Member 2 (LSTM) model.
Edit this file to change dataset paths, hyperparameters, or training settings.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(os.path.dirname(BASE_DIR), "data process")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR  = os.path.join(OUTPUT_DIR, "checkpoints")
PLOT_DIR   = os.path.join(OUTPUT_DIR, "plots")

for d in [OUTPUT_DIR, MODEL_DIR, PLOT_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Dataset files (from Member 1's data_preprocessing.py output) ──────────────
# Primary: pre-built numpy sequences — fastest to load
NPZ_FILE = os.path.join(DATA_DIR, "model_ready_sequences_window12.npz")
# Fallback: long-format CSV with traffic_flow_scaled + dataset_split columns
CSV_FILE = os.path.join(DATA_DIR, "processed_scats_time_series_with_split_scaled.csv")

# ── Scaling parameters (documented by Member 1) ────────────────────────────────
# Fitted on training data only — do NOT change these without re-running
# data_preprocessing.py, as they affect inverse-transform of predictions.
TRAIN_MIN = 0.0
TRAIN_MAX = 636.0

# ── Sequence settings (must match Member 1's window size) ─────────────────────
SEQ_LEN  = 12   # 12 × 15-min intervals = 3 hours of history
PRED_LEN = 1    # predict next 15-min volume

# ── Chronological split (matches Member 1's date-based split) ─────────────────
# train: Oct 1–21 (21 days), val: Oct 22–26 (5 days), test: Oct 27–31 (5 days)
TRAIN_RATIO = 0.70   # used only for synthetic fallback
VAL_RATIO   = 0.15   # used only for synthetic fallback

# ── Training ───────────────────────────────────────────────────────────────────
BATCH_SIZE    = 64
EPOCHS        = 50
LEARNING_RATE = 1e-3
PATIENCE      = 10
SEED          = 42

# ── Model ──────────────────────────────────────────────────────────────────────
INPUT_SIZE  = 1
HIDDEN_SIZE = 64
NUM_LAYERS  = 2
DROPOUT     = 0.2

# ── Evaluation ────────────────────────────────────────────────────────────────
METRICS = ["MAE", "RMSE", "MAPE"]
