"""
data/loader.py
─────────────
Loads preprocessed Boroondara traffic data from Member 1's pipeline.

Primary source  : model_ready_sequences_window12.npz
                  (X_train, y_train, X_val, y_val, X_test, y_test)
Fallback source : processed_scats_time_series_with_split_scaled.csv
                  (long-format with traffic_flow_scaled + dataset_split)

Scaling reference: train_min=0.0, train_max=636.0 (fitted on training data
only by Member 1 — documented in data_quality_and_processing_summary.csv).
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ── Scaler reconstruction ──────────────────────────────────────────────────────
# Member 1 used min=0.0, max=636.0 fitted on training data only.
# We reconstruct a compatible MinMaxScaler so evaluator.py can inverse-transform
# predictions back to real vehicle counts without re-fitting on test data.

def _build_scaler(train_min: float = config.TRAIN_MIN,
                  train_max: float = config.TRAIN_MAX) -> MinMaxScaler:
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(np.array([[train_min], [train_max]]))
    return scaler


# ── PyTorch Dataset ────────────────────────────────────────────────────────────

class TrafficDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # X arrives as (N, seq_len) — add feature dim → (N, seq_len, 1)
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)  # (N, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── NPZ loader (primary path) ──────────────────────────────────────────────────

def _load_from_npz(path: str, verbose: bool) -> tuple:
    if verbose:
        print(f"[loader] Loading pre-built sequences from {path}")
    data = np.load(path)
    X_tr = data["X_train"]
    y_tr = data["y_train"]
    X_va = data["X_validation"]
    y_va = data["y_validation"]
    X_te = data["X_test"]
    y_te = data["y_test"]
    if verbose:
        print(f"[loader] train={len(X_tr):,}  val={len(X_va):,}  test={len(X_te):,}  "
              f"seq_len={X_tr.shape[1]}")
    return X_tr, y_tr, X_va, y_va, X_te, y_te


# ── CSV fallback loader ────────────────────────────────────────────────────────

def _make_windows(series: np.ndarray, seq_len: int) -> tuple:
    X, y = [], []
    for i in range(seq_len, len(series)):
        X.append(series[i - seq_len:i])
        y.append(series[i])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


def _load_from_csv(path: str, verbose: bool) -> tuple:
    if verbose:
        print(f"[loader] NPZ not found — falling back to CSV: {path}")
    df = pd.read_csv(path, low_memory=False)

    # Aggregate across all sites: mean flow per timestamp per split
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    arrays = {}
    for split, key in [("train", "tr"), ("validation", "va"), ("test", "te")]:
        subset = (df[df["dataset_split"] == split]
                  .groupby("timestamp")["traffic_flow_scaled"]
                  .mean()
                  .sort_index()
                  .values
                  .astype(np.float32))
        X, y = _make_windows(subset, config.SEQ_LEN)
        arrays[f"X_{key}"] = X
        arrays[f"y_{key}"] = y
        if verbose:
            print(f"[loader]   {split}: {len(X):,} sequences")

    return (arrays["X_tr"], arrays["y_tr"],
            arrays["X_va"], arrays["y_va"],
            arrays["X_te"], arrays["y_te"])


# ── Synthetic fallback ─────────────────────────────────────────────────────────

def _generate_synthetic(verbose: bool) -> tuple:
    if verbose:
        print("[loader] No data files found — generating synthetic data for testing.")
    rng = np.random.default_rng(config.SEED)
    steps = 31 * 96  # 31 days × 96 intervals
    t = np.linspace(0, 31 * 2 * np.pi, steps)
    series = (50 + 40 * np.sin(t) + 10 * np.sin(3 * t)
              + rng.normal(0, 3, steps)).clip(0).astype(np.float32)
    # Scale to [0,1] using training portion
    train_end = int(steps * config.TRAIN_RATIO)
    mn, mx = series[:train_end].min(), series[:train_end].max()
    series = (series - mn) / (mx - mn)

    X_all, y_all = _make_windows(series, config.SEQ_LEN)
    n = len(X_all)
    t_end = int(n * config.TRAIN_RATIO)
    v_end = int(n * (config.TRAIN_RATIO + config.VAL_RATIO))
    return (X_all[:t_end],  y_all[:t_end],
            X_all[t_end:v_end], y_all[t_end:v_end],
            X_all[v_end:],  y_all[v_end:])


# ── Public API ─────────────────────────────────────────────────────────────────

def load_datasets(verbose: bool = True) -> tuple:
    """
    Returns (train_loader, val_loader, test_loader, scaler).

    Load priority:
        1. config.NPZ_FILE  — Member 1's pre-built numpy sequences (fastest)
        2. config.CSV_FILE  — Member 1's processed long-format CSV (fallback)
        3. Synthetic data   — for offline testing when no data files exist

    The returned scaler is reconstructed from Member 1's documented scaling
    parameters (train_min=0, train_max=636) and should be used to
    inverse-transform model predictions back to real vehicle counts.
    """
    if os.path.exists(config.NPZ_FILE):
        X_tr, y_tr, X_va, y_va, X_te, y_te = _load_from_npz(config.NPZ_FILE, verbose)
    elif os.path.exists(config.CSV_FILE):
        X_tr, y_tr, X_va, y_va, X_te, y_te = _load_from_csv(config.CSV_FILE, verbose)
    else:
        X_tr, y_tr, X_va, y_va, X_te, y_te = _generate_synthetic(verbose)

    scaler = _build_scaler()

    def _make_loader(X, y, shuffle):
        return DataLoader(
            TrafficDataset(X, y),
            batch_size=config.BATCH_SIZE,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

    return (_make_loader(X_tr, y_tr, shuffle=True),
            _make_loader(X_va, y_va, shuffle=False),
            _make_loader(X_te, y_te, shuffle=False),
            scaler)
