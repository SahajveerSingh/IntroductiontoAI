"""
data/loader.py
──────────────
Loads Member 1's preprocessed SCATS data and returns PyTorch DataLoaders.

Loading priority:
  1. data/model_ready_sequences_window12.npz  exists → load directly (fast path).
  2. data/Scats_Data_October_2006.xls  exists → run Member 1's pipeline → load NPZ.
  3. Neither found → synthetic fallback (smoke-test only; not for final results).

Dataset facts (Member 1, October 2006 Boroondara):
  - 40 SCATS sites, 139 location/direction entries
  - Chronological split: train 01-21 Oct / val 22-26 Oct / test 27-31 Oct
  - Train sequences: 276,348 | Val: 59,880 | Test: 61,248  (window=12)
  - Scaling: min=0.0, max=636.0 fitted on training only; val/test may exceed 1.0
  - 2,976 duplicate timestamps resolved by averaging (Member 1's pipeline)
"""

import os, sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ── Inverse-scale helper (used by evaluator and Member 4) ─────────────────────

def inverse_scale(values: np.ndarray) -> np.ndarray:
    """Convert normalised predictions → raw traffic volume (cars/15 min)."""
    return values * (config.TRAIN_FLOW_MAX - config.TRAIN_FLOW_MIN) + config.TRAIN_FLOW_MIN


# ── Dataset wrapper ───────────────────────────────────────────────────────────

class TrafficDataset(Dataset):
    """
    Wraps X (N, seq_len) and y (N,) numpy arrays into a PyTorch Dataset.
    Adds the feature dimension expected by GRU/Transformer:
        X → (N, seq_len, 1)
        y → (N, 1)
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_npz(path: str, verbose: bool) -> dict:
    data = np.load(path)
    arrays = {k: data[k] for k in data.files}
    if verbose:
        for k, v in arrays.items():
            print(f"  [loader]   {k}: {v.shape}")
    return arrays


def _run_member1_pipeline(xls_path: str, out_dir: str, verbose: bool) -> str:
    """Call Member 1's data_preprocessing.main() and return path to the NPZ."""
    import importlib.util, pathlib

    script = os.path.join(os.path.dirname(__file__), "data_preprocessing.py")
    if not os.path.exists(script):
        raise FileNotFoundError(
            f"data_preprocessing.py not found at {script}. "
            "Copy Member 1's script into the data/ folder."
        )
    if verbose:
        print("[loader] Running Member 1's preprocessing pipeline …")

    spec   = importlib.util.spec_from_file_location("data_preprocessing", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Redirect output paths to our data/ directory
    module.OUTPUT_DIR   = pathlib.Path(out_dir)
    module.DATASET_PATH = pathlib.Path(xls_path)
    module.main()

    npz = os.path.join(out_dir, "model_ready_sequences_window12.npz")
    if not os.path.exists(npz):
        raise RuntimeError("Member 1's pipeline ran but NPZ was not found.")
    return npz


def _synthetic(verbose: bool) -> dict:
    """Synthetic fallback matching Member 1's array shapes and naming."""
    if verbose:
        print("[loader] ⚠  Using SYNTHETIC data — smoke-test only.")
        print("[loader]    Add Scats_Data_October_2006.xls to data/ for real results.")
    rng = np.random.default_rng(config.SEED)

    def _make(n):
        return (rng.random((n, config.SEQ_LEN), dtype=np.float32),
                rng.random(n, dtype=np.float32))

    X_tr, y_tr = _make(2_000)
    X_va, y_va = _make(400)
    X_te, y_te = _make(400)
    return dict(X_train=X_tr, y_train=y_tr,
                X_validation=X_va, y_validation=y_va,
                X_test=X_te, y_test=y_te)


# ── Public API ────────────────────────────────────────────────────────────────

def load_datasets(verbose: bool = True):
    """
    Returns (train_loader, val_loader, test_loader).

    No scaler object is returned — Member 1's pipeline already normalised the
    data.  Use inverse_scale() or config.TRAIN_FLOW_MIN/MAX directly.

    DataLoader tensor shapes:
        X : (batch_size, seq_len=12, 1)
        y : (batch_size, 1)
    """
    if os.path.exists(config.NPZ_PATH):
        if verbose:
            print(f"[loader] Loading pre-processed NPZ: {config.NPZ_PATH}")
        arrays = _load_npz(config.NPZ_PATH, verbose)

    elif os.path.exists(config.RAW_XLS):
        if verbose:
            print(f"[loader] Raw XLS found — running Member 1's pipeline …")
        npz = _run_member1_pipeline(config.RAW_XLS, config.DATA_DIR, verbose)
        arrays = _load_npz(npz, verbose)

    else:
        arrays = _synthetic(verbose)

    # Member 1 uses 'validation' (not 'val') as the split key
    X_tr = arrays["X_train"]
    y_tr = arrays["y_train"]
    X_va = arrays.get("X_validation", arrays.get("X_val"))
    y_va = arrays.get("y_validation", arrays.get("y_val"))
    X_te = arrays["X_test"]
    y_te = arrays["y_test"]

    if verbose:
        print(f"[loader] Sequences — train: {len(X_tr):,} | "
              f"val: {len(X_va):,} | test: {len(X_te):,}")

    def _dl(X, y, shuffle):
        return DataLoader(TrafficDataset(X, y),
                          batch_size=config.BATCH_SIZE,
                          shuffle=shuffle,
                          num_workers=0)

    return _dl(X_tr, y_tr, True), _dl(X_va, y_va, False), _dl(X_te, y_te, False)
