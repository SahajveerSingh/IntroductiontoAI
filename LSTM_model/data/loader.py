"""
data/loader.py
─────────────
Loads and preprocesses the Boroondara VicRoads traffic dataset.

Expected CSV columns (case-insensitive, extras ignored):
    SCATS_SITE  – integer site number
    DATE        – date string (any format pandas can parse)
    QT_INTERVAL_COUNT – 15-min volume count  (OR 'volume' / 'flow')

Usage:
    from data.loader import load_datasets
    train_dl, val_dl, test_dl, scaler = load_datasets()
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ── Column normalisation map ───────────────────────────────────────────────────
_COL_ALIASES = {
    "qt_interval_count": "volume",
    "flow": "volume",
    "traffic_volume": "volume",
    "count": "volume",
    "scats_site": "site",
    "site_no": "site",
    "nb_scats_site": "site",
}


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns=_COL_ALIASES)
    return df


def _parse_dataset(path: str) -> pd.DataFrame:
    """
    Parse the VicRoads CSV.  Handles two common layouts:
    - Wide: one row per (site, date), 96 columns V00..V95 for each 15-min slot.
    - Long: one row per (site, datetime, volume).
    Returns a long-format DataFrame with columns: site, datetime, volume.
    """
    df = pd.read_csv(path, low_memory=False)
    df = _normalise_columns(df)

    # ── Wide format detection: columns named V00, V01 … V95 ──────────────────
    v_cols = [c for c in df.columns if c.startswith("v") and c[1:].isdigit()]
    if len(v_cols) >= 96:
        # Melt into long format
        id_cols = [c for c in ["site", "date"] if c in df.columns]
        df = df.melt(id_vars=id_cols, value_vars=v_cols,
                     var_name="slot", value_name="volume")
        df["slot_minutes"] = df["slot"].str[1:].astype(int) * 15
        df["datetime"] = (pd.to_datetime(df["date"])
                          + pd.to_timedelta(df["slot_minutes"], unit="m"))
        df = df.drop(columns=["date", "slot", "slot_minutes"])
    else:
        # Long format — find datetime column
        date_col = next((c for c in df.columns
                         if "date" in c or "time" in c), None)
        if date_col and date_col != "datetime":
            df = df.rename(columns={date_col: "datetime"})
        df["datetime"] = pd.to_datetime(df["datetime"])

    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df = df.dropna(subset=["volume"])
    df["volume"] = df["volume"].clip(lower=0)
    return df[["site", "datetime", "volume"]].sort_values(["site", "datetime"])


def _generate_synthetic(n_sites: int = 5, days: int = 31) -> pd.DataFrame:
    """
    Fallback: generate realistic-looking synthetic traffic data so the code
    runs end-to-end before the real dataset is available.
    """
    rng = np.random.default_rng(config.SEED)
    records = []
    base_time = pd.Timestamp("2006-10-01")
    steps = days * 24 * 4  # 15-min intervals
    sites = [2000 + i * 100 for i in range(n_sites)]
    for site in sites:
        for t in range(steps):
            dt = base_time + pd.Timedelta(minutes=15 * t)
            hour = dt.hour + dt.minute / 60
            # Double-peaked daily profile (AM/PM peaks)
            am = np.exp(-0.5 * ((hour - 8.0) / 1.2) ** 2)
            pm = np.exp(-0.5 * ((hour - 17.0) / 1.5) ** 2)
            base = 80 * (am + 0.85 * pm)
            noise = rng.normal(0, 5)
            records.append({"site": site, "datetime": dt,
                             "volume": max(0, base + noise)})
    return pd.DataFrame(records)


def _make_windows(series: np.ndarray, seq_len: int, pred_len: int):
    X, y = [], []
    for i in range(len(series) - seq_len - pred_len + 1):
        X.append(series[i: i + seq_len])
        y.append(series[i + seq_len: i + seq_len + pred_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class TrafficDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # X: (N, seq_len, 1)   y: (N, pred_len)
        self.X = torch.tensor(X).unsqueeze(-1)   # add feature dim
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_datasets(verbose: bool = True):
    """
    Returns (train_loader, val_loader, test_loader, scaler).
    scaler is fitted on training volume only — use it to inverse-transform
    predictions back to cars/15-min.
    """
    if os.path.exists(config.DATA_FILE):
        if verbose:
            print(f"[loader] Reading dataset from {config.DATA_FILE}")
        df = _parse_dataset(config.DATA_FILE)
    else:
        if verbose:
            print("[loader] Dataset file not found — using synthetic data for testing.")
        df = _generate_synthetic()

    # ── Site filtering ─────────────────────────────────────────────────────────
    if config.TARGET_SITES:
        df = df[df["site"].isin(config.TARGET_SITES)]

    # ── Aggregate across all selected sites (mean per timestamp) ──────────────
    series = (df.groupby("datetime")["volume"]
                .mean()
                .sort_index()
                .values
                .reshape(-1, 1)
                .astype(np.float32))

    # ── Normalise ─────────────────────────────────────────────────────────────
    n = len(series)
    train_end = int(n * config.TRAIN_RATIO)
    val_end   = int(n * (config.TRAIN_RATIO + config.VAL_RATIO))

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(series[:train_end])
    series_norm = scaler.transform(series).flatten()

    # ── Window generation ─────────────────────────────────────────────────────
    X_all, y_all = _make_windows(series_norm, config.SEQ_LEN, config.PRED_LEN)

    # Recompute split indices on windowed arrays
    n_w = len(X_all)
    t_end = int(n_w * config.TRAIN_RATIO)
    v_end = int(n_w * (config.TRAIN_RATIO + config.VAL_RATIO))

    X_tr, y_tr = X_all[:t_end], y_all[:t_end]
    X_va, y_va = X_all[t_end:v_end], y_all[t_end:v_end]
    X_te, y_te = X_all[v_end:], y_all[v_end:]

    if verbose:
        print(f"[loader] Samples — train: {len(X_tr)}, val: {len(X_va)}, test: {len(X_te)}")

    def make_loader(X, y, shuffle):
        return DataLoader(TrafficDataset(X, y),
                          batch_size=config.BATCH_SIZE,
                          shuffle=shuffle,
                          num_workers=0)

    return (make_loader(X_tr, y_tr, True),
            make_loader(X_va, y_va, False),
            make_loader(X_te, y_te, False),
            scaler)
