"""
evaluation/evaluator.py
────────────────────────
Compute MAE, RMSE, MAPE on the test set and return predictions for plotting.
Also exposes predict_next() for single-step inference used by Member 4's
travel-time integration.
"""

import os, sys
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ── Metric functions ──────────────────────────────────────────────────────────

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mape(y_true: np.ndarray, y_pred: np.ndarray,
         min_threshold: float = 5.0) -> float:
    """
    Mean Absolute Percentage Error.
    Excludes samples where actual volume < min_threshold (cars/15 min) to
    avoid division-by-near-zero inflating the metric during off-peak hours.
    """
    mask   = y_true >= min_threshold
    if mask.sum() == 0:
        return float("nan")
    y_t = y_true[mask]
    y_p = y_pred[mask]
    return float(np.mean(np.abs((y_t - y_p) / y_t)) * 100)


# ── Main evaluation function ──────────────────────────────────────────────────

def evaluate_model(
    model,
    test_loader,
    scaler,
    model_name: str = "model",
    device: str = None,
) -> dict:
    """
    Run the model on the test set.

    Returns a dict:
    {
        "model":   model_name,
        "MAE":     float,   # in original traffic volume units (cars/15min)
        "RMSE":    float,
        "MAPE":    float,   # percent
        "y_true":  np.ndarray,
        "y_pred":  np.ndarray,
        "params":  int,
    }
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    all_true, all_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            pred    = model(X_batch).cpu().numpy()
            all_pred.append(pred)
            all_true.append(y_batch.numpy())

    y_pred_norm = np.concatenate(all_pred).flatten()
    y_true_norm = np.concatenate(all_true).flatten()

    # Inverse transform to original scale
    y_pred = scaler.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()
    y_true = scaler.inverse_transform(y_true_norm.reshape(-1, 1)).flatten()

    results = {
        "model":  model_name,
        "MAE":    round(mae(y_true, y_pred),  4),
        "RMSE":   round(rmse(y_true, y_pred), 4),
        "MAPE":   round(mape(y_true, y_pred), 4),
        "y_true": y_true,
        "y_pred": y_pred,
        "params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    print(f"[eval] {model_name:20s} | MAE={results['MAE']:.2f} | "
          f"RMSE={results['RMSE']:.2f} | MAPE={results['MAPE']:.2f}%")
    return results


# ── Single-step inference for Member 4 integration ────────────────────────────

def predict_next(
    model,
    recent_volumes: list,
    scaler,
    device: str = None,
) -> float:
    """
    Predict the next 15-min traffic volume given the last SEQ_LEN readings.

    Args:
        model          : trained PyTorch model
        recent_volumes : list of the last config.SEQ_LEN raw volume values
        scaler         : fitted MinMaxScaler from load_datasets()

    Returns:
        Predicted volume (cars per 15 min), in original units.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    arr  = np.array(recent_volumes, dtype=np.float32).reshape(-1, 1)
    norm = scaler.transform(arr).flatten()
    x    = torch.tensor(norm).unsqueeze(0).unsqueeze(-1).to(device)  # (1, seq, 1)
    with torch.no_grad():
        pred_norm = model(x).cpu().numpy().flatten()[0]
    pred = scaler.inverse_transform([[pred_norm]])[0][0]
    return float(max(0.0, pred))
