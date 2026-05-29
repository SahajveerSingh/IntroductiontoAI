"""
evaluation/evaluator.py
────────────────────────
Compute MAE, RMSE, MAPE on the test set and return predictions for plotting.

Also exposes predict_next() for single-step inference used by Member 4's
travel-time integration.

Note on MAPE: samples where actual volume < min_threshold cars/15 min are
excluded to avoid division-by-near-zero inflating the metric during off-peak
hours (a known characteristic of the Boroondara dataset which has many
near-zero overnight readings — minimum flow in all splits is 0).
"""

import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from data.loader import inverse_scale


# ── Metric functions ──────────────────────────────────────────────────────────

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mape(y_true: np.ndarray, y_pred: np.ndarray,
         min_threshold: float = 5.0) -> float:
    """
    MAPE excluding near-zero actuals (< min_threshold cars/15 min).
    The Boroondara dataset has many overnight near-zero readings; including
    them would produce misleadingly large MAPE values.
    """
    mask = y_true >= min_threshold
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


# ── Main evaluation function ──────────────────────────────────────────────────

def evaluate_model(
    model,
    test_loader,
    model_name: str = "model",
    device: str = None,
) -> dict:
    """
    Run the model on the test set and compute error metrics in original units.

    Member 1's NPZ is already normalised with min=0, max=636 (training only).
    We inverse-transform using those constants (config.TRAIN_FLOW_MIN/MAX)
    to report errors in cars-per-15-min — the unit that matters for the report.

    Returns:
    {
        "model":   model_name,
        "MAE":     float,   # cars / 15 min
        "RMSE":    float,   # cars / 15 min
        "MAPE":    float,   # percent (near-zero samples excluded)
        "y_true":  np.ndarray,   # original scale
        "y_pred":  np.ndarray,   # original scale
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

    # Inverse-transform using Member 1's training-only scaling constants
    y_pred = inverse_scale(y_pred_norm)
    y_true = inverse_scale(y_true_norm)

    results = {
        "model":  model_name,
        "MAE":    round(mae(y_true, y_pred),  4),
        "RMSE":   round(rmse(y_true, y_pred), 4),
        "MAPE":   round(mape(y_true, y_pred), 4),
        "y_true": y_true,
        "y_pred": y_pred,
        "params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    print(f"[eval] {model_name:20s} | MAE={results['MAE']:.2f} cars/15min | "
          f"RMSE={results['RMSE']:.2f} | MAPE={results['MAPE']:.2f}%")
    return results


# ── Single-step inference for Member 4 integration ────────────────────────────

def predict_next(
    model,
    recent_volumes: list,
    device: str = None,
) -> float:
    """
    Predict the next 15-min traffic volume given the last SEQ_LEN raw readings.

    Args:
        model          : trained PyTorch model (GRU or Transformer)
        recent_volumes : list of the last config.SEQ_LEN raw volume values
                         (cars/15 min, in ORIGINAL scale — not normalised)

    Returns:
        Predicted volume in cars per 15 min (original scale).

    Usage by Member 4:
        from evaluation.evaluator import predict_next
        volume = predict_next(gru_model, last_12_readings)
        # then convert to travel time using the assignment formula
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    arr  = np.array(recent_volumes, dtype=np.float32)
    # Apply Member 1's scaling (same constants used during training)
    norm = (arr - config.TRAIN_FLOW_MIN) / (config.TRAIN_FLOW_MAX - config.TRAIN_FLOW_MIN)
    x    = torch.tensor(norm).unsqueeze(0).unsqueeze(-1).to(device)  # (1, seq_len, 1)

    with torch.no_grad():
        pred_norm = model(x).cpu().numpy().flatten()[0]

    return float(max(0.0, inverse_scale(np.array([pred_norm]))[0]))
