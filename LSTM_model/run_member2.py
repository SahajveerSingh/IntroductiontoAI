"""
run_member2.py
──────────────
Entry point for Member 2's LSTM work.

Usage:
    python run_member2.py              # train LSTM, evaluate, plot
    python run_member2.py --quick      # 5-epoch smoke test

The script:
  1. Loads the Boroondara traffic CSV via data/loader.py
  2. Trains the LSTM from scratch
  3. Evaluates on the shared test set
  4. Produces all comparison-ready plots + prints summary table
  5. Saves results to outputs/results.json for the report
"""

import os, sys, json, argparse, random
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data.loader import load_datasets
from models.lstm_model import LSTMModel
from utils.trainer import train_model
from evaluation.evaluator import evaluate_model
from evaluation.plotter import (
    plot_metrics_bar, plot_loss_curves, plot_prediction_overlay,
    plot_scatter, plot_error_distribution, print_summary_table,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(quick: bool = False):
    set_seed(config.SEED)
    epochs = 5 if quick else config.EPOCHS

    # ── 1. Load data ──────────────────────────────────────────────────────────
    train_dl, val_dl, test_dl, scaler = load_datasets()

    # ── 2. Train LSTM ─────────────────────────────────────────────────────────
    lstm = LSTMModel()
    lstm_history = train_model(lstm, train_dl, val_dl,
                               model_name="LSTM", epochs=epochs)

    # ── 3. Evaluate ───────────────────────────────────────────────────────────
    lstm_result = evaluate_model(lstm, test_dl, scaler, "LSTM")

    results   = [lstm_result]
    histories = {"LSTM": lstm_history}

    # ── 4. Print summary table ────────────────────────────────────────────────
    print_summary_table(results, histories)

    # ── 5. Save history ───────────────────────────────────────────────────────
    safe_hist = {k: v for k, v in lstm_history.items()
                 if isinstance(v, (list, int, float, str))}
    hist_path = os.path.join(config.OUTPUT_DIR, "LSTM_history.json")
    with open(hist_path, "w") as f:
        json.dump(safe_hist, f, indent=2)
    print(f"[run] History saved -> {hist_path}")

    # ── 6. Save numeric results (for report / Member 4) ───────────────────────
    export = [{k: v for k, v in lstm_result.items()
               if not isinstance(v, np.ndarray)}]
    results_path = os.path.join(config.OUTPUT_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(export, f, indent=2)
    print(f"[run] Results saved -> {results_path}")

    # ── 7. Plots ──────────────────────────────────────────────────────────────
    plot_metrics_bar(results)
    plot_loss_curves(histories)
    plot_prediction_overlay(results)
    plot_scatter(results)
    plot_error_distribution(results)
    print(f"\n[run] All plots saved to {config.PLOT_DIR}")
    print("[run] Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Member 2 — LSTM traffic flow prediction")
    parser.add_argument("--quick", action="store_true",
                        help="Run only 5 epochs (smoke test)")
    args = parser.parse_args()
    main(quick=args.quick)
