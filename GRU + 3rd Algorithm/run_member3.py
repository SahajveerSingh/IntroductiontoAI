"""
run_member3.py
──────────────
Entry point for Member 3's work.

Usage:
    python run_member3.py              # train GRU + Transformer, evaluate, plot
    python run_member3.py --quick      # 5-epoch smoke test
    python run_member3.py --all        # include LSTM results if Member 2 provides
                                       # a checkpoint at outputs/checkpoints/LSTM_best.pt

The script:
  1. Loads (or generates synthetic) traffic data via data/loader.py
  2. Trains GRU and Transformer from scratch
  3. Optionally loads pre-trained LSTM weights (from Member 2)
  4. Evaluates all models on the shared test set
  5. Produces all comparison plots + prints summary table
  6. Saves results to outputs/results.json for the report
"""

import os, sys, json, argparse, random
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data.loader import load_datasets
from models.gru_model import GRUModel
from models.transformer_model import TransformerModel
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


def main(quick: bool = False, include_lstm: bool = False):
    set_seed(config.SEED)
    epochs = 5 if quick else config.EPOCHS

    # ── 1. Load data ──────────────────────────────────────────────────────────
    train_dl, val_dl, test_dl, scaler = load_datasets()

    # ── 2. Train GRU ──────────────────────────────────────────────────────────
    gru = GRUModel()
    gru_history = train_model(gru, train_dl, val_dl,
                              model_name="GRU", epochs=epochs)

    # ── 3. Train Transformer ──────────────────────────────────────────────────
    transformer = TransformerModel()
    tra_history = train_model(transformer, train_dl, val_dl,
                              model_name="Transformer", epochs=epochs)

    # ── 4. Optionally load LSTM (Member 2's checkpoint) ───────────────────────
    lstm_result  = None
    lstm_history = None
    if include_lstm:
        lstm_ckpt = os.path.join(config.MODEL_DIR, "LSTM_best.pt")
        if os.path.exists(lstm_ckpt):
            # Import Member 2's model — assumes same interface
            try:
                from models.lstm_model import LSTMModel
                lstm = LSTMModel()
                lstm.load_state_dict(torch.load(lstm_ckpt, map_location="cpu"))
                lstm_result = evaluate_model(lstm, test_dl, scaler, "LSTM")
                # Load history if saved
                hist_path = os.path.join(config.OUTPUT_DIR, "LSTM_history.json")
                if os.path.exists(hist_path):
                    with open(hist_path) as f:
                        lstm_history = json.load(f)
            except ImportError:
                print("[run] lstm_model.py not found — skipping LSTM comparison.")
        else:
            print(f"[run] LSTM checkpoint not found at {lstm_ckpt} — skipping.")

    # ── 5. Evaluate ───────────────────────────────────────────────────────────
    gru_result = evaluate_model(gru,         test_dl, scaler, "GRU")
    tra_result = evaluate_model(transformer, test_dl, scaler, "Transformer")

    results   = [gru_result, tra_result]
    histories = {"GRU": gru_history, "Transformer": tra_history}
    if lstm_result:
        results.insert(0, lstm_result)   # LSTM first for conventional ordering
        if lstm_history:
            histories["LSTM"] = lstm_history

    # ── 6. Print summary table ────────────────────────────────────────────────
    print_summary_table(results, histories)

    # ── 7. Save histories (so Member 2 can slot LSTM in later) ────────────────
    for name, hist in histories.items():
        safe = {k: v for k, v in hist.items()
                if isinstance(v, (list, int, float, str))}
        path = os.path.join(config.OUTPUT_DIR, f"{name}_history.json")
        with open(path, "w") as f:
            json.dump(safe, f, indent=2)

    # ── 8. Save numeric results (for report / Member 4) ───────────────────────
    export = []
    for r in results:
        export.append({k: v for k, v in r.items()
                        if not isinstance(v, np.ndarray)})
    results_path = os.path.join(config.OUTPUT_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(export, f, indent=2)
    print(f"[run] Results saved → {results_path}")

    # ── 9. Plots ──────────────────────────────────────────────────────────────
    plot_metrics_bar(results)
    plot_loss_curves(histories)
    plot_prediction_overlay(results)
    plot_scatter(results)
    plot_error_distribution(results)
    print(f"\n[run] All plots saved to {config.PLOT_DIR}")
    print("[run] Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Member 3 — GRU + Transformer training")
    parser.add_argument("--quick",       action="store_true",
                        help="Run only 5 epochs (smoke test)")
    parser.add_argument("--all",         action="store_true",
                        help="Include LSTM checkpoint from Member 2")
    args = parser.parse_args()
    main(quick=args.quick, include_lstm=args.all)
