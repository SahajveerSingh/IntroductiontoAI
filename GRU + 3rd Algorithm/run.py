"""
run_member3.py
──────────────
Entry point for Member 3's work.

Usage:
    python run_member3.py              # full training (50 epochs)
    python run_member3.py --quick      # 5-epoch smoke test
    python run_member3.py --all        # include LSTM from Member 2's checkpoint

What this script does:
  1. Loads the Boroondara dataset via Member 1's pipeline (or NPZ if already processed)
  2. Trains GRU and Transformer models
  3. Optionally loads Member 2's pre-trained LSTM checkpoint for comparison
  4. Evaluates all models on the shared test set (MAE, RMSE, MAPE)
  5. Produces 5 comparison plots + summary table for the report Insights section
  6. Saves results.json for Member 4's integration reference
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

    # ── 1. Load data (Member 1's pipeline / NPZ) ──────────────────────────────
    train_dl, val_dl, test_dl = load_datasets()

    # ── 2. Train GRU ──────────────────────────────────────────────────────────
    gru         = GRUModel()
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
        ckpt = os.path.join(config.MODEL_DIR, "LSTM_best.pt")
        if os.path.exists(ckpt):
            try:
                from models.lstm_model import LSTMModel
                lstm = LSTMModel()
                lstm.load_state_dict(torch.load(ckpt, map_location="cpu"))
                lstm_result = evaluate_model(lstm, test_dl, "LSTM")
                hist_path   = os.path.join(config.OUTPUT_DIR, "LSTM_history.json")
                if os.path.exists(hist_path):
                    with open(hist_path) as f:
                        lstm_history = json.load(f)
            except ImportError:
                print("[run] lstm_model.py not found — skipping LSTM.")
        else:
            print(f"[run] LSTM checkpoint not found at {ckpt} — skipping.")

    # ── 5. Evaluate GRU and Transformer ───────────────────────────────────────
    gru_result = evaluate_model(gru,         test_dl, "GRU")
    tra_result = evaluate_model(transformer, test_dl, "Transformer")

    results   = [gru_result, tra_result]
    histories = {"GRU": gru_history, "Transformer": tra_history}
    if lstm_result:
        results.insert(0, lstm_result)
        if lstm_history:
            histories["LSTM"] = lstm_history

    # ── 6. Print summary table ────────────────────────────────────────────────
    print_summary_table(results, histories)

    # ── 7. Save histories as JSON (for Member 2 to slot LSTM in later) ────────
    for name, hist in histories.items():
        safe = {k: v for k, v in hist.items()
                if isinstance(v, (list, int, float, str))}
        with open(os.path.join(config.OUTPUT_DIR, f"{name}_history.json"), "w") as f:
            json.dump(safe, f, indent=2)

    # ── 8. Save numeric results for Member 4 / report ─────────────────────────
    export = [{k: v for k, v in r.items() if not isinstance(v, np.ndarray)}
              for r in results]
    rpath  = os.path.join(config.OUTPUT_DIR, "results.json")
    with open(rpath, "w") as f:
        json.dump(export, f, indent=2)
    print(f"[run] Results saved → {rpath}")

    # ── 9. Generate all comparison plots ──────────────────────────────────────
    plot_metrics_bar(results)
    plot_loss_curves(histories)
    plot_prediction_overlay(results)
    plot_scatter(results)
    plot_error_distribution(results)
    print(f"\n[run] All plots saved to {config.PLOT_DIR}")
    print("[run] Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Member 3 — GRU + Transformer training & evaluation"
    )
    parser.add_argument("--quick", action="store_true",
                        help="5-epoch smoke test")
    parser.add_argument("--all",   action="store_true",
                        help="Include Member 2's LSTM checkpoint in comparison")
    args = parser.parse_args()
    main(quick=args.quick, include_lstm=args.all)
