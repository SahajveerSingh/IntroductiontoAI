"""
evaluation/plotter.py
──────────────────────
Generate all comparison figures needed for the report Insights section.

Figures produced (saved to config.PLOT_DIR):
  1. metrics_bar.png        – MAE / RMSE / MAPE side-by-side bar chart
  2. loss_curves.png        – training & validation loss curves per model
  3. prediction_overlay.png – actual vs predicted for all models (first 200 test pts)
  4. scatter.png            – predicted vs actual scatter per model
  5. error_dist.png         – error distribution histogram per model
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

MODEL_COLORS = {"GRU": "#1D9E75", "Transformer": "#534AB7", "LSTM": "#D85A30"}


def _color(name):
    for k, v in MODEL_COLORS.items():
        if k.lower() in name.lower():
            return v
    return "#888780"


# ── 1. Metrics bar chart ──────────────────────────────────────────────────────

def plot_metrics_bar(results: list, save: bool = True):
    """
    results: list of dicts from evaluator.evaluate_model()
    """
    metrics  = ["MAE", "RMSE", "MAPE"]
    n_models = len(results)
    x        = np.arange(len(metrics))
    width    = 0.25
    offsets  = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, res in enumerate(results):
        vals = [res[m] for m in metrics]
        bars = ax.bar(x + offsets[i], vals, width * 0.9,
                      label=res["model"], color=_color(res["model"]), alpha=0.88)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(["MAE\n(cars/15 min)", "RMSE\n(cars/15 min)", "MAPE (%)"])
    ax.set_ylabel("Error")
    ax.set_title("Model comparison — test-set error metrics")
    ax.legend(frameon=False)
    fig.tight_layout()
    if save:
        path = os.path.join(config.PLOT_DIR, "metrics_bar.png")
        fig.savefig(path, bbox_inches="tight")
        print(f"[plotter] Saved → {path}")
    return fig


# ── 2. Loss curves ────────────────────────────────────────────────────────────

def plot_loss_curves(histories: dict, save: bool = True):
    """
    histories: {"ModelName": {"train_loss": [...], "val_loss": [...]}, ...}
    """
    fig, axes = plt.subplots(1, len(histories),
                             figsize=(5 * len(histories), 4), sharey=False)
    if len(histories) == 1:
        axes = [axes]
    for ax, (name, hist) in zip(axes, histories.items()):
        c = _color(name)
        ax.plot(hist["train_loss"], color=c, label="Train", linewidth=1.8)
        ax.plot(hist["val_loss"],   color=c, label="Val",
                linewidth=1.8, linestyle="--", alpha=0.75)
        best = hist.get("best_epoch")
        if best:
            ax.axvline(best - 1, color="gray", linewidth=0.8,
                       linestyle=":", label=f"Best (ep {best})")
        ax.set_title(name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE loss")
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Training & validation loss curves", y=1.02)
    fig.tight_layout()
    if save:
        path = os.path.join(config.PLOT_DIR, "loss_curves.png")
        fig.savefig(path, bbox_inches="tight")
        print(f"[plotter] Saved → {path}")
    return fig


# ── 3. Prediction overlay ─────────────────────────────────────────────────────

def plot_prediction_overlay(results: list, n_points: int = 200, save: bool = True):
    fig, ax = plt.subplots(figsize=(12, 4))
    y_true = results[0]["y_true"][:n_points]
    ax.plot(y_true, color="#444441", linewidth=1.5, label="Actual", zorder=5)
    for res in results:
        ax.plot(res["y_pred"][:n_points], color=_color(res["model"]),
                linewidth=1.2, linestyle="--", alpha=0.8, label=res["model"])
    ax.set_xlabel("Time step (× 15 min)")
    ax.set_ylabel("Traffic volume (cars / 15 min)")
    ax.set_title(f"Actual vs predicted — first {n_points} test samples")
    ax.legend(frameon=False)
    fig.tight_layout()
    if save:
        path = os.path.join(config.PLOT_DIR, "prediction_overlay.png")
        fig.savefig(path, bbox_inches="tight")
        print(f"[plotter] Saved → {path}")
    return fig


# ── 4. Scatter plot ───────────────────────────────────────────────────────────

def plot_scatter(results: list, save: bool = True):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5))
    if n == 1:
        axes = [axes]
    for ax, res in zip(axes, results):
        y_true, y_pred = res["y_true"], res["y_pred"]
        ax.scatter(y_true, y_pred, alpha=0.3, s=8, color=_color(res["model"]))
        lim = max(y_true.max(), y_pred.max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", linewidth=1, label="Perfect")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(res["model"])
        ax.set_aspect("equal")
    fig.suptitle("Predicted vs actual scatter", y=1.02)
    fig.tight_layout()
    if save:
        path = os.path.join(config.PLOT_DIR, "scatter.png")
        fig.savefig(path, bbox_inches="tight")
        print(f"[plotter] Saved → {path}")
    return fig


# ── 5. Error distribution ─────────────────────────────────────────────────────

def plot_error_distribution(results: list, save: bool = True):
    fig, ax = plt.subplots(figsize=(8, 4))
    for res in results:
        errors = res["y_pred"] - res["y_true"]
        ax.hist(errors, bins=60, alpha=0.55,
                color=_color(res["model"]), label=res["model"], density=True)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Prediction error (cars / 15 min)")
    ax.set_ylabel("Density")
    ax.set_title("Error distribution — all models")
    ax.legend(frameon=False)
    fig.tight_layout()
    if save:
        path = os.path.join(config.PLOT_DIR, "error_dist.png")
        fig.savefig(path, bbox_inches="tight")
        print(f"[plotter] Saved → {path}")
    return fig


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary_table(results: list, histories: dict = None):
    header = f"{'Model':<20} {'MAE':>8} {'RMSE':>8} {'MAPE%':>8} {'Params':>10}"
    if histories:
        header += f"  {'Best ep':>8}  {'Time(s)':>8}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for res in results:
        row = (f"{res['model']:<20} {res['MAE']:>8.2f} {res['RMSE']:>8.2f} "
               f"{res['MAPE']:>7.2f}% {res['params']:>10,}")
        if histories and res["model"] in histories:
            h = histories[res["model"]]
            row += f"  {h.get('best_epoch', '?'):>8}  {h.get('train_time', '?'):>8}"
        print(row)
    print("=" * len(header) + "\n")
