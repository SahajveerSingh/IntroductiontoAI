"""
evaluation/plotter.py
──────────────────────
Generate all comparison figures for the report Insights section.

Figures saved to config.PLOT_DIR:
  1. metrics_bar.png         – MAE / RMSE / MAPE side-by-side bar chart
  2. loss_curves.png         – train & val loss curves per model
  3. prediction_overlay.png  – actual vs predicted (first 200 test points)
  4. scatter.png             – predicted vs actual scatter per model
  5. error_dist.png          – prediction error distribution histogram
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size":   11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

COLORS = {"GRU": "#1D9E75", "Transformer": "#534AB7", "LSTM": "#D85A30"}

def _c(name):
    for k, v in COLORS.items():
        if k.lower() in name.lower():
            return v
    return "#888780"


def plot_metrics_bar(results: list, save: bool = True):
    metrics  = ["MAE", "RMSE", "MAPE"]
    labels   = ["MAE\n(cars/15 min)", "RMSE\n(cars/15 min)", "MAPE (%)"]
    n        = len(results)
    x        = np.arange(len(metrics))
    width    = 0.22
    offsets  = np.linspace(-(n-1)/2, (n-1)/2, n) * width

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, res in enumerate(results):
        vals = [res[m] for m in metrics]
        bars = ax.bar(x + offsets[i], vals, width * 0.9,
                      label=res["model"], color=_c(res["model"]), alpha=0.88)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Error")
    ax.set_title("Model comparison — test-set error metrics\n"
                 "(Boroondara dataset, October 2006, window=12)")
    ax.legend(frameon=False)
    fig.tight_layout()
    if save:
        p = os.path.join(config.PLOT_DIR, "metrics_bar.png")
        fig.savefig(p, bbox_inches="tight"); print(f"[plotter] → {p}")
    return fig


def plot_loss_curves(histories: dict, save: bool = True):
    n    = len(histories)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]
    for ax, (name, h) in zip(axes, histories.items()):
        c = _c(name)
        ax.plot(h["train_loss"], color=c, lw=1.8, label="Train")
        ax.plot(h["val_loss"],   color=c, lw=1.8, ls="--", alpha=0.75, label="Val")
        best = h.get("best_epoch")
        if best:
            ax.axvline(best - 1, color="gray", lw=0.8, ls=":",
                       label=f"Best (ep {best})")
        ax.set_title(name); ax.set_xlabel("Epoch"); ax.set_ylabel("MSE loss")
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Training & validation loss curves", y=1.02)
    fig.tight_layout()
    if save:
        p = os.path.join(config.PLOT_DIR, "loss_curves.png")
        fig.savefig(p, bbox_inches="tight"); print(f"[plotter] → {p}")
    return fig


def plot_prediction_overlay(results: list, n_points: int = 200, save: bool = True):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(results[0]["y_true"][:n_points], color="#333", lw=1.6,
            label="Actual", zorder=5)
    for res in results:
        ax.plot(res["y_pred"][:n_points], color=_c(res["model"]),
                lw=1.2, ls="--", alpha=0.8, label=res["model"])
    ax.set_xlabel("Time step (× 15 min)")
    ax.set_ylabel("Traffic volume (cars / 15 min)")
    ax.set_title(f"Actual vs predicted — first {n_points} test samples")
    ax.legend(frameon=False)
    fig.tight_layout()
    if save:
        p = os.path.join(config.PLOT_DIR, "prediction_overlay.png")
        fig.savefig(p, bbox_inches="tight"); print(f"[plotter] → {p}")
    return fig


def plot_scatter(results: list, save: bool = True):
    n    = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5))
    if n == 1:
        axes = [axes]
    for ax, res in zip(axes, results):
        yt, yp = res["y_true"], res["y_pred"]
        ax.scatter(yt, yp, alpha=0.25, s=6, color=_c(res["model"]))
        lim = max(yt.max(), yp.max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", lw=1)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
        ax.set_title(res["model"]); ax.set_aspect("equal")
    fig.suptitle("Predicted vs actual scatter (cars/15 min)", y=1.02)
    fig.tight_layout()
    if save:
        p = os.path.join(config.PLOT_DIR, "scatter.png")
        fig.savefig(p, bbox_inches="tight"); print(f"[plotter] → {p}")
    return fig


def plot_error_distribution(results: list, save: bool = True):
    fig, ax = plt.subplots(figsize=(8, 4))
    for res in results:
        ax.hist(res["y_pred"] - res["y_true"], bins=60,
                alpha=0.55, color=_c(res["model"]),
                label=res["model"], density=True)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Prediction error (cars / 15 min)")
    ax.set_ylabel("Density")
    ax.set_title("Error distribution — all models")
    ax.legend(frameon=False)
    fig.tight_layout()
    if save:
        p = os.path.join(config.PLOT_DIR, "error_dist.png")
        fig.savefig(p, bbox_inches="tight"); print(f"[plotter] → {p}")
    return fig


def print_summary_table(results: list, histories: dict = None):
    hdr = f"{'Model':<20} {'MAE':>8} {'RMSE':>8} {'MAPE%':>8} {'Params':>10}"
    if histories:
        hdr += f"  {'Best ep':>8}  {'Time(s)':>8}"
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for res in results:
        row = (f"{res['model']:<20} {res['MAE']:>8.2f} {res['RMSE']:>8.2f} "
               f"{res['MAPE']:>7.2f}% {res['params']:>10,}")
        if histories and res["model"] in histories:
            h = histories[res["model"]]
            row += f"  {h.get('best_epoch','?'):>8}  {h.get('train_time','?'):>8}"
        print(row)
    print("=" * len(hdr) + "\n")
