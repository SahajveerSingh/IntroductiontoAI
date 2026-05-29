"""
utils/trainer.py
────────────────
Generic training loop shared by all Member 3 models (GRU, Transformer).
Member 2 (LSTM) should use the same loop for a consistent comparison.

Features:
  - MSELoss + Adam optimiser
  - ReduceLROnPlateau scheduler
  - Early stopping on validation loss
  - Gradient clipping (max_norm=1.0)
  - Best-weights checkpoint saved to config.MODEL_DIR
  - Returns history dict for plotting
"""

import os, sys, time, copy
import torch
import torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def train_model(
    model,
    train_loader,
    val_loader,
    model_name: str   = "model",
    epochs:     int   = config.EPOCHS,
    lr:         float = config.LEARNING_RATE,
    patience:   int   = config.PATIENCE,
    device:     str   = None,
) -> dict:
    """
    Train a model and return a history dict:
    {
      "train_loss": [...],   # per-epoch MSE
      "val_loss":   [...],
      "best_epoch": int,
      "train_time": float,   # seconds
    }
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=patience // 2, min_lr=1e-6
    )

    best_val   = float("inf")
    best_w     = copy.deepcopy(model.state_dict())
    no_improve = 0
    history    = {"train_loss": [], "val_loss": []}
    ckpt       = os.path.join(config.MODEL_DIR, f"{model_name}_best.pt")

    print(f"\n[trainer] {model_name} | device={device} | epochs={epochs} | "
          f"patience={patience} | batch={config.BATCH_SIZE}")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        t_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimiser.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            t_loss += loss.item() * len(Xb)
        t_loss /= len(train_loader.dataset)

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                v_loss += criterion(model(Xb), yb).item() * len(Xb)
        v_loss /= len(val_loader.dataset)

        scheduler.step(v_loss)
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:03d}/{epochs} | "
                  f"train={t_loss:.6f} | val={v_loss:.6f}")

        if v_loss < best_val:
            best_val = v_loss
            best_w   = copy.deepcopy(model.state_dict())
            history["best_epoch"] = epoch
            no_improve = 0
            torch.save(best_w, ckpt)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stop at epoch {epoch} "
                      f"(best: epoch {history['best_epoch']})")
                break

    history["train_time"] = round(time.time() - t0, 2)
    model.load_state_dict(best_w)
    print(f"[trainer] Done. best_val={best_val:.6f} | "
          f"time={history['train_time']}s | saved → {ckpt}")
    return history
