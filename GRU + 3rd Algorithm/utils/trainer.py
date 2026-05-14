"""
utils/trainer.py
────────────────
Generic training loop shared by all Member 3 models (GRU, Transformer).
Also used as a reference by Member 2 (LSTM) for consistency.

Features:
  - MSE loss with Adam optimiser
  - ReduceLROnPlateau scheduler
  - Early stopping on validation loss
  - Checkpoint saving (best model weights)
  - Returns history dict for plotting
"""

import os, sys, time, copy
import torch
import torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def train_model(
    model:       nn.Module,
    train_loader,
    val_loader,
    model_name:  str  = "model",
    epochs:      int  = config.EPOCHS,
    lr:          float = config.LEARNING_RATE,
    patience:    int  = config.PATIENCE,
    device:      str  = None,
) -> dict:
    """
    Train a model and return a history dictionary.

    Returns:
        {
          "train_loss": [...],   # per-epoch MSE on training set
          "val_loss":   [...],   # per-epoch MSE on validation set
          "best_epoch": int,
          "train_time": float,   # seconds
        }
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    criterion  = nn.MSELoss()
    optimiser  = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=patience // 2, min_lr=1e-6
    )

    best_val_loss = float("inf")
    best_weights  = copy.deepcopy(model.state_dict())
    no_improve    = 0
    history       = {"train_loss": [], "val_loss": []}
    ckpt_path     = os.path.join(config.MODEL_DIR, f"{model_name}_best.pt")

    print(f"\n[trainer] Training {model_name} on {device} "
          f"| epochs={epochs} | patience={patience}")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        # ── Training ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimiser.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            train_loss += loss.item() * len(X_batch)
        train_loss /= len(train_loader.dataset)

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred  = model(X_batch)
                loss  = criterion(pred, y_batch)
                val_loss += loss.item() * len(X_batch)
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:03d}/{epochs} | "
                  f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        # ── Early stopping ────────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss         = val_loss
            best_weights          = copy.deepcopy(model.state_dict())
            history["best_epoch"] = epoch
            no_improve            = 0
            torch.save(best_weights, ckpt_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch} "
                      f"(best epoch: {history['best_epoch']})")
                break

    history["train_time"] = round(time.time() - t0, 2)
    model.load_state_dict(best_weights)   # restore best
    print(f"[trainer] Done. Best val_loss={best_val_loss:.6f} "
          f"| time={history['train_time']}s | saved → {ckpt_path}")
    return history
