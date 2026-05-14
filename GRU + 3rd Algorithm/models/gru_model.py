"""
models/gru_model.py
───────────────────
Gated Recurrent Unit (GRU) model for traffic flow prediction.

Architecture:
    Input  → GRU (num_layers, hidden_size, dropout) → last hidden state
           → Linear(hidden_size → pred_len) → Output
"""

import torch
import torch.nn as nn
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class GRUModel(nn.Module):
    """
    Multi-layer GRU followed by a fully-connected output head.

    Args:
        input_size   : number of input features per time-step (default 1)
        hidden_size  : number of GRU hidden units
        num_layers   : number of stacked GRU layers
        pred_len     : number of future steps to predict
        dropout      : dropout probability between GRU layers (ignored if num_layers=1)
    """

    def __init__(
        self,
        input_size:  int = config.INPUT_SIZE,
        hidden_size: int = config.HIDDEN_SIZE,
        num_layers:  int = config.NUM_LAYERS,
        pred_len:    int = config.PRED_LEN,
        dropout:     float = config.DROPOUT,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.gru = nn.GRU(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, seq_len, input_size)
        returns : (batch, pred_len)
        """
        out, _ = self.gru(x)          # out: (batch, seq_len, hidden_size)
        last    = out[:, -1, :]       # take last time-step
        last    = self.dropout(last)
        return self.fc(last)          # (batch, pred_len)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = GRUModel()
    dummy = torch.randn(8, config.SEQ_LEN, config.INPUT_SIZE)
    out   = model(dummy)
    print(f"GRU output shape : {out.shape}")
    print(f"Trainable params : {model.count_parameters():,}")
