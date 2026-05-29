"""
models/gru_model.py
───────────────────
Gated Recurrent Unit (GRU) model for traffic flow prediction.

Architecture:
    Input (batch, seq_len, 1)
      → GRU (num_layers, hidden_size, dropout)
      → last hidden state
      → Dropout
      → Linear(hidden_size → pred_len)
      → Output (batch, pred_len)
"""

import torch
import torch.nn as nn
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class GRUModel(nn.Module):
    def __init__(
        self,
        input_size:  int   = config.INPUT_SIZE,
        hidden_size: int   = config.HIDDEN_SIZE,
        num_layers:  int   = config.NUM_LAYERS,
        pred_len:    int   = config.PRED_LEN,
        dropout:     float = config.DROPOUT,
    ):
        super().__init__()
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
        # x: (batch, seq_len, input_size)
        out, _ = self.gru(x)        # (batch, seq_len, hidden_size)
        last   = self.dropout(out[:, -1, :])
        return self.fc(last)        # (batch, pred_len)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = GRUModel()
    x = torch.randn(8, config.SEQ_LEN, config.INPUT_SIZE)
    print("Output shape :", m(x).shape)
    print("Params       :", f"{m.count_parameters():,}")
