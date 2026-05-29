"""
models/transformer_model.py
────────────────────────────
Transformer encoder model for traffic flow prediction.

Architecture:
    Input (batch, seq_len, 1)
      → Linear projection → d_model
      → Sinusoidal positional encoding
      → TransformerEncoder (N × pre-norm EncoderLayer)
      → Global average pool over sequence
      → Linear → Output (batch, pred_len)

Why Transformer for traffic prediction?
    Self-attention attends to ALL past time-steps simultaneously, capturing
    long-range temporal patterns (e.g. recurring AM/PM peaks across days)
    without the vanishing-gradient issues of RNNs.
"""

import math, warnings
import torch
import torch.nn as nn
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017)."""
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1), :])


class TransformerModel(nn.Module):
    def __init__(
        self,
        input_size:      int   = config.INPUT_SIZE,
        d_model:         int   = config.HIDDEN_SIZE,
        nhead:           int   = config.NHEAD,
        num_layers:      int   = config.NUM_LAYERS,
        dim_feedforward: int   = config.DIM_FEEDFORWARD,
        pred_len:        int   = config.PRED_LEN,
        dropout:         float = config.DROPOUT,
    ):
        super().__init__()
        assert d_model % nhead == 0, \
            f"d_model ({d_model}) must be divisible by nhead ({nhead})"

        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc    = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
            norm_first=True,   # pre-norm for more stable training
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.transformer = nn.TransformerEncoder(encoder_layer,
                                                      num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.fc   = nn.Linear(d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        x = self.pos_enc(self.input_proj(x))   # (batch, seq_len, d_model)
        x = self.norm(self.transformer(x))
        return self.fc(x.mean(dim=1))           # (batch, pred_len)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = TransformerModel()
    x = torch.randn(8, config.SEQ_LEN, config.INPUT_SIZE)
    print("Output shape :", m(x).shape)
    print("Params       :", f"{m.count_parameters():,}")
