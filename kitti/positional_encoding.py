"""Positional encodings used by the GeoFlow models."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding2D(nn.Module):
    """2D sine/cosine positional encoding for spatial feature maps."""

    def __init__(self, d_model: int, max_h: int = 32, max_w: int = 32):
        super().__init__()
        if d_model % 4 != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by 4.")

        pe = torch.zeros(d_model, max_h, max_w)
        half_dim = d_model // 2
        div_term = torch.exp(
            torch.arange(0.0, half_dim, 2) * -(math.log(10000.0) / half_dim)
        )

        pos_w = torch.arange(0.0, max_w).unsqueeze(1)
        pos_h = torch.arange(0.0, max_h).unsqueeze(1)

        pe[0:half_dim:2, :, :] = torch.sin(pos_w * div_term.unsqueeze(0)).transpose(0, 1).unsqueeze(1)
        pe[1:half_dim:2, :, :] = torch.cos(pos_w * div_term.unsqueeze(0)).transpose(0, 1).unsqueeze(1)
        pe[half_dim::2, :, :] = torch.sin(pos_h * div_term.unsqueeze(0)).transpose(0, 1).unsqueeze(2)
        pe[half_dim + 1 :: 2, :, :] = torch.cos(pos_h * div_term.unsqueeze(0)).transpose(0, 1).unsqueeze(2)

        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("Expected a 4D tensor in BCHW format.")
        return x + self.pe[:, :, : x.size(2), : x.size(3)]
