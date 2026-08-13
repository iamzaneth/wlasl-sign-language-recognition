"""Transformer encoder-only classifier used by the landmark experiments."""
from __future__ import annotations

import torch
from torch import nn


class KeypointTransformerEncoderOnly(nn.Module):
    """CLS-token Transformer classifier for fixed-length landmark sequences."""

    def __init__(self, input_dim: int, num_classes: int, seq_len: int, d_model: int, num_heads: int, num_layers: int, dim_feedforward: int, dropout: float) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.seq_len = seq_len
        self.input_proj = nn.Linear(input_dim, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len + 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor, frame_mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.shape[1] != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {x.shape[1]}")
        x = self.input_proj(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.pos_embedding[:, : x.shape[1]]
        padding_mask = None
        if frame_mask is not None:
            padding_mask = torch.cat((torch.zeros((x.shape[0], 1), device=x.device, dtype=torch.bool), ~frame_mask.bool()), dim=1)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        return self.classifier(self.dropout(self.norm(x[:, 0])))
