"""Modality-specific input projectors and output heads for the VT-WM predictor."""
from __future__ import annotations

import torch
import torch.nn as nn


class InputProjector(nn.Module):
    """Linear projection from a modality's native token dim into the unified width d."""

    def __init__(self, in_dim: int, dim: int):
        super().__init__()
        # self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class OutputHead(nn.Module):
    """Projects unified-width tokens back to a modality's native latent dim (LN + Linear)."""

    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(x))
