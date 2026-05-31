"""Sinusoidal positional embeddings added to projected tokens.

Paper sec 3.2.1: vision/tactile latents are "augmented with sinusoidal positional
embeddings, and projected into a unified representation R(b,t,s,d)". We add a spatial
PE (over the concatenated vision+tactile token axis) and a temporal PE (over timesteps).
"""
from __future__ import annotations

import math

import torch


def sinusoidal_embedding(num_positions: int, dim: int) -> torch.Tensor:
    """Standard transformer sinusoidal table: (num_positions, dim)."""
    pe = torch.zeros(num_positions, dim)
    position = torch.arange(0, num_positions, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe
