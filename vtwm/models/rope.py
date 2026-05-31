"""1D Rotary Position Embeddings (RoPE), Su et al. 2023.

Applied to query/key tensors inside attention. The paper uses RoPE in all attention
layers for relative position encoding (sec 3.2.1). We apply it along whichever axis is
the current attention sequence (spatial positions for spatial attn, timesteps for
temporal attn).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class Rope1D(nn.Module):
    def __init__(self, head_dim: int, max_len: int, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE needs an even head_dim"
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        pos = torch.arange(max_len).float()
        freqs = torch.outer(pos, inv_freq)  # (max_len, head_dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (max_len, head_dim)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, num_heads, L, head_dim) -> rotated by positions 0..L-1."""
        L = x.shape[-2]
        cos = self.cos[:L].to(x.dtype)[None, None]
        sin = self.sin[:L].to(x.dtype)[None, None]
        return x * cos + rotate_half(x) * sin
