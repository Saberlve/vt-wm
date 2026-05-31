"""Multi-head self/cross attention with optional RoPE, used by the VT-WM predictor."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import Rope1D


class MultiheadSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        rope: Optional[Rope1D] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        # x: (Bf, L, d)
        bf, L, d = x.shape
        qkv = self.qkv(x).reshape(bf, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (Bf, nh, L, hd)
        if rope is not None:
            q, k = rope(q), rope(k)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        out = out.transpose(1, 2).reshape(bf, L, d)
        return self.proj(out)


class MultiheadCrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        # x: (Bf, Lq, d) queries; ctx: (Bf, Lk, d) keys/values
        bf, Lq, d = x.shape
        Lk = ctx.shape[1]
        q = self.q(x).reshape(bf, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv(ctx).reshape(bf, Lk, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(bf, Lq, d)
        return self.proj(out)


class Mlp(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))
