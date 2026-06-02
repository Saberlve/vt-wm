"""VT-WM autoregressive predictor (transition model).

Architecture (paper sec 3.2.1), with depth reduced for the first runnable version:
  - vision + tactile latents projected to a unified width d, concatenated along the
    spatial axis into R(b, t, s, d); positions are encoded only via RoPE in attention.
  - N transformer blocks, each: factorized spatio-temporal self-attention (spatial then
    temporal) on sensory tokens AND on action tokens, followed by action-conditioning
    cross-attention (sensory queries attend to action keys/values), then MLPs.
  - RoPE in all attention; temporal attention is causal so predictions are autoregressive.
  - modality-specific output heads project back to vision (16ch, 12x20) and tactile (768)
    latents, yielding the predicted next-step states (s_{k+1}, t_{k+1}).
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .attention import Mlp, MultiheadCrossAttention, MultiheadSelfAttention
from .projectors import InputProjector, OutputHead
from .rope import Rope1D


def _vision_to_tokens(s: torch.Tensor) -> torch.Tensor:
    # (B,T,C,H,W) -> (B,T,H*W,C)
    b, t, c, h, w = s.shape
    return s.reshape(b, t, c, h * w).transpose(-1, -2)


def _tokens_to_vision(x: torch.Tensor, c: int, h: int, w: int) -> torch.Tensor:
    # (B,T,H*W,C) -> (B,T,C,H,W)
    b, t = x.shape[:2]
    return x.transpose(-1, -2).reshape(b, t, c, h, w)


class PredictorBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        # sensory factorized self-attention
        self.norm_s_sp = nn.LayerNorm(dim)
        self.attn_s_sp = MultiheadSelfAttention(dim, num_heads)
        self.norm_s_tp = nn.LayerNorm(dim)
        self.attn_s_tp = MultiheadSelfAttention(dim, num_heads)
        # action factorized self-attention
        self.norm_a_sp = nn.LayerNorm(dim)
        self.attn_a_sp = MultiheadSelfAttention(dim, num_heads)
        self.norm_a_tp = nn.LayerNorm(dim)
        self.attn_a_tp = MultiheadSelfAttention(dim, num_heads)
        # action-conditioning cross attention (sensory <- action)
        self.norm_x = nn.LayerNorm(dim)
        self.norm_ctx = nn.LayerNorm(dim)
        self.cross = MultiheadCrossAttention(dim, num_heads)
        # MLPs
        self.norm_s_mlp = nn.LayerNorm(dim)
        self.mlp_s = Mlp(dim, mlp_ratio)
        self.norm_a_mlp = nn.LayerNorm(dim)
        self.mlp_a = Mlp(dim, mlp_ratio)

    def forward(
        self,
        s: torch.Tensor,
        a: torch.Tensor,
        rope_spatial: Rope1D,
        rope_temporal: Rope1D,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, t, ls, d = s.shape
        la = a.shape[2]

        # --- spatial self-attention: all tokens within a timestep interact ---
        s = s + self.attn_s_sp(self.norm_s_sp(s).reshape(b * t, ls, d), rope=rope_spatial).reshape(b, t, ls, d)
        a = a + self.attn_a_sp(self.norm_a_sp(a).reshape(b * t, la, d), rope=rope_spatial).reshape(b, t, la, d)

        # --- temporal self-attention: each token evolves across timesteps (causal) ---
        s_t = self.norm_s_tp(s).permute(0, 2, 1, 3).reshape(b * ls, t, d)
        s = s + self.attn_s_tp(s_t, rope=rope_temporal, is_causal=True).reshape(b, ls, t, d).permute(0, 2, 1, 3)
        a_t = self.norm_a_tp(a).permute(0, 2, 1, 3).reshape(b * la, t, d)
        a = a + self.attn_a_tp(a_t, rope=rope_temporal, is_causal=True).reshape(b, la, t, d).permute(0, 2, 1, 3)

        # --- action conditioning via cross-attention (within each timestep) ---
        xq = self.norm_x(s).reshape(b * t, ls, d)
        ctx = self.norm_ctx(a).reshape(b * t, la, d)
        s = s + self.cross(xq, ctx).reshape(b, t, ls, d)

        # --- MLPs ---
        s = s + self.mlp_s(self.norm_s_mlp(s))
        a = a + self.mlp_a(self.norm_a_mlp(a))
        return s, a


class VTWMPredictor(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        vision_ch: int = 16,
        vision_hw: Tuple[int, int] = (12, 20),
        tactile_dim: int = 768,
        tactile_tokens_per_sensor: int = 196,
        num_sensors: int = 4,
        action_dim: int = 7,
        action_chunk: int = 5,
        max_temporal: int = 64,
    ):
        super().__init__()
        self.vision_ch = vision_ch
        self.vision_hw = vision_hw
        self.vision_tokens = vision_hw[0] * vision_hw[1]
        self.tactile_dim = tactile_dim
        self.tactile_tokens_per_sensor = tactile_tokens_per_sensor
        self.num_sensors = num_sensors
        self.tactile_tokens = num_sensors * tactile_tokens_per_sensor
        self.action_chunk = action_chunk

        self.vis_proj = InputProjector(vision_ch, dim)
        self.tac_proj = InputProjector(tactile_dim, dim)
        self.act_proj = InputProjector(action_dim, dim)
        self.vis_head = OutputHead(dim, vision_ch)
        self.tac_head = OutputHead(dim, tactile_dim)

        ls = self.vision_tokens + self.tactile_tokens
        head_dim = dim // num_heads
        self.rope_spatial = Rope1D(head_dim, max(ls, action_chunk))
        self.rope_temporal = Rope1D(head_dim, max_temporal)

        self.blocks = nn.ModuleList([PredictorBlock(dim, num_heads, mlp_ratio) for _ in range(depth)])

    def forward(self, s: torch.Tensor, t: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """s:(B,T,16,12,20)  t:(B,T,4,196,768)  a:(B,T,5,7) ->
        predicted next-step (s_hat, t_hat) of the same shapes (position k predicts k+1)."""
        b, T = s.shape[:2]
        vis = self.vis_proj(_vision_to_tokens(s))  # (B,T,240,d)
        tac = self.tac_proj(t.reshape(b, T, self.tactile_tokens, self.tactile_dim))  # (B,T,784,d)
        sensory = torch.cat([vis, tac], dim=2)  # (B,T,Ls,d)
        act = self.act_proj(a)  # (B,T,5,d)

        for blk in self.blocks:
            sensory, _ = blk(sensory, act, self.rope_spatial, self.rope_temporal)

        vis_out = sensory[:, :, : self.vision_tokens]
        tac_out = sensory[:, :, self.vision_tokens :]
        s_hat = _tokens_to_vision(self.vis_head(vis_out), self.vision_ch, *self.vision_hw)
        t_hat = self.tac_head(tac_out).reshape(b, T, self.num_sensors, self.tactile_tokens_per_sensor, self.tactile_dim)
        return s_hat, t_hat

    @torch.no_grad()
    def rollout(
        self,
        s_ctx: torch.Tensor,
        t_ctx: torch.Tensor,
        actions: torch.Tensor,
        horizon: int,
        max_context: int = 9,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Autoregressively imagine `horizon` future latents.

        s_ctx:(B,Tc,16,12,20)  t_ctx:(B,Tc,4,196,768)
        actions:(B,La,5,7) with La >= Tc+horizon-1; action at absolute frame k drives k->k+1.
        Returns predicted future (B,horizon,...) for vision and tactile.
        """
        s_frames = [s_ctx[:, i] for i in range(s_ctx.shape[1])]
        t_frames = [t_ctx[:, i] for i in range(t_ctx.shape[1])]
        out_s, out_t = [], []
        for _ in range(horizon):
            cur_len = len(s_frames)
            win = min(max_context, cur_len)
            s_win = torch.stack(s_frames[-win:], dim=1)
            t_win = torch.stack(t_frames[-win:], dim=1)
            a_win = actions[:, cur_len - win : cur_len]
            ps, pt = self.forward(s_win, t_win, a_win)
            next_s, next_t = ps[:, -1], pt[:, -1]
            s_frames.append(next_s)
            t_frames.append(next_t)
            out_s.append(next_s)
            out_t.append(next_t)
        return torch.stack(out_s, dim=1), torch.stack(out_t, dim=1)
