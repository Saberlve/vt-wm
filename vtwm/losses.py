"""Training losses for VT-WM (paper sec 3.2.2): teacher forcing + autoregressive sampling.

L = L_teacher + L_sampling, both L1 over predicted vs. ground-truth encoded latents.
"""
from __future__ import annotations

from typing import Tuple

import torch

from .models.predictor import VTWMPredictor


def teacher_forcing_loss(
    predictor: VTWMPredictor,
    s: torch.Tensor,
    t: torch.Tensor,
    a: torch.Tensor,
) -> torch.Tensor:
    """Next-step prediction from ground-truth context.

    s:(B,T,16,12,20) t:(B,T,4,196,768) a:(B,T,5,7). Position k predicts k+1, so we
    compare predictions at 0..T-2 against encoded latents at 1..T-1.
    """
    s_hat, t_hat = predictor(s, t, a)
    # Targets are stop-gradient: when the tactile encoder is trainable this prevents the
    # latent targets from collapsing (encoder is trained only via the predictor input path).
    s_loss = (s_hat[:, :-1] - s[:, 1:].detach()).abs().mean()
    t_loss = (t_hat[:, :-1] - t[:, 1:].detach()).abs().mean()
    return s_loss + t_loss


def sampling_loss(
    predictor: VTWMPredictor,
    s: torch.Tensor,
    t: torch.Tensor,
    a: torch.Tensor,
    horizon: int,
    max_context: int = 9,
) -> torch.Tensor:
    """Autoregressive sampling loss. Sample H future states without gradients, then
    predict from the (detached) sampled context and supervise against GT latents.

    We start from the first (T - H) ground-truth frames and imagine the last H frames.
    """
    B, T = s.shape[:2]
    H = min(horizon, T - 1)
    ctx = T - H
    # Sampled rollout without gradients (paper: sampled states generated without grad).
    sampled_s, sampled_t = predictor.rollout(
        s[:, :ctx], t[:, :ctx], a, horizon=H, max_context=max_context
    )  # (B,H,...), no grad inside

    # Recompute a gradient-carrying prediction conditioned on the detached sampled context.
    s_in = torch.cat([s[:, :ctx], sampled_s.detach()], dim=1)
    t_in = torch.cat([t[:, :ctx], sampled_t.detach()], dim=1)
    s_hat, t_hat = predictor(s_in, t_in, a[:, :T])
    s_loss = (s_hat[:, ctx - 1 : T - 1] - s[:, ctx:T].detach()).abs().mean()
    t_loss = (t_hat[:, ctx - 1 : T - 1] - t[:, ctx:T].detach()).abs().mean()
    return s_loss + t_loss


def vtwm_loss(
    predictor: VTWMPredictor,
    s: torch.Tensor,
    t: torch.Tensor,
    a: torch.Tensor,
    sampling_horizon: int,
    max_context: int = 9,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    l_teacher = teacher_forcing_loss(predictor, s, t, a)
    l_sampling = sampling_loss(predictor, s, t, a, sampling_horizon, max_context)
    return l_teacher + l_sampling, l_teacher, l_sampling
