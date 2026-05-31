"""Single nn.Module wrapping the trainable parts (tactile encoder + predictor) so it can
be DDP-wrapped correctly. Vision (Cosmos, frozen) is encoded outside and passed in.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from vtwm.losses import vtwm_loss


class VTWMTrainModule(nn.Module):
    def __init__(self, tactile, predictor, sampling_horizon: int, max_context: int):
        super().__init__()
        self.tactile = tactile        # SparshGelSightEncoder / SparshXTactileEncoder (nn.Module)
        self.predictor = predictor
        self.sampling_horizon = sampling_horizon
        self.max_context = max_context

    def forward(self, s: torch.Tensor, tactile_raw: torch.Tensor, action: torch.Tensor):
        """s: vision latents (B,T,16,12,20) already encoded (no_grad). Returns (loss, teacher, sampling)."""
        t = self.tactile.encode(tactile_raw)  # grad iff tactile encoder is trainable
        return vtwm_loss(
            self.predictor, s, t, action,
            sampling_horizon=self.sampling_horizon, max_context=self.max_context,
        )
