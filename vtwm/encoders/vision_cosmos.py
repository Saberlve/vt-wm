"""Frozen NVIDIA Cosmos image tokenizer (CI16x16-360p) as the VT-WM vision encoder.

The encoder.jit takes an RGB image (B, 3, 192, 320) in [-1, 1] and returns a
continuous latent of shape (B, 16, 12, 20): 16 channels over a 12x20 spatial grid,
i.e. 240 spatial tokens of dim 16 per frame (paper sec 3.2.1).
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Cosmos latent geometry for CI16x16 at 192x320 input.
VISION_LATENT_CH = 16
VISION_LATENT_HW = (12, 20)
VISION_NUM_TOKENS = VISION_LATENT_HW[0] * VISION_LATENT_HW[1]  # 240


class CosmosVisionEncoder(nn.Module):
    """Wraps the frozen Cosmos encoder.jit. All params frozen; runs under no_grad."""

    def __init__(self, encoder_jit_path: str, device: str = "cuda", dtype: torch.dtype = torch.float32,
                 decoder_jit_path: str | None = None):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.encoder = torch.jit.load(encoder_jit_path, map_location=device)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        # Optional decoder (for visualizing rollouts): latent (B,16,12,20) -> image.
        self.decoder = None
        if decoder_jit_path is None:
            import os
            cand = os.path.join(os.path.dirname(encoder_jit_path), "decoder.jit")
            decoder_jit_path = cand if os.path.exists(cand) else None
        if decoder_jit_path is not None:
            self.decoder = torch.jit.load(decoder_jit_path, map_location=device)
            self.decoder.eval()
            for p in self.decoder.parameters():
                p.requires_grad_(False)

    @staticmethod
    def preprocess(frames: torch.Tensor) -> torch.Tensor:
        """Map RGB in [0, 1] to the [-1, 1] range expected by the tokenizer.

        If the caller already provides values in [-1, 1] (e.g. fake data), this is a
        harmless monotonic rescale for the smoke test.
        """
        return frames * 2.0 - 1.0

    @torch.no_grad()
    def encode(self, frames: torch.Tensor, do_preprocess: bool = True) -> torch.Tensor:
        """frames: (B, T, 3, 192, 320) -> latents (B, T, 16, 12, 20)."""
        assert frames.dim() == 5, f"expected (B,T,3,H,W), got {tuple(frames.shape)}"
        b, t = frames.shape[:2]
        x = frames.reshape(b * t, *frames.shape[2:]).to(self.device, self.dtype)
        if do_preprocess:
            x = self.preprocess(x)
        out = self.encoder(x)
        latent = out[0] if isinstance(out, (tuple, list)) else out
        # The traced tokenizer runs in bf16; cast back to the wrapper dtype for the predictor.
        latent = latent.to(self.dtype)
        return latent.reshape(b, t, *latent.shape[1:])

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """latents: (B, T, 16, 12, 20) -> RGB images (B, T, 3, 192, 320) in [0, 1]."""
        assert self.decoder is not None, "decoder.jit not loaded"
        assert latents.dim() == 5, f"expected (B,T,16,12,20), got {tuple(latents.shape)}"
        b, t = latents.shape[:2]
        # The traced decoder runs in bf16; match its weight dtype.
        z = latents.reshape(b * t, *latents.shape[2:]).to(self.device, torch.bfloat16)
        out = self.decoder(z)
        img = out[0] if isinstance(out, (tuple, list)) else out
        img = ((img.float() + 1.0) / 2.0).clamp(0.0, 1.0)  # [-1,1] -> [0,1]
        return img.reshape(b, t, *img.shape[1:])
