"""Frozen Sparsh-X (img-only, base) tactile encoder for Digit 360.

Input per sensor is a 6-channel image: two frames (stride 5) concatenated on the
channel dim, cropped/resized to 224x224. The encoder returns 196 patch tokens of
dim 768 per sensor (14x14 patches). With 4 Digit 360 sensors this gives the
"compact" tactile state t_k (paper sec 3.2.1). All params frozen; runs under no_grad.

We instantiate the architecture directly (dit_base, fusion_type='vanilla') matching
the d360_sparshx_img_base.pth checkpoint, avoiding hydra global-state and the
side-effect file write inside build_encoder.
"""
from __future__ import annotations

import os
import sys
from functools import partial

import torch
import torch.nn as nn
from omegaconf import OmegaConf

# Make sure xformers' optional fast paths don't get imported (CPU/edge GPUs).
os.environ.setdefault("XFORMERS_DISABLED", "1")

_REPO = os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "sparsh-multisensory-touch")
_REPO = os.path.abspath(_REPO)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

TACTILE_EMBED_DIM = 768
TACTILE_NUM_TOKENS = 196  # 224/16 = 14 -> 14*14
TACTILE_NUM_SENSORS = 4

# 3-channel img normalization from config/encoder/digit360_sparshx.yaml (repeated to 6 ch).
_IMG_NORM = {
    "img": {
        "avg": [0.09562227, 0.26593734, 0.32464101],
        "std": [0.08179263, 0.17255498, 0.18509875],
        "div": 1,
    }
}


def _build_dit_base() -> nn.Module:
    from tactile_ssl.model.d360_transformer import dit_base
    from tactile_ssl.model.layers import Attention

    model = dit_base(
        use_img=True,
        use_mic=False,
        use_imu=False,
        use_pressure=False,
        sensor_sizes={"img": [224, 224]},
        sensor_chans={"img": 6},
        patch_sizes={"img": 16},
        num_register_tokens=1,
        fusion_type="vanilla",
        fusion_layer=8,
        num_bottlenecks=4,
        drop_path_rate=0.0,
        pos_embed_fn="sinusoidal",
        normalization=OmegaConf.create(_IMG_NORM),
        attn_class=Attention,
    )
    return model


class SparshXTactileEncoder(nn.Module):
    def __init__(self, ckpt_path: str, device: str = "cuda", dtype: torch.dtype = torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype
        model = _build_dit_base()
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(sd, strict=True)
        assert not missing, f"missing keys: {missing}"
        assert not unexpected, f"unexpected keys: {unexpected}"
        model.eval().to(device=device, dtype=dtype)
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model

    @torch.no_grad()
    def encode(self, tactile: torch.Tensor) -> torch.Tensor:
        """tactile: (B, T, S, 6, 224, 224) -> tokens (B, T, S, 196, 768).

        S = number of Digit 360 sensors (4). Sensors and time are folded into the
        batch for a single encoder pass.
        """
        assert tactile.dim() == 6, f"expected (B,T,S,6,H,W), got {tuple(tactile.shape)}"
        b, t, s = tactile.shape[:3]
        x = tactile.reshape(b * t * s, *tactile.shape[3:]).to(self.device, self.dtype)
        tokens = self.model({"img": x})["img"]  # (B*T*S, 196, 768)
        return tokens.reshape(b, t, s, tokens.shape[-2], tokens.shape[-1])
