"""Frozen Sparsh (DINO ViT-base) GelSight tactile encoder for the ManiFeel pipeline.

ManiFeel uses a single GelSight (Taxim-rendered) tactile camera. We encode it with the
original Sparsh DINO backbone (facebook/sparsh-dino-base), which is the in-domain choice
for GelSight/DIGIT vision-based tactile images (Sparsh-X is Digit-360-only).

Input recipe (matching Sparsh's GelSight config): two stride-5 frames, each
background-subtracted (compute_diff, offset 0.5), concatenated on channels -> 6 channels,
224x224. Output: 196 patch tokens x 768 per sensor.

NOTE: this wrapper puts `third_party/sparsh-orig` on sys.path. Its `tactile_ssl` package
collides by name with the Sparsh-X repo's, so do not import both encoders in one process.
"""
from __future__ import annotations

import contextlib
import os
import sys

import torch
import torch.nn as nn

os.environ.setdefault("XFORMERS_DISABLED", "1")

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "sparsh-orig"))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

TACTILE_EMBED_DIM = 768
TACTILE_NUM_TOKENS = 196


def _build_vit_base() -> nn.Module:
    from tactile_ssl.model import vit_base

    # in_chans=6: two RGB frames concatenated (Sparsh GelSight "concat_ch_img", num_frames=2).
    # pos_embed_fn="sinusoidal": checkpoint stores pos_embed.frequency_bands (not a learned table).
    return vit_base(
        patch_size=16, num_register_tokens=1, img_size=224, in_chans=6, num_frames=1,
        pos_embed_fn="sinusoidal",
    )


def _load_weights(model: nn.Module, ckpt_path: str) -> None:
    if ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        sd = load_file(ckpt_path)
    else:
        obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = obj.get("state_dict", obj) if isinstance(obj, dict) else obj

    model_keys = set(model.state_dict().keys())
    # Strip common prefixes until keys line up with the model.
    for prefix in ("", "encoder.", "module.encoder.", "module.", "student.backbone.", "backbone.", "teacher.backbone."):
        cand = {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in sd.items()}
        if len(model_keys & set(cand.keys())) >= 0.8 * len(model_keys):
            sd = cand
            break
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [m for m in missing if "mask_token" not in m]  # mask_token unused at inference
    assert not missing, f"missing keys when loading Sparsh-DINO: {missing[:8]} ..."


class SparshGelSightEncoder(nn.Module):
    def __init__(self, ckpt_path: str, device: str = "cuda", dtype: torch.dtype = torch.float32,
                 freeze: bool = True):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.trainable = not freeze
        model = _build_vit_base()
        _load_weights(model, ckpt_path)
        model.to(device=device, dtype=dtype)
        model.train(self.trainable)
        for p in model.parameters():
            p.requires_grad_(self.trainable)
        self.model = model

    def encode(self, tactile: torch.Tensor) -> torch.Tensor:
        """tactile: (B, T, S, 6, 224, 224) -> tokens (B, T, S, 196, 768).

        Runs with gradients when the encoder is trainable (fine-tuning), else no_grad.
        """
        assert tactile.dim() == 6, f"expected (B,T,S,6,H,W), got {tuple(tactile.shape)}"
        b, t, s = tactile.shape[:3]
        x = tactile.reshape(b * t * s, *tactile.shape[3:]).to(self.device, self.dtype)
        ctx = contextlib.nullcontext() if self.trainable else torch.no_grad()
        with ctx:
            tokens = self.model.forward_features(x)["x_norm_patchtokens"]  # (B*T*S, 196, 768)
        return tokens.reshape(b, t, s, tokens.shape[-2], tokens.shape[-1])
