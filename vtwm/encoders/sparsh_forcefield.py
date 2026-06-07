"""Sparsh (DINO) force-field decoder wrapper for interpretable tactile visualization.

The Sparsh force-field decoder (facebook/sparsh-gelsight-forcefield-decoder) is a
DPT-style head that turns Sparsh-DINO features into a physical readout:
  - normal: (1,H,W) contact-pressure / depth map
  - shear : (2,H,W) tangential force flow field

It consumes the encoder's intermediate block activations at hooks [2,5,8,11] (NOT just
the final patch tokens). Two decode modes:

  decode_from_images(imgs):  run the real encoder, grab the 4 hook activations, decode.
      This is the faithful "what the gel actually felt" readout for GT tactile frames.

  decode_from_latent(lat):   the world-model predictor only produces the FINAL-layer
      latent, with no intermediate activations. We feed that single latent to all 4
      hook slots ("degenerate" decode). This is an APPROXIMATION (proper-vs-degenerate
      normal-field correlation ~0.65 on GelSight): it recovers the dominant contact
      region but is noisier. Use for qualitative imagined-vs-GT comparison only, never
      as a metric.

NOTE: importing the Sparsh downstream_task package triggers slip/pose/grasp modules that
need xformers/sklearn (unused here). We stub xformers and load forcefield_sl directly,
bypassing the package __init__.
"""
from __future__ import annotations

import importlib
import os
import sys
import types

import torch
import torch.nn as nn

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                     "third_party", "sparsh-orig"))
HOOKS = (2, 5, 8, 11)


def _import_forcefield_decoder_cls():
    os.environ.setdefault("XFORMERS_DISABLED", "1")
    if _REPO not in sys.path:
        sys.path.insert(0, _REPO)
    # Stub xformers (dinov2 imports `from xformers.ops import fmha` unconditionally).
    if "xformers" not in sys.modules:
        xf = types.ModuleType("xformers")
        ops = types.ModuleType("xformers.ops")
        ops.fmha = types.ModuleType("xformers.ops.fmha")
        xf.ops = ops
        sys.modules.update({"xformers": xf, "xformers.ops": ops,
                            "xformers.ops.fmha": ops.fmha})
    import tactile_ssl  # noqa: F401
    pkg = "tactile_ssl.downstream_task"
    if pkg not in sys.modules:  # skip package __init__ (pulls sklearn-dependent modules)
        m = types.ModuleType(pkg)
        m.__path__ = [os.path.join(_REPO, "tactile_ssl", "downstream_task")]
        m.__package__ = pkg
        sys.modules[pkg] = m
    return importlib.import_module(f"{pkg}.forcefield_sl").ForceFieldDecoder


class SparshForceFieldDecoder(nn.Module):
    def __init__(self, ckpt_path: str, encoder: nn.Module, device: str = "cuda",
                 embed_dim: str = "base", hooks=HOOKS):
        super().__init__()
        self.device = device
        self.encoder = encoder           # the vit_base from SparshGelSightEncoder
        self.hooks = tuple(hooks)
        self._acts: dict[str, torch.Tensor] = {}

        decoder_cls = _import_forcefield_decoder_cls()
        decoder = decoder_cls(embed_dim=embed_dim)
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        sd = {k[len("model_task."):]: v for k, v in sd.items()
              if k.startswith("model_task.")}
        missing, unexpected = decoder.load_state_dict(sd, strict=False)
        assert not missing, f"force-field decoder missing keys: {missing[:8]}"
        decoder.to(device).eval()
        for p in decoder.parameters():
            p.requires_grad_(False)
        self.decoder = decoder

        # register forward hooks once on the encoder blocks
        def mk(name):
            def hook(_m, _i, out):
                self._acts[name] = out
            return hook
        for h in self.hooks:
            self.encoder.blocks[h].register_forward_hook(mk(f"t{h}"))

    @torch.no_grad()
    def decode_from_images(self, imgs: torch.Tensor):
        """imgs: (N,6,224,224) GT tactile -> normal (N,1,H,W), shear (N,2,H,W). Faithful."""
        self._acts.clear()
        _ = self.encoder.forward_features(imgs.to(self.device))
        hook_map = {f"t{h}": self._acts[f"t{h}"].clone() for h in self.hooks}
        out = self.decoder(hook_map, mode="normal_shear")
        return out["normal"], out["shear"]

    @torch.no_grad()
    def decode_from_latent(self, lat: torch.Tensor):
        """lat: (N,196,768) final-layer latent -> normal, shear. Degenerate/approximate."""
        lat = lat.to(self.device)
        hook_map = {f"t{h}": lat for h in self.hooks}
        out = self.decoder(hook_map, mode="normal_shear")
        return out["normal"], out["shear"]
