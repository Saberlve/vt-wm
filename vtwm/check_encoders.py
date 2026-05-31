"""Standalone encoder shape/load checks (used by the smoke test)."""
from __future__ import annotations

import argparse

import torch
from omegaconf import OmegaConf

from vtwm.encoders.tactile_sparshx import SparshXTactileEncoder
from vtwm.encoders.vision_cosmos import CosmosVisionEncoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config)
    device = cfg.device if torch.cuda.is_available() else "cpu"

    vision = CosmosVisionEncoder(cfg.paths.cosmos_encoder_jit, device=device)
    rgb = torch.rand(2, cfg.data.T, 3, *cfg.data.rgb_hw)
    s = vision.encode(rgb)
    assert tuple(s.shape) == (2, cfg.data.T, 16, 12, 20), s.shape
    print(f"[cosmos] {tuple(rgb.shape)} -> {tuple(s.shape)}  OK")

    tactile = SparshXTactileEncoder(cfg.paths.sparshx_ckpt, device=device)
    tac = torch.rand(2, cfg.data.T, cfg.data.num_sensors, 6, *cfg.data.tactile_hw)
    t = tactile.encode(tac)
    assert tuple(t.shape) == (2, cfg.data.T, cfg.data.num_sensors, 196, 768), t.shape
    print(f"[sparsh-x] {tuple(tac.shape)} -> {tuple(t.shape)}  OK (strict load passed)")


if __name__ == "__main__":
    main()
