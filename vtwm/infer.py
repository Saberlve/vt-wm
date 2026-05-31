"""Inference demo: autoregressive rollout + one CEM plan on fake context."""
from __future__ import annotations

import argparse
import os

import torch
from omegaconf import OmegaConf

from vtwm.build import build_dataset, build_tactile_encoder, build_vision_encoder
from vtwm.models.predictor import VTWMPredictor
from vtwm.planning.cem import cem_plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", default=None, help="predictor checkpoint; if omitted uses random init")
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config)
    device = cfg.device if torch.cuda.is_available() else "cpu"

    vision = build_vision_encoder(cfg, device)
    tactile = build_tactile_encoder(cfg, device)
    predictor = VTWMPredictor(
        dim=cfg.model.dim, depth=cfg.model.depth, num_heads=cfg.model.num_heads,
        mlp_ratio=cfg.model.mlp_ratio, num_sensors=cfg.data.num_sensors,
        action_dim=cfg.data.action_dim, action_chunk=cfg.data.action_chunk,
        max_temporal=cfg.model.max_temporal, tactile_dim=cfg.model.get("tactile_dim", 768),
    ).to(device)
    if args.ckpt and os.path.exists(args.ckpt):
        sd = torch.load(args.ckpt, map_location=device)
        predictor.load_state_dict(sd["model"])
        if sd.get("tactile") is not None and hasattr(tactile, "model"):
            tactile.model.load_state_dict(sd["tactile"])
            print("[loaded] fine-tuned tactile encoder weights")
        print(f"[loaded] {args.ckpt}")
    predictor.eval()

    ds = build_dataset(cfg)
    item = ds[0]
    rgb = item["rgb"].unsqueeze(0).to(device)
    tac = item["tactile"].unsqueeze(0).to(device)

    with torch.no_grad():
        s = vision.encode(rgb)   # (1,T,16,12,20)
        t = tactile.encode(tac)  # (1,T,4,196,768)

    # --- 1) Autoregressive rollout demo ---
    H = cfg.planning.horizon_s * cfg.planning.freq
    Tc = 1  # start from a single context frame
    actions = 0.05 * torch.randn(1, Tc - 1 + H, cfg.data.action_chunk, cfg.data.action_dim, device=device)
    roll_s, roll_t = predictor.rollout(s[:, :Tc], t[:, :Tc], actions, horizon=H, max_context=cfg.planning.max_context)
    print(f"[rollout] vision future {tuple(roll_s.shape)}  tactile future {tuple(roll_t.shape)}")

    # --- 2) CEM planning demo (goal = last encoded frame's latent) ---
    s_goal = s[:, -1]
    best_action, cost_history = cem_plan(
        predictor, s[:, :Tc], t[:, :Tc], s_goal,
        horizon=H, action_chunk=cfg.data.action_chunk, action_dim=cfg.data.action_dim,
        particles=cfg.planning.particles, iters=cfg.planning.iters, elites=cfg.planning.elites,
        max_context=cfg.planning.max_context, device=device,
    )
    print(f"[cem] best_action {tuple(best_action.shape)}")
    print(f"[cem] cost history (best-so-far): {[round(c, 4) for c in cost_history]}")


if __name__ == "__main__":
    main()
