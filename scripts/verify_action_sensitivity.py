"""Diagnose whether the VT-WM is actually action-conditioned (controllable) or just a good
action-agnostic video predictor. If perturbing/zeroing actions barely changes the predicted
vision latent — relative to how accurately the model predicts and to the goal distance CEM
must optimize over — then the planner has no gradient and wanders regardless of interp/warm-start.
"""
import argparse, os
import numpy as np
import torch
from omegaconf import OmegaConf

from vtwm.build import build_dataset, build_tactile_encoder, build_vision_encoder
from vtwm.models.predictor import VTWMPredictor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/univtac_lift_bottle_ada.yaml")
    ap.add_argument("--ckpt", default="runs/univtac_lift_bottle/predictor.pt")
    ap.add_argument("--ctx", type=int, default=1)
    ap.add_argument("--windows", type=int, default=8)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vision = build_vision_encoder(cfg, device)
    tactile = build_tactile_encoder(cfg, device)
    P = VTWMPredictor(
        dim=cfg.model.dim, depth=cfg.model.depth, num_heads=cfg.model.num_heads,
        mlp_ratio=cfg.model.mlp_ratio, num_sensors=cfg.data.num_sensors,
        action_dim=cfg.data.action_dim, action_chunk=cfg.data.action_chunk,
        max_temporal=cfg.model.max_temporal, tactile_dim=cfg.model.get("tactile_dim", 768),
    ).to(device)
    sd = torch.load(args.ckpt, map_location=device)
    P.load_state_dict(sd["model"])
    if sd.get("tactile") is not None and hasattr(tactile, "model"):
        tactile.model.load_state_dict(sd["tactile"])
    P.eval()
    if hasattr(tactile, "model"):
        tactile.model.eval()

    cfg.data.T = max(9, int(cfg.data.T))
    ds = build_dataset(cfg, val=True)
    ctx, T = args.ctx, cfg.data.T
    horizon = T - ctx
    max_ctx = int(cfg.planning.max_context)
    rng = np.random.default_rng(0)

    # pick motion-rich windows
    cand = rng.choice(len(ds), size=min(len(ds), 120), replace=False)
    mot = sorted(((float((ds[int(i)]["rgb"][ctx:] - ds[int(i)]["rgb"][ctx-1:ctx]).abs().mean()), int(i))
                  for i in cand), reverse=True)
    idxs = [i for _, i in mot[:args.windows]]

    agg = {k: [] for k in ["d_zero", "d_pert10", "d_big", "pred_err", "goal_dist", "gt_motion"]}
    for di in idxs:
        it = ds[int(di)]
        rgb = it["rgb"].unsqueeze(0).to(device)
        tac = it["tactile"].unsqueeze(0).to(device)
        act = it["action"].unsqueeze(0).to(device)          # (1,T,1,A) raw absolute qpos
        with torch.no_grad():
            s = vision.encode(rgb)
            t = tactile.encode(tac)
            s_ctx, t_ctx = s[:, :ctx], t[:, :ctx]

            def roll(a):
                return P.rollout(s_ctx, t_ctx, a, horizon=horizon, max_context=max_ctx)[0][:, -1]

            f_gt = roll(act)
            f_zero = roll(torch.zeros_like(act))
            f_p10 = roll(act + 0.1 * torch.randn_like(act))     # CEM qpos_sigma scale
            f_big = roll(act + 1.0 * torch.randn_like(act))     # huge perturbation
            gt_final = s[:, -1]

        def l2(a, b): return float((a - b).flatten(1).pow(2).sum(1).sqrt().mean())
        agg["d_zero"].append(l2(f_gt, f_zero))         # effect of removing action entirely
        agg["d_pert10"].append(l2(f_gt, f_p10))        # effect of CEM-scale action change
        agg["d_big"].append(l2(f_gt, f_big))           # effect of huge action change
        agg["pred_err"].append(l2(f_gt, gt_final))     # how wrong the prediction is
        agg["goal_dist"].append(l2(gt_final, s_ctx[:, -1]))  # latent dist CEM optimizes over
        agg["gt_motion"].append(l2(gt_final, s[:, ctx-1]))

    m = {k: float(np.mean(v)) for k, v in agg.items()}
    print("\n=== VT-WM action-sensitivity (final visual-latent L2) ===")
    print(f"  Δ(GT action -> ZERO action)      : {m['d_zero']:.3f}   <- action's TOTAL effect")
    print(f"  Δ(GT action -> +N(0,0.1) pert)   : {m['d_pert10']:.3f}   <- effect at CEM sampling scale")
    print(f"  Δ(GT action -> +N(0,1.0) pert)   : {m['d_big']:.3f}   <- effect at huge action change")
    print(f"  prediction error (GT-act vs GT)  : {m['pred_err']:.3f}   <- model accuracy")
    print(f"  goal/context latent distance     : {m['goal_dist']:.3f}   <- what CEM must reduce")
    print(f"  gt motion (ctx-1 -> final)       : {m['gt_motion']:.3f}")
    print("\n  Interpretable ratios:")
    print(f"  action effect / prediction error : {m['d_zero']/max(m['pred_err'],1e-6):.2f}  "
          f"(<<1 => action ignored vs noise floor)")
    print(f"  CEM-scale effect / goal distance : {m['d_pert10']/max(m['goal_dist'],1e-6):.3f}  "
          f"(~0 => flat cost, CEM has no signal)")


if __name__ == "__main__":
    main()
