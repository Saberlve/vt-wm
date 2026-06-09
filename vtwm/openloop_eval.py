"""Open-loop CEM evaluation of a trained VT-WM against the UniVTAC dataset.

Unlike the closed-loop Isaac-Sim deploy (policy/VTWM/deploy_policy.py) and the imagination
benchmark (vtwm/eval.py, which rolls out under the *ground-truth* actions), this script asks
the question the closed-loop wandering really hinges on:

    "Given a real demonstration window, can the world model's CEM planner recover the
     demonstrated actions, and does imagining under its own sampled actions actually drive
     the predicted visual latent toward the goal?"

For each selected window it reproduces the EXACT training/deploy cold-start CEM regime
(single-frame context, goal = the EPISODE's final visual latent (deploy frame: -1), mu_init =
the single current pose broadcast over the horizon, sigma_init = the qpos_sigma ramp) and
writes, per window:

  - actions.npz / action plot : the demonstrated GT action chunks vs the CEM-sampled chunks,
                                plus the per-window action MSE (the headline open-loop metric);
  - panel.png                 : [ current frame | goal frame | imagined-final (CEM) ], i.e. the
                                real current image, the real goal image, and the Cosmos-decoded
                                final predicted latent reached by imagining under the CEM actions;
  - rollout.mp4               : per horizon step, [ GT future | Cosmos recon (ceiling) |
                                imagined (CEM actions) | imagined (GT actions) ].

A bad open-loop result (high action MSE / the CEM-imagined panel not converging to the goal)
localizes the closed-loop wandering to the planner/world-model itself rather than the sim
domain gap.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf

from vtwm.build import build_dataset, build_tactile_encoder, build_vision_encoder
from vtwm.models.predictor import VTWMPredictor
from vtwm.planning.cem import cem_plan, qpos_sigma_ramp


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = (a - b).pow(2).mean().item()
    return 99.0 if mse <= 1e-12 else 10.0 * math.log10(1.0 / mse)


def to_uint8(img_chw: torch.Tensor) -> np.ndarray:
    return (img_chw.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def _label(img: np.ndarray, text: str, scale: float = 0.5, color=(255, 255, 255)) -> np.ndarray:
    """Draw a caption with a dark background strip at the top-left of an RGB image."""
    img = img.copy()
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.rectangle(img, (0, 0), (min(img.shape[1], tw + 8), th + bl + 6), (0, 0, 0), -1)
    cv2.putText(img, text, (4, th + 3), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
    return img


def _title_bar(width: int, up: int, text: str, color=(0, 255, 0)) -> np.ndarray:
    bar = np.zeros((26 * up, width, 3), np.uint8)
    cv2.putText(bar, text, (6, 18 * up), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * up, color, 1, cv2.LINE_AA)
    return bar


def _action_plot(gt_act: np.ndarray, cem_act: np.ndarray, out_path: str, title: str):
    """Plot the flattened raw command sequence (H*chunk steps) per action dim: GT vs CEM.

    gt_act / cem_act: (H, chunk, dim). The dataset action is an ACT-style chunk of consecutive
    raw qpos commands, so flattening (H, chunk) -> H*chunk recovers the actual command stream
    the model was conditioned on; we overlay the demonstrated and the CEM-sampled streams.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    H, chunk, dim = gt_act.shape
    gt_flat = gt_act.reshape(H * chunk, dim)
    cem_flat = cem_act.reshape(H * chunk, dim)
    x = np.arange(H * chunk)

    ncol = min(4, dim)
    nrow = int(math.ceil(dim / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 2.4 * nrow), squeeze=False)
    for d in range(dim):
        ax = axes[d // ncol][d % ncol]
        ax.plot(x, gt_flat[:, d], "-o", ms=2.5, lw=1.4, label="GT", color="tab:green")
        ax.plot(x, cem_flat[:, d], "-x", ms=3.5, lw=1.4, label="CEM", color="tab:red")
        # keyframe boundaries (every `chunk` raw steps == one ~6Hz planned step)
        for kf in range(1, H):
            ax.axvline(kf * chunk - 0.5, color="0.85", lw=0.6, zorder=0)
        dmse = float(np.mean((gt_flat[:, d] - cem_flat[:, d]) ** 2))
        ax.set_title(f"dim {d}  mse={dmse:.4f}", fontsize=9)
        ax.tick_params(labelsize=7)
        if d == 0:
            ax.legend(fontsize=8, loc="best")
    for d in range(dim, nrow * ncol):
        axes[d // ncol][d % ncol].axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _cost_plot(gt_cost: np.ndarray, cem_cost: np.ndarray, static_cost: float,
               out_path: str, title: str):
    """Plot the CEM planning cost (||imagined latent - goal latent||_2) per rollout step,
    for the GT-action rollout vs the CEM-action rollout.

    This is exactly the objective cem_plan minimizes (over the final step). If the two curves
    nearly overlap — and both sit near the `static` (no-move) baseline — the world model's
    imagined latent barely depends on the action, so CEM has no signal to recover the true
    action (the mechanism behind closed-loop wandering).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    H = len(gt_cost)
    x = np.arange(1, H + 1)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(x, gt_cost, "-o", ms=4, lw=1.6, color="tab:green", label="GT action")
    ax.plot(x, cem_cost, "-x", ms=5, lw=1.6, color="tab:red", label="CEM action")
    ax.axhline(static_cost, ls="--", lw=1.2, color="0.5",
               label=f"static (no-move) = {static_cost:.2f}")
    ax.set_xlabel("rollout step k")
    ax.set_ylabel(r"cost = $\|$imagined latent$_k$ - goal$\|_2$")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/univtac_lift_bottle_ada.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", default="./eval_out_openloop")
    ap.add_argument("--split", choices=["train", "val"], default="train",
                    help="which dataset split to draw demonstration windows from (default: train)")
    ap.add_argument("--num_windows", type=int, default=6, help="windows to evaluate")
    ap.add_argument("--ctx", type=int, default=1,
                    help="context frames before planning (1 = single-frame cold start, matches train/deploy)")
    ap.add_argument("--horizon", type=int, default=4,
                    help="planned horizon in ~6Hz steps; 0 = use (window_T - ctx)")
    ap.add_argument("--eval_T", type=int, default=0,
                    help="window length; 0 = cfg.data.T. Larger gives a longer plan horizon / GT-action span (goal is always the episode final frame).")
    ap.add_argument("--mu_init", choices=["pose", "zero"], default="pose",
                    help="CEM mean seed: the single current pose broadcast over the horizon "
                         "(abs qpos prior, matches train/deploy cold start) or zeros")
    ap.add_argument("--scan", type=int, default=160, help="candidate windows scanned for RGB motion")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=int, default=3)
    ap.add_argument("--upscale", type=int, default=2)
    # CEM overrides (default to cfg.planning)
    ap.add_argument("--particles", type=int, default=0)
    ap.add_argument("--iters", type=int, default=0)
    ap.add_argument("--elites", type=int, default=0)
    ap.add_argument("--qpos_sigma", type=float, default=0.0,
                    help="per-step |Δqpos| ramped over the horizon; 0 = use cfg.train.val_plan_sigma_step")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    device = cfg.device if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    # --- models -----------------------------------------------------------------
    vision = build_vision_encoder(cfg, device)
    assert vision.decoder is not None, "need the Cosmos decoder.jit next to encoder.jit to decode imagined latents"
    tactile = build_tactile_encoder(cfg, device)
    predictor = VTWMPredictor(
        dim=cfg.model.dim, depth=cfg.model.depth, num_heads=cfg.model.num_heads,
        mlp_ratio=cfg.model.mlp_ratio, num_sensors=cfg.data.num_sensors,
        action_dim=cfg.data.action_dim, action_chunk=cfg.data.action_chunk,
        max_temporal=cfg.model.max_temporal, tactile_dim=cfg.model.get("tactile_dim", 768),
    ).to(device)
    sd = torch.load(args.ckpt, map_location=device)
    predictor.load_state_dict(sd["model"])
    if sd.get("tactile") is not None and hasattr(tactile, "model"):
        tactile.model.load_state_dict(sd["tactile"])
        print("[openloop] loaded fine-tuned tactile encoder weights")
    predictor.eval()
    if hasattr(tactile, "model"):
        tactile.model.eval()
    print(f"[openloop] loaded predictor from {args.ckpt}")

    # --- dataset window geometry ------------------------------------------------
    if args.eval_T > 0:
        cfg.data.T = max(args.eval_T, int(cfg.data.T))
    ds = build_dataset(cfg, val=(args.split == "val"))
    T = int(cfg.data.T)
    ctx = max(1, args.ctx)
    horizon = args.horizon if args.horizon > 0 else (T - ctx)
    horizon = min(horizon, T - ctx)
    assert horizon >= 1, f"horizon must be >=1 (T={T}, ctx={ctx})"
    max_ctx = int(cfg.planning.get("max_context", 9))

    # CEM hyperparameters: CLI override else cfg.planning / cfg.train.
    pcfg = cfg.get("planning", {})
    particles = args.particles or int(pcfg.get("particles", 36))
    iters = args.iters or int(pcfg.get("iters", 10))
    elites = args.elites or int(pcfg.get("elites", 5))
    qpos_sigma = args.qpos_sigma or float(cfg.train.get("val_plan_sigma_step", 0.002))
    frame_stride = int(cfg.data.frame_stride)
    action_chunk = int(cfg.data.action_chunk)
    action_dim = int(cfg.data.action_dim)

    # --- pick RGB-motion-rich windows (so goal != current, planning is non-trivial) ---
    rng = np.random.default_rng(args.seed)
    cand = rng.choice(len(ds), size=min(len(ds), args.scan), replace=False)
    motion = []
    for di in cand:
        item = ds[int(di)]
        rgb = item["rgb"]
        H = horizon
        motion.append((float((rgb[ctx:ctx + H] - rgb[ctx - 1:ctx]).abs().mean()), int(di)))
    motion.sort(reverse=True)
    idxs = [di for _, di in motion[:args.num_windows]]
    print(f"[openloop] split={args.split} | scanned {len(cand)} windows, picked {len(idxs)} "
          f"(motion {motion[0][0]:.3f}..{motion[len(idxs)-1][0]:.3f}) | "
          f"T={T} ctx={ctx} horizon={horizon} | CEM(P={particles},it={iters},E={elites},"
          f"sigma={qpos_sigma},mu={args.mu_init})")

    up = max(1, args.upscale)

    def _write_video(path, frames):
        imageio.mimwrite(path, frames, fps=args.fps, codec="libx264", quality=8, macro_block_size=1)

    rows = []
    for n, di in enumerate(idxs):
        item = ds[int(di)]
        rgb = item["rgb"].unsqueeze(0).to(device)        # (1,T,3,192,320) GT [0,1]
        tac = item["tactile"].unsqueeze(0).to(device)    # (1,T,S,6,H,W)
        act = item["action"].unsqueeze(0).to(device)     # (1,T,chunk,dim) GT
        goal_rgb = item["goal_rgb"][None, None].to(device)  # (1,1,3,192,320) EPISODE final frame
        task = item.get("task", "?")
        cam = item.get("cam", "?")
        H = horizon

        with torch.no_grad():
            s = vision.encode(rgb)                        # (1,T,16,12,20)
            t = tactile.encode(tac)                       # (1,T,S,196,768)
            # Goal latent = the EPISODE's final frame (demonstrated task-completion state, matching
            # deploy frame: -1), NOT a frame inside this window.
            s_goal = vision.encode(goal_rgb)[:, 0]        # (1,16,12,20)
            gt_act = act[0, ctx - 1:ctx - 1 + H]          # (H,chunk,dim): drives frames (ctx-1)..(ctx-1+H-1)

            # CEM plan from the single context frame toward the goal latent (train/deploy regime).
            if args.mu_init == "pose":
                # Deploy-faithful seed: a SINGLE current pose broadcast over all keyframes/chunks,
                # NOT the GT keyframe-0 chunk (that leaks the answer and, with a near-zero h=0 std,
                # pins keyframe 0 to GT -> action_mse trivially 0 there). Every (keyframe, chunk)
                # position drifts from the seed, so the std ramps along BOTH axes
                # (chunk_accumulate=True). Mirrors action_mse_eval in train.py.
                mu_init = gt_act[0, 0]                     # (dim,) current pose, broadcast in cem
                sigma_init = qpos_sigma_ramp(
                    H, action_chunk, action_dim, qpos_sigma, frame_stride,
                    device=device, chunk_accumulate=True)
            else:
                mu_init = None                            # zero seed -> unit-std prior (cem default)
                sigma_init = None
            cem_act, _ = cem_plan(
                predictor, s[:, ctx - 1:ctx], t[:, ctx - 1:ctx], s_goal,
                horizon=H, action_chunk=action_chunk, action_dim=action_dim,
                particles=particles, iters=iters, elites=elites, max_context=max_ctx,
                device=device, mu_init=mu_init,
                sigma_init=sigma_init,
            )                                             # (H,chunk,dim)

            # Imagine forward under the CEM-sampled actions AND under the GT actions.
            roll_cem, _ = predictor.rollout(s[:, ctx - 1:ctx], t[:, ctx - 1:ctx],
                                            cem_act.unsqueeze(0), horizon=H, max_context=max_ctx)
            roll_gt, _ = predictor.rollout(s[:, ctx - 1:ctx], t[:, ctx - 1:ctx],
                                           gt_act.unsqueeze(0), horizon=H, max_context=max_ctx)
            img_cem = vision.decode(roll_cem)[0]          # (H,3,192,320) imagined under CEM actions
            img_gt = vision.decode(roll_gt)[0]            # (H,...) imagined under GT actions
            gt_recon = vision.decode(s[:, ctx:ctx + H])[0]  # encode/decode ceiling for GT future

        # Real images.
        cur_img = rgb[0, ctx - 1]                          # current frame (planning origin)
        goal_img = goal_rgb[0, 0]                          # real goal frame (episode final)
        gt_future = rgb[0, ctx:ctx + H]                    # (H,...) real future

        # Metrics.
        gt_np = gt_act.cpu().numpy()
        cem_np = cem_act.cpu().numpy()
        action_mse = float(np.mean((gt_np - cem_np) ** 2))
        # Per-step CEM planning cost: ||imagined latent_k - goal latent||_2, for GT vs CEM actions.
        cem_cost = (roll_cem[0] - s_goal[0]).flatten(1).pow(2).sum(1).sqrt().cpu().numpy()  # (H,)
        gt_cost = (roll_gt[0] - s_goal[0]).flatten(1).pow(2).sum(1).sqrt().cpu().numpy()    # (H,)
        # Static (no-move) baseline: the cost of just staying at the current latent.
        static_cost = float((s[0, ctx - 1] - s_goal[0]).pow(2).sum().sqrt().item())
        # how close imagination gets to the goal latent, CEM vs GT actions (final step)
        cem_goal_latent_l2 = float(cem_cost[-1])
        gt_goal_latent_l2 = float(gt_cost[-1])
        # imagined-final image vs real goal image (PSNR), CEM vs GT actions
        psnr_cem_goal = psnr(img_cem[-1], goal_img)
        psnr_gt_goal = psnr(img_gt[-1], goal_img)

        tag = f"w{n:02d}_{task}_idx{di}"

        # --- action plot ---
        _action_plot(gt_np, cem_np, os.path.join(args.out_dir, f"{tag}_actions.png"),
                     title=f"{task} idx={di} | action MSE={action_mse:.4f} "
                           f"(GT green vs CEM red, {H}x{action_chunk} raw steps)")

        # --- cost plot: GT-action vs CEM-action imagined-latent->goal distance per step ---
        _cost_plot(gt_cost, cem_cost, static_cost,
                   os.path.join(args.out_dir, f"{tag}_cost.png"),
                   title=f"{task} idx={di} | goal-latent cost: GT vs CEM action "
                         f"(final GT={gt_goal_latent_l2:.2f} CEM={cem_goal_latent_l2:.2f})")

        # --- static panel: current | goal | imagined-final(CEM) ---
        panel = np.concatenate([
            _label(to_uint8(cur_img), "current (frame 0)"),
            _label(to_uint8(goal_img), "goal (episode final)"),
            _label(to_uint8(img_cem[-1]), f"imagined final (CEM) PSNR={psnr_cem_goal:.1f}"),
        ], axis=1)
        if up > 1:
            panel = cv2.resize(panel, (panel.shape[1] * up, panel.shape[0] * up),
                               interpolation=cv2.INTER_NEAREST)
        bar = _title_bar(panel.shape[1], up, f"{task} idx={di} | action MSE={action_mse:.4f}")
        panel = np.concatenate([bar, panel], axis=0)
        cv2.imwrite(os.path.join(args.out_dir, f"{tag}_panel.png"),
                    cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

        # --- rollout video: GT future | Cosmos recon | imagined CEM | imagined GT ---
        frames = []
        for k in range(H):
            row = np.concatenate([
                _label(to_uint8(gt_future[k]), "GT future"),
                _label(to_uint8(gt_recon[k]), "Cosmos recon (ceiling)"),
                _label(to_uint8(img_cem[k]), "imagined (CEM action)"),
                _label(to_uint8(img_gt[k]), "imagined (GT action)"),
            ], axis=1)
            if up > 1:
                row = cv2.resize(row, (row.shape[1] * up, row.shape[0] * up),
                                 interpolation=cv2.INTER_NEAREST)
            tb = _title_bar(row.shape[1], up, f"{task} idx={di} | step {k+1}/{H}")
            frames.append(np.concatenate([tb, row], axis=0))
        _write_video(os.path.join(args.out_dir, f"{tag}_rollout.mp4"), frames)

        rows.append(dict(window=n, episode=int(di), task=task, cam=cam,
                         action_mse=round(action_mse, 5),
                         cem_goal_latent_l2=round(cem_goal_latent_l2, 4),
                         gt_goal_latent_l2=round(gt_goal_latent_l2, 4),
                         static_goal_latent_l2=round(static_cost, 4),
                         psnr_cem_goal=round(psnr_cem_goal, 2),
                         psnr_gt_goal=round(psnr_gt_goal, 2)))
        print(f"  {tag}: action_mse={action_mse:.4f} | goal_latent_l2 CEM={cem_goal_latent_l2:.3f} "
              f"GT={gt_goal_latent_l2:.3f} | PSNR(imag_final vs goal) CEM={psnr_cem_goal:.2f} "
              f"GT={psnr_gt_goal:.2f}")

    summary = dict(
        config=args.config, ckpt=args.ckpt, split=args.split,
        window_T=T, ctx=ctx, horizon=horizon, max_context=max_ctx,
        cem=dict(particles=particles, iters=iters, elites=elites,
                 qpos_sigma=qpos_sigma, mu_init=args.mu_init),
        mean_action_mse=round(float(np.mean([r["action_mse"] for r in rows])), 5),
        mean_cem_goal_latent_l2=round(float(np.mean([r["cem_goal_latent_l2"] for r in rows])), 4),
        mean_gt_goal_latent_l2=round(float(np.mean([r["gt_goal_latent_l2"] for r in rows])), 4),
        mean_static_goal_latent_l2=round(float(np.mean([r["static_goal_latent_l2"] for r in rows])), 4),
        mean_psnr_cem_goal=round(float(np.mean([r["psnr_cem_goal"] for r in rows])), 2),
        mean_psnr_gt_goal=round(float(np.mean([r["psnr_gt_goal"] for r in rows])), 2),
        windows=rows,
    )
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[openloop] windows={len(rows)} | mean action MSE={summary['mean_action_mse']:.4f}")
    print(f"[openloop] mean goal-latent L2: CEM={summary['mean_cem_goal_latent_l2']:.3f} "
          f"(GT-action ceiling={summary['mean_gt_goal_latent_l2']:.3f})")
    print(f"[openloop] mean PSNR(imagined final vs goal): CEM={summary['mean_psnr_cem_goal']:.2f} "
          f"GT={summary['mean_psnr_gt_goal']:.2f} dB")
    print(f"[openloop] per-window actions.npz/.png, panel.png, rollout.mp4 + metrics.json in {args.out_dir}")


if __name__ == "__main__":
    main()
