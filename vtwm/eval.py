"""Benchmark the VT-WM by imagination quality on held-out ManiFeel episodes.

For each window we encode the first `--ctx` frames as context, autoregressively roll out
the world model under the ground-truth action sequence, decode the predicted visual
latents back to RGB (Cosmos decoder), and compare to the real future frames.

Each output video is a labeled, upscaled panel:

    [ GT (real future) | Cosmos recon (ceiling) | VT-WM prediction ]

with the ManiFeel task name, camera, and a frame counter burned in, so it is clear what
task is being shown and what each panel means.

Metrics: PSNR of the prediction vs GT, the encode/decode ceiling, and a static
"frame-freeze" baseline. SUCCESS RATE = fraction of windows whose imagined rollout beats
the frame-freeze baseline in mean PSNR (the action-conditioned prediction is better than
assuming nothing moves). Real-robot planning success requires hardware and is not measured.
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


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = (a - b).pow(2).mean().item()
    return 99.0 if mse <= 1e-12 else 10.0 * math.log10(1.0 / mse)


def to_uint8(img_chw: torch.Tensor) -> np.ndarray:
    return (img_chw.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def _meta(ds, di):
    """Read a window's metadata (task/episode/frame) without decoding heavy tensors."""
    item = ds[int(di)]
    return item


def _label(img: np.ndarray, text: str, scale: float = 0.5, color=(255, 255, 255)) -> np.ndarray:
    """Draw a caption with a dark background strip at the top-left of an RGB image."""
    img = img.copy()
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.rectangle(img, (0, 0), (min(img.shape[1], tw + 8), th + bl + 6), (0, 0, 0), -1)
    cv2.putText(img, text, (4, th + 3), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/manifeel.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--num_episodes", type=int, default=6)
    ap.add_argument("--ctx", type=int, default=2, help="context frames before rollout")
    ap.add_argument("--eval_T", type=int, default=16, help="window length for eval videos (>= cfg.data.T)")
    ap.add_argument("--out_dir", default="./eval_out")
    ap.add_argument("--fps", type=int, default=3)
    ap.add_argument("--upscale", type=int, default=2, help="panel upscale factor for readable videos")
    ap.add_argument("--scan", type=int, default=120, help="candidate windows scanned for motion")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    device = cfg.device if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    vision = build_vision_encoder(cfg, device)
    assert vision.decoder is not None, "need Cosmos decoder.jit next to encoder.jit"
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
    predictor.eval()
    if hasattr(tactile, "model"):
        tactile.model.eval()
    print(f"[eval] loaded {args.ckpt}")

    # Longer eval windows than the T=9 training window give watchable clips that show the
    # actual manipulation. The model still imagines with its trained context (max_context).
    cfg.data.T = max(args.eval_T, int(cfg.data.T))
    ds = build_dataset(cfg, val=True)
    T = cfg.data.T
    ctx = args.ctx
    horizon = T - ctx
    max_ctx = int(cfg.planning.max_context)

    # Select motion-rich windows: a frame-freeze baseline is trivially perfect on static
    # clips, so benchmark on windows where the scene actually moves.
    rng = np.random.default_rng(0)
    cand = rng.choice(len(ds), size=min(len(ds), args.scan), replace=False)
    motion = []
    for di in cand:
        rgb = ds[int(di)]["rgb"]
        motion.append((float((rgb[ctx:] - rgb[ctx - 1:ctx]).abs().mean()), int(di)))
    motion.sort(reverse=True)
    idxs = [di for _, di in motion[:args.num_episodes]]
    print(f"[eval] picked {len(idxs)} motion windows from {len(cand)} scanned "
          f"(motion {motion[0][0]:.3f}..{motion[len(idxs)-1][0]:.3f}); "
          f"window T={T}, ctx={ctx}, horizon={horizon}")

    up = max(1, args.upscale)
    rows = []
    step_psnr_acc = [[] for _ in range(horizon)]
    for n, di in enumerate(idxs):
        item = ds[int(di)]
        rgb = item["rgb"].unsqueeze(0).to(device)       # (1,T,3,192,320) GT [0,1]
        tac = item["tactile"].unsqueeze(0).to(device)
        act = item["action"].unsqueeze(0).to(device)
        task = item.get("task", "?")
        cam = item.get("cam", "?")
        with torch.no_grad():
            s = vision.encode(rgb)                        # (1,T,16,12,20)
            t = tactile.encode(tac)                       # (1,T,S,196,768)
            roll_s, roll_t = predictor.rollout(s[:, :ctx], t[:, :ctx], act, horizon=horizon,
                                               max_context=max_ctx)
            pred_frames = vision.decode(roll_s)[0]        # (horizon,3,192,320)
            gt_recon = vision.decode(s[:, ctx:])[0]       # encode/decode ceiling
        gt_frames = rgb[0, ctx:]                          # (horizon,3,192,320) real
        static = rgb[0, ctx - 1:ctx].repeat(horizon, 1, 1, 1)  # freeze last context frame

        p_pred = psnr(pred_frames, gt_frames)
        p_ceiling = psnr(gt_recon, gt_frames)
        p_static = psnr(static, gt_frames)
        success = p_pred > p_static
        # per-step vision PSNR (how the imagination degrades over the horizon)
        per_step = [round(psnr(pred_frames[k], gt_frames[k]), 2) for k in range(horizon)]
        for k in range(horizon):
            step_psnr_acc[k].append(per_step[k])
        # tactile latent prediction error vs a frame-freeze tactile baseline
        gt_tac = t[:, ctx:]                               # (1,horizon,S,196,768)
        tac_static = t[:, ctx - 1:ctx].repeat(1, horizon, 1, 1, 1)
        tl_pred = float((roll_t - gt_tac).abs().mean())
        tl_static = float((tac_static - gt_tac).abs().mean())
        tac_success = tl_pred < tl_static
        rows.append({"episode": int(di), "task": task, "cam": cam,
                     "psnr_pred": round(p_pred, 2), "psnr_ceiling": round(p_ceiling, 2),
                     "psnr_static": round(p_static, 2), "beats_static": bool(success),
                     "psnr_per_step": per_step,
                     "tactile_l1_pred": round(tl_pred, 4), "tactile_l1_static": round(tl_static, 4),
                     "tactile_beats_static": bool(tac_success)})

        # video: [ GT | Cosmos recon | VT-WM prediction ], labeled + upscaled.
        frames = []
        for k in range(horizon):
            panels = [
                (_label(to_uint8(gt_frames[k]), "GT (real future)"), ),
                (_label(to_uint8(gt_recon[k]), "Cosmos recon (ceiling)"), ),
                (_label(to_uint8(pred_frames[k]), "VT-WM prediction"), ),
            ]
            row = np.concatenate([p[0] for p in panels], axis=1)
            if up > 1:
                row = cv2.resize(row, (row.shape[1] * up, row.shape[0] * up),
                                 interpolation=cv2.INTER_NEAREST)
            # top title bar: task + camera + frame counter
            bar = np.zeros((26 * up, row.shape[1], 3), np.uint8)
            title = f"task: {task}  |  cam: {cam}  |  predicted frame {k + 1}/{horizon}"
            cv2.putText(bar, title, (6, 18 * up // 1), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5 * up, (0, 255, 0), 1, cv2.LINE_AA)
            frames.append(np.concatenate([bar, row], axis=0))
        vid_path = os.path.join(args.out_dir, f"ep{n:02d}_{task}.mp4")
        imageio.mimwrite(vid_path, frames, fps=args.fps, codec="libx264", quality=8,
                         macro_block_size=1)
        print(f"  ep{n:02d} [{task}] idx={int(di):6d} | PSNR pred={p_pred:5.2f} "
              f"ceiling={p_ceiling:5.2f} static={p_static:5.2f} | {'OK' if success else '--'} "
              f"| {os.path.basename(vid_path)}")

    succ = float(np.mean([r["beats_static"] for r in rows]))
    tac_succ = float(np.mean([r["tactile_beats_static"] for r in rows]))
    mp = float(np.mean([r["psnr_pred"] for r in rows]))
    mc = float(np.mean([r["psnr_ceiling"] for r in rows]))
    msx = float(np.mean([r["psnr_static"] for r in rows]))
    step_curve = [round(float(np.mean(v)), 2) if v else None for v in step_psnr_acc]

    # per-task aggregation
    per_task = {}
    for r in rows:
        per_task.setdefault(r["task"], []).append(r)
    task_summary = {tk: {"n": len(rs),
                         "vision_success": round(float(np.mean([x["beats_static"] for x in rs])), 2),
                         "tactile_success": round(float(np.mean([x["tactile_beats_static"] for x in rs])), 2),
                         "mean_psnr_pred": round(float(np.mean([x["psnr_pred"] for x in rs])), 2)}
                    for tk, rs in per_task.items()}

    summary = {"num_episodes": len(rows), "window_T": T, "ctx": ctx, "horizon": horizon,
               "vision_success_rate_vs_static": round(succ, 3),
               "tactile_success_rate_vs_static": round(tac_succ, 3),
               "mean_psnr_pred": round(mp, 2), "mean_psnr_ceiling": round(mc, 2),
               "mean_psnr_static": round(msx, 2), "psnr_per_step": step_curve,
               "per_task": task_summary, "episodes": rows}
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[benchmark] episodes={len(rows)} | window T={T} ctx={ctx} horizon={horizon}")
    print(f"[benchmark] VISION  success(pred>freeze)={succ*100:.0f}%  | mean PSNR  "
          f"pred={mp:.2f}  ceiling(enc-dec)={mc:.2f}  freeze={msx:.2f} dB")
    print(f"[benchmark] TACTILE success(pred>freeze in latent L1)={tac_succ*100:.0f}%")
    print(f"[benchmark] PSNR vs rollout step: {step_curve}")
    print(f"[benchmark] per task:")
    for tk, v in sorted(task_summary.items()):
        print(f"    {tk:24s} n={v['n']} | vis_succ={v['vision_success']} "
              f"tac_succ={v['tactile_success']} psnr_pred={v['mean_psnr_pred']}")
    print(f"[benchmark] labeled videos + metrics.json in {args.out_dir}")


if __name__ == "__main__":
    main()
