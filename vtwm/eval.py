"""Benchmark the VT-WM by imagination quality on held-out ManiFeel episodes.

For each window we encode the first `--ctx` frames as context, autoregressively roll out
the world model under the ground-truth action sequence, decode the predicted visual
latents back to RGB (Cosmos decoder), and compare to the real future frames.

Each output video is a labeled, upscaled panel:

    [ GT (real future) | Cosmos recon (ceiling) | VT-WM prediction ]

with the ManiFeel task name, camera, and a frame counter burned in, so it is clear what
task is being shown and what each panel means.

When a Sparsh force-field decoder is configured (paths.sparsh_forcefield_ckpt), a tactile
row is appended showing, per sensor, the physical readout:

    [ GT tactile | GT normal | pred normal~ | GT shear | pred shear~ ]

GT panels are decoded faithfully from the real encoder activations; the predicted-latent
panels (marked ~) use the degenerate single-latent decode and are approximate (the world
model produces only the final-layer latent, not the decoder's intermediate hooks), so they
are qualitative only. Without the decoder, the row falls back to a token-L1 error heatmap.

Metrics: PSNR of the prediction vs GT, the encode/decode ceiling, and a static
"frame-freeze" baseline. SUCCESS RATE = fraction of windows whose imagined rollout beats
the frame-freeze baseline in mean PSNR (the action-conditioned prediction is better than
assuming nothing moves). Real-robot planning success requires hardware and is not measured.
"""
from __future__ import annotations

import argparse
import bisect
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


def _dynamic_tactile_sequence(ds, di: int):
    """Return per-timestep tactile for datasets that expose raw dynamic UniVTAC frames."""
    di = int(di)
    if hasattr(ds, "get_tactile_sequence"):
        return ds.get_tactile_sequence(di)
    if hasattr(ds, "datasets") and hasattr(ds, "cumulative_sizes"):
        subset_idx = bisect.bisect_right(ds.cumulative_sizes, di)
        prev = 0 if subset_idx == 0 else ds.cumulative_sizes[subset_idx - 1]
        subset = ds.datasets[subset_idx]
        if hasattr(subset, "get_tactile_sequence"):
            return subset.get_tactile_sequence(di - prev)
    return None


def _eval_tactile_sequence(ds, di: int, item):
    """Use dynamic tactile GT when available; otherwise fall back to item['tactile']."""
    dyn = _dynamic_tactile_sequence(ds, di)
    if dyn is not None:
        return dyn, True
    return item["tactile"], False


def _label(img: np.ndarray, text: str, scale: float = 0.5, color=(255, 255, 255)) -> np.ndarray:
    """Draw a caption with a dark background strip at the top-left of an RGB image."""
    img = img.copy()
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.rectangle(img, (0, 0), (min(img.shape[1], tw + 8), th + bl + 6), (0, 0, 0), -1)
    cv2.putText(img, text, (4, th + 3), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
    return img


def _tac_gt_panel(tac_6chw: torch.Tensor, label: str) -> np.ndarray:
    """Render the GT tactile input image (current frame = first 3 of the 6 channels)."""
    return _label(to_uint8(tac_6chw[:3]), label)


def _heatmap_panel(err_2d: torch.Tensor, vmax: float, label: str, size=(192, 192)) -> np.ndarray:
    """Render a 14x14 token-L1 error map as a JET heatmap upscaled to `size` (h, w)."""
    norm = (err_2d / vmax).clamp(0, 1).cpu().numpy()
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)  # BGR
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    heat = cv2.resize(heat, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
    return _label(heat, label)


def _normal_panel(normal_2d: torch.Tensor, vmax: float, label: str, size=(224, 224)) -> np.ndarray:
    """Render a decoded normal-force (contact pressure) map as a JET heatmap."""
    norm = (normal_2d.float() / vmax).clamp(0, 1).cpu().numpy()
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    heat = cv2.resize(heat, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
    return _label(heat, label)


def _shear_panel(shear_2chw: torch.Tensor, label: str, size=(224, 224)) -> np.ndarray:
    """Render a decoded shear-force flow field (2,H,W) as an RGB flow image."""
    from torchvision.utils import flow_to_image
    img = flow_to_image(shear_2chw.float().cpu()).permute(1, 2, 0).numpy().astype(np.uint8)
    img = cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
    return _label(img, label)


def _title_bar(width: int, up: int, task: str, cam: str, k: int, horizon: int, kind: str) -> np.ndarray:
    bar = np.zeros((26 * up, width, 3), np.uint8)
    title = f"task: {task}  |  cam: {cam}  |  {kind} frame {k + 1}/{horizon}"
    cv2.putText(bar, title, (6, 18 * up), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * up,
                (0, 255, 0), 1, cv2.LINE_AA)
    return bar


def render_vision_frames(gt_frames, gt_recon, pred_frames, task, cam, up, horizon):
    """Vision-only clip: [ GT | Cosmos recon (ceiling) | VT-WM prediction ]."""
    frames = []
    for k in range(horizon):
        row = np.concatenate([
            _label(to_uint8(gt_frames[k]), "GT (real future)"),
            _label(to_uint8(gt_recon[k]), "Cosmos recon (ceiling)"),
            _label(to_uint8(pred_frames[k]), "VT-WM prediction"),
        ], axis=1)
        if up > 1:
            row = cv2.resize(row, (row.shape[1] * up, row.shape[0] * up),
                             interpolation=cv2.INTER_NEAREST)
        bar = _title_bar(row.shape[1], up, task, cam, k, horizon, "predicted")
        frames.append(np.concatenate([bar, row], axis=0))
    return frames


def render_tactile_frames(gt_tac_imgs, S, task, cam, up, horizon, ff_fields=None,
                          tac_err=None, tac_vmax=1.0, panel=224):
    """Tactile-only clip: one full-width row per sensor (stacked vertically).

    With the force-field decoder: [ GT tactile | normal GT | normal pred~ | shear GT |
    shear pred~ ]. Otherwise: [ GT tactile | predicted-latent L1 heatmap ].
    """
    frames = []
    for k in range(horizon):
        sensor_rows = []
        for s in range(S):
            panels = [_tac_gt_panel(gt_tac_imgs[k, s], f"S{s} GT tactile")]
            if ff_fields is not None:
                n_gt, n_pr, s_gt, s_pr, nvmax = ff_fields
                panels += [
                    _normal_panel(n_gt[k, s], nvmax, f"S{s} normal GT", size=(panel, panel)),
                    _normal_panel(n_pr[k, s], nvmax, f"S{s} normal pred~", size=(panel, panel)),
                    _shear_panel(s_gt[k, s], f"S{s} shear GT", size=(panel, panel)),
                    _shear_panel(s_pr[k, s], f"S{s} shear pred~", size=(panel, panel)),
                ]
            else:
                panels.append(_heatmap_panel(
                    tac_err[k, s], tac_vmax, f"S{s} err {float(tac_err[k, s].mean()):.3f}",
                    size=(panel, panel)))
            panels = [cv2.resize(p, (panel, panel), interpolation=cv2.INTER_NEAREST)
                      for p in panels]
            sensor_rows.append(np.concatenate(panels, axis=1))
        grid = np.concatenate(sensor_rows, axis=0)
        if up > 1:
            grid = cv2.resize(grid, (grid.shape[1] * up, grid.shape[0] * up),
                              interpolation=cv2.INTER_NEAREST)
        bar = _title_bar(grid.shape[1], up, task, cam, k, horizon, "tactile (GT vs pred~)")
        frames.append(np.concatenate([bar, grid], axis=0))
    return frames


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

    # Optional: Sparsh force-field decoder for interpretable tactile videos. GT frames are
    # decoded faithfully (real encoder hooks); the predicted latent is decoded via the
    # degenerate single-latent path (approximate, qualitative only).
    ff = None
    ff_ckpt = cfg.paths.get("sparsh_forcefield_ckpt", None) if "paths" in cfg else None
    if ff_ckpt and os.path.exists(ff_ckpt) and hasattr(tactile, "model"):
        from vtwm.encoders.sparsh_forcefield import SparshForceFieldDecoder
        ff = SparshForceFieldDecoder(ff_ckpt, tactile.model, device=device)
        print(f"[eval] force-field decoder loaded ({ff_ckpt})")

    # Longer eval windows than the T=9 training window give watchable clips that show the
    # actual manipulation. The model still imagines with its trained context (max_context).
    cfg.data.T = max(args.eval_T, int(cfg.data.T))
    ds = build_dataset(cfg, val=True)
    T = cfg.data.T
    ctx = args.ctx
    horizon = T - ctx
    max_ctx = int(cfg.planning.max_context)

    # Select motion-rich windows. Vision videos use RGB motion; tactile videos use the
    # dynamic per-timestep tactile sequence when the dataset exposes it.
    rng = np.random.default_rng(0)
    cand = rng.choice(len(ds), size=min(len(ds), args.scan), replace=False)
    rgb_motion = []
    tac_motion = []
    dynamic_tactile_seen = False
    for di in cand:
        item = ds[int(di)]
        rgb = item["rgb"]
        tac_eval, is_dynamic = _eval_tactile_sequence(ds, int(di), item)
        dynamic_tactile_seen = dynamic_tactile_seen or is_dynamic
        rgb_motion.append((float((rgb[ctx:] - rgb[ctx - 1:ctx]).abs().mean()), int(di)))
        tac_motion.append((float((tac_eval[ctx:] - tac_eval[ctx - 1:ctx]).abs().mean()), int(di)))
    rgb_motion.sort(reverse=True)
    tac_motion.sort(reverse=True)
    vis_idxs = [di for _, di in rgb_motion[:args.num_episodes]]
    tac_idxs = [di for _, di in tac_motion[:args.num_episodes]] if dynamic_tactile_seen else vis_idxs
    idxs = list(dict.fromkeys(vis_idxs + tac_idxs))
    vis_set, tac_set = set(vis_idxs), set(tac_idxs)
    print(f"[eval] picked {len(vis_idxs)} RGB-motion windows from {len(cand)} scanned "
          f"(motion {rgb_motion[0][0]:.3f}..{rgb_motion[len(vis_idxs)-1][0]:.3f}); "
          f"{len(tac_idxs)} tactile-motion windows "
          f"(motion {tac_motion[0][0]:.3f}..{tac_motion[len(tac_idxs)-1][0]:.3f}, "
          f"dynamic_gt={dynamic_tactile_seen}); "
          f"window T={T}, ctx={ctx}, horizon={horizon}")

    up = max(1, args.upscale)
    cache = {}                                    # di -> per-window tensors / metrics
    step_psnr_acc = [[] for _ in range(horizon)]
    for di in idxs:
        item = ds[int(di)]
        rgb = item["rgb"].unsqueeze(0).to(device)       # (1,T,3,192,320) GT [0,1]
        tac_eval, tac_dynamic = _eval_tactile_sequence(ds, int(di), item)
        tac = tac_eval.unsqueeze(0).to(device)           # dynamic GT tactile when available
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
        per_step = [round(psnr(pred_frames[k], gt_frames[k]), 2) for k in range(horizon)]
        if di in vis_set:
            for k in range(horizon):
                step_psnr_acc[k].append(per_step[k])
        # tactile latent prediction error vs a frame-freeze tactile baseline
        gt_tac = t[:, ctx:]                               # (1,horizon,S,196,768)
        tac_static = t[:, ctx - 1:ctx].repeat(1, horizon, 1, 1, 1)
        tl_pred = float((roll_t - gt_tac).abs().mean())
        tl_static = float((tac_static - gt_tac).abs().mean())
        tac_success = tl_pred < tl_static
        S = tac.shape[2]
        tac_err = (roll_t - gt_tac).abs().mean(dim=-1)[0].reshape(horizon, S, 14, 14)
        tac_vmax = max(float(tac_err.max()), 1e-8)
        gt_tac_imgs = tac[0, ctx:]                        # (horizon,S,6,H,W) real tactile

        cache[di] = dict(task=task, cam=cam, p_pred=p_pred, p_ceiling=p_ceiling,
                         p_static=p_static, success=success, per_step=per_step,
                         tl_pred=tl_pred, tl_static=tl_static, tac_success=tac_success,
                         dynamic_tactile_gt=tac_dynamic)

        # Render + write the VISION clip for vision-motion windows.
        if di in vis_set:
            frames = render_vision_frames(gt_frames, gt_recon, pred_frames, task, cam, up, horizon)
            cache[di]["vision_frames"] = frames

        # Render + write the TACTILE clip for tactile-motion windows.
        if di in tac_set:
            ff_fields = None
            if ff is not None:
                with torch.no_grad():
                    n_gt, s_gt = ff.decode_from_images(
                        gt_tac_imgs.reshape(horizon * S, *gt_tac_imgs.shape[2:]))
                    n_pr, s_pr = ff.decode_from_latent(
                        roll_t[0].reshape(horizon * S, *roll_t.shape[3:]))
                hh, ww = n_gt.shape[-2:]
                n_gt = n_gt.reshape(horizon, S, hh, ww)
                n_pr = n_pr.reshape(horizon, S, hh, ww)
                s_gt = s_gt.reshape(horizon, S, 2, hh, ww)
                s_pr = s_pr.reshape(horizon, S, 2, hh, ww)
                nvmax = max(float(n_gt.max()), float(n_pr.max()), 1e-8)
                ff_fields = (n_gt, n_pr, s_gt, s_pr, nvmax)
            frames = render_tactile_frames(gt_tac_imgs, S, task, cam, up, horizon,
                                           ff_fields=ff_fields, tac_err=tac_err, tac_vmax=tac_vmax)
            cache[di]["tactile_frames"] = frames

    def _write(path, frames):
        imageio.mimwrite(path, frames, fps=args.fps, codec="libx264", quality=8,
                         macro_block_size=1)

    print("[vision] clips (RGB-motion windows):")
    for n, di in enumerate(vis_idxs):
        c = cache[di]
        vp = os.path.join(args.out_dir, f"ep{n:02d}_{c['task']}_vision.mp4")
        _write(vp, c["vision_frames"])
        print(f"  ep{n:02d} [{c['task']}] idx={di:6d} | PSNR pred={c['p_pred']:5.2f} "
              f"ceiling={c['p_ceiling']:5.2f} static={c['p_static']:5.2f} | "
              f"{'OK' if c['success'] else '--'} | {os.path.basename(vp)}")
    print("[tactile] clips (dynamic GT tactile when available; pred~ is rollout latent decode):")
    for n, di in enumerate(tac_idxs):
        c = cache[di]
        tp = os.path.join(args.out_dir, f"ep{n:02d}_{c['task']}_tactile.mp4")
        _write(tp, c["tactile_frames"])
        print(f"  ep{n:02d} [{c['task']}] idx={di:6d} | tac L1 pred={c['tl_pred']:.4f} "
              f"freeze={c['tl_static']:.4f} | {'OK' if c['tac_success'] else '--'} | "
              f"{os.path.basename(tp)}")

    rows = [dict(episode=di, task=cache[di]["task"], cam=cache[di]["cam"],
                 psnr_pred=round(cache[di]["p_pred"], 2),
                 psnr_ceiling=round(cache[di]["p_ceiling"], 2),
                 psnr_static=round(cache[di]["p_static"], 2),
                 beats_static=bool(cache[di]["success"]),
                 psnr_per_step=cache[di]["per_step"]) for di in vis_idxs]
    tac_rows = [dict(episode=di, task=cache[di]["task"],
                     tactile_l1_pred=round(cache[di]["tl_pred"], 4),
                     tactile_l1_static=round(cache[di]["tl_static"], 4),
                     tactile_beats_static=bool(cache[di]["tac_success"]),
                     dynamic_tactile_gt=bool(cache[di]["dynamic_tactile_gt"])) for di in tac_idxs]

    succ = float(np.mean([r["beats_static"] for r in rows]))
    tac_succ = float(np.mean([r["tactile_beats_static"] for r in tac_rows]))
    mp = float(np.mean([r["psnr_pred"] for r in rows]))
    mc = float(np.mean([r["psnr_ceiling"] for r in rows]))
    msx = float(np.mean([r["psnr_static"] for r in rows]))
    step_curve = [round(float(np.mean(v)), 2) if v else None for v in step_psnr_acc]

    # per-task aggregation (vision and tactile come from separate window sets)
    vt, tt = {}, {}
    for r in rows:
        vt.setdefault(r["task"], []).append(r)
    for r in tac_rows:
        tt.setdefault(r["task"], []).append(r)
    task_summary = {}
    for tk in sorted(set(vt) | set(tt)):
        vs, ts = vt.get(tk, []), tt.get(tk, [])
        task_summary[tk] = {
            "n_vision": len(vs), "n_tactile": len(ts),
            "vision_success": round(float(np.mean([x["beats_static"] for x in vs])), 2) if vs else None,
            "tactile_success": round(float(np.mean([x["tactile_beats_static"] for x in ts])), 2) if ts else None,
            "mean_psnr_pred": round(float(np.mean([x["psnr_pred"] for x in vs])), 2) if vs else None}

    summary = {"window_T": T, "ctx": ctx, "horizon": horizon,
               "num_vision_episodes": len(rows), "num_tactile_episodes": len(tac_rows),
               "dynamic_tactile_gt": bool(dynamic_tactile_seen),
               "vision_success_rate_vs_static": round(succ, 3),
               "tactile_success_rate_vs_static": round(tac_succ, 3),
               "mean_psnr_pred": round(mp, 2), "mean_psnr_ceiling": round(mc, 2),
               "mean_psnr_static": round(msx, 2), "psnr_per_step": step_curve,
               "per_task": task_summary,
               "vision_episodes": rows, "tactile_episodes": tac_rows}
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[benchmark] vision_eps={len(rows)} tactile_eps={len(tac_rows)} | "
          f"window T={T} ctx={ctx} horizon={horizon}")
    print(f"[benchmark] VISION  success(pred>freeze)={succ*100:.0f}%  | mean PSNR  "
          f"pred={mp:.2f}  ceiling(enc-dec)={mc:.2f}  freeze={msx:.2f} dB")
    print(f"[benchmark] TACTILE success(pred>freeze in latent L1)={tac_succ*100:.0f}% "
          f"(dynamic GT tactile={dynamic_tactile_seen})")
    print(f"[benchmark] PSNR vs rollout step: {step_curve}")
    print(f"[benchmark] per task:")
    for tk, v in sorted(task_summary.items()):
        print(f"    {tk:24s} vis(n={v['n_vision']})={v['vision_success']} psnr={v['mean_psnr_pred']} | "
              f"tac(n={v['n_tactile']})={v['tactile_success']}")
    print(f"[benchmark] labeled videos + metrics.json in {args.out_dir}")


if __name__ == "__main__":
    main()
