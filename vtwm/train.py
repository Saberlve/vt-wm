"""Train the VT-WM predictor (and optionally fine-tune the tactile encoder).

Supports single-GPU and multi-GPU DDP (launched via determined.launch.torch_distributed
or torchrun). Features: AdamW + warmup/cosine LR; trainable tactile encoder with scaled LR
and stop-gradient targets; periodic validation eval; checkpoint save/resume; wandb logging.
Only rank 0 logs / evaluates / checkpoints / writes wandb.
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import shutil

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from vtwm.build import build_dataset, build_tactile_encoder, build_vision_encoder
from vtwm.losses import vtwm_loss
from vtwm.models.predictor import VTWMPredictor
from vtwm.models.train_module import VTWMTrainModule
from vtwm.planning.cem import cem_plan


def lr_at(step, warmup, total, peak, final):
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return final + 0.5 * (peak - final) * (1.0 + math.cos(math.pi * progress))


def dist_info():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return world, rank, local_rank


@torch.no_grad()
def evaluate(core, vision, val_loader, device, n_batches, amp=False):
    core.eval()
    tot = tt = ts = 0.0
    n = 0
    for i, batch in enumerate(val_loader):
        if i >= n_batches:
            break
        s = vision.encode(batch["rgb"].to(device))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            loss, l_teacher, l_sampling = core(s, batch["tactile"].to(device), batch["action"].to(device))
        tot += loss.item(); tt += l_teacher.item(); ts += l_sampling.item(); n += 1
    core.train()
    n = max(1, n)
    return tot / n, tt / n, ts / n


@torch.no_grad()
def action_mse_eval(core, vision, val_loader, device, n_windows, cfg, amp=False):
    """Policy-quality metric: the world model has no action output, so we recover one with
    CEM planning. For each val window, plan the last H action steps to reach the window's
    final visual latent (goal), then MSE the planned sequence against the demonstrated GT
    actions. Returns mean per-window action MSE over up to `n_windows` windows.
    """
    core.eval()
    predictor, tactile = core.predictor, core.tactile
    H_cfg = int(cfg.train.get("val_plan_horizon", cfg.train.sampling_horizon))
    sigma0 = float(cfg.train.get("val_plan_sigma", 0.1))
    pcfg = cfg.get("planning", {})
    se = 0.0
    n = 0
    for batch in val_loader:
        if n >= n_windows:
            break
        s = vision.encode(batch["rgb"].to(device))            # (B,T,16,12,20)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            t = tactile.encode(batch["tactile"].to(device))   # (B,T,4,196,768)
            a = batch["action"].to(device)                    # (B,T,chunk,dim)
            B, T = s.shape[:2]
            H = min(H_cfg, T - 1)
            Tc = T - H
            if Tc < 1:
                continue
            for b in range(B):
                if n >= n_windows:
                    break
                gt_act = a[b, Tc - 1 : T - 1]                  # (H,chunk,dim): drives frames Tc-1..T-2
                best_action, _ = cem_plan(
                    predictor, s[b : b + 1, :Tc], t[b : b + 1, :Tc], s[b, -1:],
                    horizon=H, action_chunk=cfg.data.action_chunk, action_dim=cfg.data.action_dim,
                    particles=int(pcfg.get("particles", 36)), iters=int(pcfg.get("iters", 10)),
                    elites=int(pcfg.get("elites", 5)), max_context=int(pcfg.get("max_context", 9)),
                    device=device,
                    mu_init=a[b, Tc - 1],                      # seed from last context action (abs qpos prior)
                    sigma_init=torch.tensor(sigma0, device=device),
                )
                se += (best_action - gt_act).pow(2).mean().item()
                n += 1
    core.train()
    return se / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--resume", default=None, help="checkpoint path or 'auto'")
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config)

    world, rank, local_rank = dist_info()
    is_dist = world > 1
    is_main = rank == 0
    if is_dist:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = cfg.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.train.seed + rank)
    # Mixed precision: bf16 autocast (no GradScaler needed; params stay fp32). Set
    # train.precision to "fp32"/"none" to disable. Frozen vision encoder runs outside autocast.
    prec = str(cfg.train.get("precision", "bf16")).lower()
    amp = prec in ("bf16", "bfloat16") and str(device) != "cpu"
    if is_main:
        print(f"[setup] precision={prec} (autocast bf16={amp})")
    if is_main:
        os.makedirs(cfg.paths.out_dir, exist_ok=True)

    # --- models: vision (frozen, outside DDP) + tactile + predictor (in the DDP module) ---
    vision = build_vision_encoder(cfg, device)
    tactile = build_tactile_encoder(cfg, device)
    predictor = VTWMPredictor(
        dim=cfg.model.dim, depth=cfg.model.depth, num_heads=cfg.model.num_heads,
        mlp_ratio=cfg.model.mlp_ratio, num_sensors=cfg.data.num_sensors,
        action_dim=cfg.data.action_dim, action_chunk=cfg.data.action_chunk,
        max_temporal=cfg.model.max_temporal, tactile_dim=cfg.model.get("tactile_dim", 768),
    ).to(device)
    tactile_trainable = getattr(tactile, "trainable", False)
    core = VTWMTrainModule(tactile, predictor, cfg.train.sampling_horizon, cfg.data.T).to(device)

    if is_main:
        n_pred = sum(p.numel() for p in predictor.parameters() if p.requires_grad)
        n_tac = sum(p.numel() for p in tactile.parameters() if p.requires_grad)
        print(f"[setup] world={world} | predictor={n_pred/1e6:.1f}M "
              f"| tactile(trainable={tactile_trainable})={n_tac/1e6:.1f}M | device={device}")

    # optimizer param groups (built from raw module params; DDP shares the same tensors)
    tac_scale = cfg.train.get("tactile_lr_scale", 0.1)
    groups = [{"params": [p for p in predictor.parameters() if p.requires_grad], "lr_scale": 1.0}]
    if tactile_trainable:
        groups.append({"params": [p for p in tactile.model.parameters() if p.requires_grad], "lr_scale": tac_scale})
    opt = torch.optim.AdamW(groups, lr=cfg.train.peak_lr, betas=tuple(cfg.train.betas),
                            weight_decay=cfg.train.weight_decay)

    # resume (all ranks load the shared checkpoint before DDP wrap)
    step = 0
    resume_path = args.resume
    if resume_path == "auto":
        # pick the highest-step checkpoint subfolder, falling back to a legacy last.pt
        step_ckpts = sorted(
            (p for p in glob.glob(os.path.join(cfg.paths.out_dir, "step_*", "checkpoint.pt"))),
            key=lambda p: int(os.path.basename(os.path.dirname(p)).split("_")[1]),
        )
        if step_ckpts:
            resume_path = step_ckpts[-1]
        else:
            legacy = os.path.join(cfg.paths.out_dir, "last.pt")
            resume_path = legacy if os.path.exists(legacy) else None
    if resume_path and os.path.exists(resume_path):
        ck = torch.load(resume_path, map_location=device)
        predictor.load_state_dict(ck["model"])
        if tactile_trainable and ck.get("tactile") is not None:
            tactile.model.load_state_dict(ck["tactile"])
        if ck.get("optimizer") is not None:
            opt.load_state_dict(ck["optimizer"])
        step = ck.get("step", 0)
        if is_main:
            print(f"[resume] {resume_path} @ step {step}")

    model = DDP(core, device_ids=[local_rank], find_unused_parameters=True) if is_dist else core

    # --- data ---
    # batch_size_per_gpu is PER-GPU; DDP only shards dataset indices, so global = per_gpu * world.
    bs = int(cfg.train.get("batch_size_per_gpu", cfg.train.get("batch_size", 1)))
    if is_main:
        print(f"[data] batch_size_per_gpu={bs} | world={world} | global_batch={bs * world}")
    train_ds = build_dataset(cfg, val=False)
    sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True) if is_dist else None
    loader = DataLoader(train_ds, batch_size=bs, sampler=sampler,
                        shuffle=(sampler is None), drop_last=True,
                        num_workers=cfg.train.get("num_workers", 0), pin_memory=True)
    val_loader = None
    if is_main and cfg.data.get("val_ratio", 0) and cfg.train.get("eval_every", 0):
        val_ds = build_dataset(cfg, val=True)
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=True, drop_last=True,
                                num_workers=cfg.train.get("num_workers", 0), pin_memory=True)
        print(f"[data] train windows={len(train_ds)}  val windows={len(val_ds)}")

    # --- wandb (rank 0) ---
    wcfg = cfg.get("wandb", {})
    use_wandb = bool(wcfg.get("enabled", False)) and is_main
    run = None
    if use_wandb:
        try:
            import wandb
            run = wandb.init(project=wcfg.get("project", "vt-wm"), name=wcfg.get("name", None),
                             mode=wcfg.get("mode", "online"), config=OmegaConf.to_container(cfg, resolve=True),
                             resume="allow")
            # one-time RGB sanity check: upload 4 frames before training
            dbg = next(iter(loader))["rgb"]             # (B,T,3,H,W) in [0,1]
            imgs = dbg.reshape(-1, *dbg.shape[2:])[:4].clamp(0, 1).cpu()
            run.log({"debug/rgb_batch": [wandb.Image(img) for img in imgs]}, step=step)
            print(f"[wandb] logged {len(imgs)} debug RGB images (shape {tuple(imgs.shape[1:])})", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[wandb] disabled ({type(e).__name__}: {str(e)[:80]})")
            use_wandb = False

    def save(tag, subdir=None):
        out = os.path.join(cfg.paths.out_dir, subdir) if subdir else cfg.paths.out_dir
        os.makedirs(out, exist_ok=True)
        path = os.path.join(out, tag)
        torch.save({
            "model": predictor.state_dict(),
            "tactile": tactile.model.state_dict() if tactile_trainable else None,
            "optimizer": opt.state_dict(),
            "step": step,
            "config": OmegaConf.to_container(cfg, resolve=True),
        }, path)
        return path

    def save_step_ckpt(keep=2):
        """Save into a step-named subfolder (out_dir/step_<n>/checkpoint.pt) and
        keep only the most recent `keep` such folders."""
        path = save("checkpoint.pt", subdir=f"step_{step}")
        step_dirs = sorted(
            (d for d in glob.glob(os.path.join(cfg.paths.out_dir, "step_*")) if os.path.isdir(d)),
            key=lambda d: int(os.path.basename(d).split("_")[1]),
        )
        for d in step_dirs[:-keep]:  # drop everything but the newest `keep`
            shutil.rmtree(d, ignore_errors=True)
        return path

    model.train()
    # flat list of all trainable params, used for global grad-norm clipping below
    clip_params = [p for g in groups for p in g["params"]]
    epoch = 0
    if sampler is not None:
        sampler.set_epoch(epoch)  # DistributedSampler needs the epoch for deterministic shuffling
    data_iter = iter(loader)
    pbar = tqdm(total=cfg.train.steps, initial=step, desc="train",
                dynamic_ncols=True, disable=not is_main)
    # step-based loop: one optimizer update per iteration, runs until cfg.train.steps
    while step < cfg.train.steps:
        # pull the next batch; when the loader is exhausted, bump the epoch and restart it
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)  # reshuffle for the new epoch across ranks
            data_iter = iter(loader)
            batch = next(data_iter)

        # frozen vision backbone -> latent states s; encoder runs under no_grad internally
        s = vision.encode(batch["rgb"].to(device))
        # cosine/warmup LR schedule; apply per-group scale (different LR for different param groups)
        lr = lr_at(step, cfg.train.warmup_steps, cfg.train.steps, cfg.train.peak_lr, cfg.train.final_lr)
        for g in opt.param_groups:
            g["lr"] = lr * g["lr_scale"]

        # forward under bf16 autocast; predictor returns total loss + teacher/sampling components
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            loss, l_teacher, l_sampling = model(s, batch["tactile"].to(device), batch["action"].to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(clip_params, 1.0)  # clip to unit norm before stepping
        opt.step()

        # --- console logging (rank 0 only): live postfix + periodic detailed line ---
        if is_main:
            pbar.set_postfix(ep=epoch, L=f"{loss.item():.4f}", lr=f"{lr:.2e}", refresh=False)
            if step % cfg.train.log_every == 0:
                tqdm.write(f"step {step:6d} | epoch {epoch:3d} | lr {lr:.2e} | L {loss.item():.4f} "
                           f"| teacher {l_teacher.item():.4f} | sampling {l_sampling.item():.4f}")
        # per-step metrics to wandb
        if use_wandb:
            run.log({"train/loss": loss.item(), "train/teacher": l_teacher.item(),
                     "train/sampling": l_sampling.item(), "train/lr": lr,
                     "train/grad_norm": float(gnorm), "train/epoch": epoch}, step=step)

        # --- periodic validation (after eval_start, every eval_every steps) ---
        if (val_loader is not None and step > 0 and step >= cfg.train.get("eval_start", 0)
                and step % cfg.train.eval_every == 0):
            v, vt, vs = evaluate(core, vision, val_loader, device, cfg.train.get("val_batches", 8), amp)
            log = {"val/loss": v, "val/teacher": vt, "val/sampling": vs}
            msg = f"  [val] step {step} | L {v:.4f} | teacher {vt:.4f} | sampling {vs:.4f}"
            if cfg.train.get("eval_action_mse", True):
                amse = action_mse_eval(core, vision, val_loader, device,
                                       cfg.train.get("val_action_windows", 4), cfg, amp)
                log["val/action_mse"] = amse
                msg += f" | action_mse {amse:.4f}"
            tqdm.write(msg)
            if use_wandb:
                run.log(log, step=step)
        # --- periodic checkpoint (rank 0): step-named subfolder, keep newest few ---
        if is_main and cfg.train.get("save_every", 0) and step > 0 and step % cfg.train.save_every == 0:
            save_step_ckpt(cfg.train.get("keep_last", 2))
        step += 1
        pbar.update(1)
    pbar.close()

    if is_main:
        if val_loader is not None:
            v, vt, vs = evaluate(core, vision, val_loader, device, cfg.train.get("val_batches", 8), amp)
            log = {"val/loss": v, "val/teacher": vt, "val/sampling": vs}
            msg = f"[final-val] L {v:.4f} | teacher {vt:.4f} | sampling {vs:.4f}"
            if cfg.train.get("eval_action_mse", True):
                amse = action_mse_eval(core, vision, val_loader, device,
                                       cfg.train.get("val_action_windows", 4), cfg, amp)
                log["val/action_mse"] = amse
                msg += f" | action_mse {amse:.4f}"
            print(msg, flush=True)
            if use_wandb:
                run.log(log, step=step)
        save_step_ckpt(cfg.train.get("keep_last", 2))
        print(f"[saved] {save('predictor.pt')}", flush=True)
        if use_wandb:
            run.finish()
    if is_dist:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
