"""Closed-loop VT-WM policy for the UniVTAC benchmark.

VT-WM is a *world model*, not a controller: it predicts future visual/tactile latents given
an action sequence and selects actions by CEM planning that minimizes the L2 distance between
the final imagined visual latent and a goal latent. This wrapper makes it act in the UniVTAC
Isaac Sim closed loop (scripts/eval_policy.py) as a `BasePolicy`:

  reset()  -> clear rolling latent context + tactile frame history, capture tactile background
  eval()   -> encode the live obs, append to context, CEM-plan toward the cached goal latent,
              execute the first planned joint qpos via task.take_action(..., 'qpos').

The goal latent is the encoded final head-camera frame of a reference expert episode from the
UniVTAC dataset (configured in deploy.yml). Observation preprocessing mirrors
`vtwm/data/univtac_dataset.py` exactly so the live latents match the training distribution.
"""
import os
import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))  # for `_base_policy`

import cv2
import yaml
import numpy as np
import torch

from .._base_policy import BasePolicy


def _resolve_repo(deploy_repo):
    if deploy_repo:
        return Path(deploy_repo).expanduser().resolve()
    # …/vt-wm/third_party/UniVTAC/policy/VTWM/deploy_policy.py -> parents[4] == vt-wm root
    return Path(__file__).resolve().parents[4]


class Policy(BasePolicy):
    def __init__(self, args):
        super().__init__(args)
        self.repo = _resolve_repo(args.get("vtwm_repo"))
        # Make vtwm importable inside the UniVTAC python (+ vendored pure-python deps).
        for p in (str(self.repo), str(self.repo / ".univtac_pydeps")):
            if p not in sys.path:
                sys.path.insert(0, p)

        from omegaconf import OmegaConf
        from vtwm.build import build_tactile_encoder, build_vision_encoder
        from vtwm.models.predictor import VTWMPredictor
        from vtwm.planning.cem import cem_plan, qpos_sigma_ramp

        self._cem_plan = cem_plan
        self._qpos_sigma_ramp = qpos_sigma_ramp
        self.task_name = args["task_name"]
        # Device pinning. The launch script puts Isaac Sim's RTX renderer + PhysX on one GPU
        # and this policy's CEM inference on another to avoid the single-card Vulkan-render +
        # CUDA-inference driver deadlock. VTWM_DEVICE (set by the launcher) wins over the yml
        # `device`; both refer to the *visible* index after CUDA_VISIBLE_DEVICES remapping.
        self.device = os.environ.get("VTWM_DEVICE", args.get("device", "cuda:0"))
        if not torch.cuda.is_available():
            self.device = "cpu"

        cfg_path = self.repo / args.get("vtwm_config", "configs/univtac.yaml")
        self.cfg = OmegaConf.load(str(cfg_path))
        d = self.cfg.data
        self.rgb_hw = tuple(d.rgb_hw)
        self.tactile_hw = tuple(d.tactile_hw)
        self.num_frames = int(d.get("tactile_num_frames", 2))
        self.stride = int(d.get("tactile_stride", 5))
        # Live obs arrive at the sim control rate (~60Hz); training downsamples vision to ~6Hz.
        # Replan + grow the rolling context only every `frame_stride` live steps to match.
        self.frame_stride = max(1, int(d.get("frame_stride", 1)))
        self.action_dim = int(d.action_dim)
        self.action_chunk = int(d.action_chunk)
        # Match the dataset's RGB resize (default cover-crop, see univtac_dataset.rgb_resize) so
        # live frames land in the same aspect/crop distribution the predictor trained on, instead
        # of the plain stretch this wrapper used before.
        from vtwm.data.univtac_dataset import _RGB_RESIZERS
        self._rgb_resize = _RGB_RESIZERS[str(d.get("rgb_resize", "crop"))]
        self._setup_rgb_domain(args.get("rgb_domain", {}))

        pl = args.get("planning", {})
        self.horizon = int(pl.get("horizon", 6))
        self.particles = int(pl.get("particles", 16))
        self.iters = int(pl.get("iters", 4))
        self.elites = int(pl.get("elites", 4))
        self.qpos_sigma_step = float(pl.get("qpos_sigma_step", 0.002))  # per-step |Δqpos|, ramped in CEM
        self.max_context = int(pl.get("max_context", self.cfg.planning.get("max_context", 9)))
        # How many of the H planned steps to execute open-loop before replanning. Default = horizon
        # (use the whole plan, fewer CEM calls); set to 1 for pure receding-horizon MPC.
        self.exec_steps = max(1, min(self.horizon, int(pl.get("exec_steps", self.horizon))))
        # Optional episode-length cap for quick smoke runs. The eval loop runs until
        # task.take_action_cnt >= task.cfg.step_lim (300 by default), each step doing a full CEM
        # plan, so one episode can take a long time. VTWM_MAX_STEPS (or deploy `max_steps`) clamps
        # task.cfg.step_lim on the first eval() so a single episode finishes fast. 0 == no cap.
        self.max_steps = int(os.environ.get("VTWM_MAX_STEPS", args.get("max_steps", 0) or 0))

        # --- build encoders + predictor, load checkpoint -------------------------------
        self.vision = build_vision_encoder(self.cfg, self.device)
        self.tactile = build_tactile_encoder(self.cfg, self.device)
        self.predictor = VTWMPredictor(
            dim=self.cfg.model.dim, depth=self.cfg.model.depth, num_heads=self.cfg.model.num_heads,
            mlp_ratio=self.cfg.model.mlp_ratio, num_sensors=d.num_sensors,
            action_dim=self.action_dim, action_chunk=self.action_chunk,
            max_temporal=self.cfg.model.max_temporal, tactile_dim=self.cfg.model.get("tactile_dim", 768),
        ).to(self.device)
        ckpt_path = self.repo / args.get("ckpt", "runs/univtac/predictor.pt")
        sd = torch.load(str(ckpt_path), map_location=self.device)
        self.predictor.load_state_dict(sd["model"])
        if sd.get("tactile") is not None and hasattr(self.tactile, "model"):
            self.tactile.model.load_state_dict(sd["tactile"])
            print("[VTWM] loaded fine-tuned tactile encoder weights")
        self.predictor.eval()
        if hasattr(self.tactile, "model"):
            self.tactile.model.eval()
        print(f"[VTWM] loaded predictor from {ckpt_path}")

        # --- goal latent (encoded reference expert final frame) ------------------------
        self.s_goal = self._build_goal(args.get("goal", {}))
        print(f"[VTWM] goal latent ready: {tuple(self.s_goal.shape)}")

        # rolling state (initialized in reset())
        self.s_hist = []
        self.t_hist = []
        self._tac_hist = {}
        self._tac_bg = {}
        self.reset()

    # --------------------------------------------------------------------------------
    def _build_goal(self, goal_cfg):
        """Encode the final head frame of a reference UniVTAC episode -> (1,16,12,20)."""
        from vtwm.data.univtac_dataset import _decode_image, _resize_hwc, find_task_dirs

        data_root = goal_cfg.get("data_root") or self.cfg.data.data_root
        dirs = find_task_dirs(str(data_root))
        assert dirs, f"no UniVTAC episode store under {data_root}"
        # prefer the dir matching this task; else the first
        episode_dir = next((d for name, d in dirs if name == self.task_name), dirs[0][1])
        files = sorted(
            (f for f in Path(episode_dir).glob("*.hdf5") if f.stat().st_size > 0),
            key=lambda x: int("".join(c for c in x.stem if c.isdigit()) or 0),
        )
        assert files, f"no episodes under {episode_dir}"
        ep = files[int(goal_cfg.get("episode_index", 0)) % len(files)]

        import h5py

        with h5py.File(ep, "r") as h:
            raw = h["observation/head/rgb"][int(goal_cfg.get("frame", -1))]
        # _decode_image already applies BGR2RGB (the model's training domain), so no R/B swap
        # here; just match the training cover-crop resize.
        img = self._rgb_resize(_decode_image(raw), *self.rgb_hw)     # (H,W,3) [0,1] model-domain
        rgb = torch.from_numpy(np.moveaxis(img, -1, 0))[None, None]   # (1,1,3,H,W)
        with torch.no_grad():
            return self.vision.encode(rgb.to(self.device))[:, -1]     # (1,16,12,20)

    # --------------------------------------------------------------------------------
    def reset(self):
        self.s_hist = []
        self.t_hist = []          # rolling per-step tactile latents, aligned 1:1 with s_hist
        self._tac_hist = {}
        self._tac_bg = {}
        self._call_count = 0      # live-step counter for ~6Hz decimation
        self._cur_target = None   # last commanded qpos
        self._prev_plan = None    # last CEM plan (H,chunk,dim), for warm-start
        self._plan = None         # current CEM plan (H,chunk,dim) being executed open-loop

    # --------------------------------------------------------------------------------
    @staticmethod
    def _to_hwc01(img: torch.Tensor) -> np.ndarray:
        """Live obs image tensor -> numpy HWC float in [0,1]."""
        arr = img.detach().cpu().numpy()
        if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
            arr = np.moveaxis(arr, 0, -1)
        arr = arr[..., :3].astype(np.float32, copy=False)
        if arr.size and float(np.nanmax(arr)) > 2.0:
            arr = arr / 255.0
        return np.clip(arr, 0.0, 1.0)

    def _setup_rgb_domain(self, cfg):
        cfg = cfg or {}
        self._rgb_channel_order = str(cfg.get("channel_order", "bgr")).lower()
        self._rgb_match_enabled = bool(cfg.get("match_dataset_stats", False))
        self._rgb_match_source = str(cfg.get("source_stats", "ema")).lower()
        self._rgb_match_decay = float(cfg.get("ema_decay", 0.95))
        self._rgb_match_eps = float(cfg.get("eps", 1e-4))
        self._rgb_source_mean = None
        self._rgb_source_std = None
        self._rgb_target_mean = None
        self._rgb_target_std = None

        if self._rgb_channel_order not in {"rgb", "bgr", "swap_rb"}:
            raise ValueError("rgb_domain.channel_order must be one of: rgb, bgr, swap_rb")
        if self._rgb_match_source not in {"frame", "ema"}:
            raise ValueError("rgb_domain.source_stats must be one of: frame, ema")
        if not self._rgb_match_enabled:
            return

        if "target_mean" in cfg and "target_std" in cfg:
            self._rgb_target_mean = np.asarray(cfg["target_mean"], dtype=np.float32)
            self._rgb_target_std = np.asarray(cfg["target_std"], dtype=np.float32)
        else:
            self._rgb_target_mean, self._rgb_target_std = self._compute_dataset_rgb_stats(
                data_root=cfg.get("data_root") or self.cfg.data.data_root,
                max_episodes=int(cfg.get("max_episodes", 8)),
                max_frames=int(cfg.get("max_frames", 512)),
                sample_stride=int(cfg.get("sample_stride", max(1, self.frame_stride))),
            )
        self._rgb_target_std = np.maximum(self._rgb_target_std, self._rgb_match_eps)
        print("[VTWM] RGB domain adapter: "
              f"channel_order={self._rgb_channel_order}, source={self._rgb_match_source}, "
              f"target_mean={np.round(self._rgb_target_mean, 4).tolist()}, "
              f"target_std={np.round(self._rgb_target_std, 4).tolist()}")

    def _compute_dataset_rgb_stats(self, data_root, max_episodes: int, max_frames: int, sample_stride: int):
        """Estimate post-resize training-domain RGB stats from the UniVTAC HDF5 store."""
        from vtwm.data.univtac_dataset import _decode_image, find_task_dirs
        import h5py

        dirs = find_task_dirs(str(data_root))
        assert dirs, f"no UniVTAC episode store under {data_root}"
        episode_dir = next((d for name, d in dirs if name == self.task_name), dirs[0][1])
        files = sorted(
            (f for f in Path(episode_dir).glob("*.hdf5") if f.stat().st_size > 0),
            key=lambda x: int("".join(c for c in x.stem if c.isdigit()) or 0),
        )[:max_episodes]
        assert files, f"no episodes under {episode_dir}"

        camera = str(self.cfg.data.get("camera", "head"))
        key = f"observation/{camera}/rgb"
        total = 0
        sum_rgb = np.zeros(3, dtype=np.float64)
        sumsq_rgb = np.zeros(3, dtype=np.float64)
        frames = 0
        for ep in files:
            if frames >= max_frames:
                break
            with h5py.File(ep, "r") as h:
                assert key in h, f"{key} missing in {ep}"
                for k in range(0, int(h[key].shape[0]), sample_stride):
                    img = self._rgb_resize(_decode_image(h[key][k]), *self.rgb_hw)
                    flat = img.reshape(-1, 3).astype(np.float64)
                    sum_rgb += flat.sum(axis=0)
                    sumsq_rgb += np.square(flat).sum(axis=0)
                    total += flat.shape[0]
                    frames += 1
                    if frames >= max_frames:
                        break

        mean = sum_rgb / max(1, total)
        var = np.maximum(sumsq_rgb / max(1, total) - np.square(mean), 0.0)
        return mean.astype(np.float32), np.sqrt(var).astype(np.float32)

    def _rgb_to_training_domain(self, img01: np.ndarray) -> np.ndarray:
        if self._rgb_channel_order in {"bgr", "swap_rb"}:
            img01 = img01[..., ::-1]
        return np.ascontiguousarray(img01)

    def _match_rgb_domain(self, img01: np.ndarray) -> np.ndarray:
        if not self._rgb_match_enabled:
            return img01

        flat = img01.reshape(-1, 3)
        frame_mean = flat.mean(axis=0).astype(np.float32)
        frame_std = flat.std(axis=0).astype(np.float32)
        frame_std = np.maximum(frame_std, self._rgb_match_eps)

        if self._rgb_match_source == "frame" or self._rgb_source_mean is None:
            self._rgb_source_mean = frame_mean
            self._rgb_source_std = frame_std
        else:
            d = self._rgb_match_decay
            self._rgb_source_mean = d * self._rgb_source_mean + (1.0 - d) * frame_mean
            self._rgb_source_std = d * self._rgb_source_std + (1.0 - d) * frame_std

        out = (img01 - self._rgb_source_mean) / np.maximum(self._rgb_source_std, self._rgb_match_eps)
        out = out * self._rgb_target_std + self._rgb_target_mean
        return np.ascontiguousarray(np.clip(out, 0.0, 1.0).astype(np.float32))

    def _tactile_keys(self, tactile_obs):
        left = "left_gsmini" if "left_gsmini" in tactile_obs else "left_tactile"
        right = "right_gsmini" if "right_gsmini" in tactile_obs else "right_tactile"
        return [left, right]

    def _sensor_6ch(self, key: str, cur01: np.ndarray) -> np.ndarray:
        """Build the Sparsh 6-channel input for one sensor from the rolling frame history."""
        if key not in self._tac_bg:                  # first frame after reset == background
            self._tac_bg[key] = cur01
            self._tac_hist[key] = []
        hist = self._tac_hist[key]
        hist.append(cur01)
        keep = (self.num_frames - 1) * self.stride + 1
        if len(hist) > keep:
            del hist[:-keep]
        bg = self._tac_bg[key]
        frames = []
        for j in range(self.num_frames):
            idx = max(0, len(hist) - 1 - j * self.stride)
            f = np.clip(hist[idx] - bg + 0.5, 0.0, 1.0)
            f = _resize_hwc_local(f, *self.tactile_hw)
            frames.append(np.moveaxis(f, -1, 0))      # CHW
        return np.concatenate(frames, axis=0).astype(np.float32)  # (6,H,W)

    def encode_obs(self, observation):
        """Live obs -> (rgb (1,1,3,H,W), tactile (1,1,S,6,H,W), qpos (action_dim,))."""
        head = observation["observation"]["head"]["rgb"]
        # Training data was JPEG-encoded through OpenCV from raw env tensors, so live RGB must
        # first be mapped into that saved-data channel convention, then resized and optionally
        # matched to the lift_bottle dataset color statistics before Cosmos encoding.
        img = self._rgb_to_training_domain(self._to_hwc01(head))
        img = self._rgb_resize(img, *self.rgb_hw)
        img = self._match_rgb_domain(img)
        # One-shot RGB debug: dump the raw env obs (true color) alongside the ACTUAL Cosmos input
        # (training domain). Set VTWM_DEBUG_RGB=1 to verify the live model input is the swapped
        # "blue table" domain, not the raw "yellow table" render. Disabled by default (no cost).
        if os.environ.get("VTWM_DEBUG_RGB") and not getattr(self, "_rgb_dumped", False):
            self._rgb_dumped = True
            dbg = Path(os.environ.get("VTWM_DEBUG_RGB_DIR", str(self.repo / "eval_out/rgbcal_live")))
            dbg.mkdir(parents=True, exist_ok=True)

            def _save(p, rgb01):
                cv2.imwrite(str(dbg / p), cv2.cvtColor((np.clip(rgb01, 0, 1) * 255).astype(np.uint8),
                                                       cv2.COLOR_RGB2BGR))
            _save("00_raw_obs_true_color.png", self._to_hwc01(head))   # env render (should be yellow)
            _save("01_model_input_trained_domain.png", img)           # actual Cosmos input (should be blue)
            print(f"[VTWM] RGB debug dumped to {dbg}: 00=raw env obs (true color), "
                  f"01=actual model input (training domain). Mean[ch] input={img.reshape(-1, 3).mean(0).round(3)}")
        rgb = torch.from_numpy(np.ascontiguousarray(np.moveaxis(img, -1, 0)))[None, None].to(self.device)

        tac_obs = observation["tactile"]
        keys = self._tactile_keys(tac_obs)
        # same R/B swap so live tactile matches the _decode_image training domain
        sensors = [self._sensor_6ch(k, np.ascontiguousarray(self._to_hwc01(tac_obs[k]["rgb_marker"])[..., ::-1]))
                   for k in keys]
        tac = torch.from_numpy(np.stack(sensors, 0))[None, None].to(self.device)  # (1,1,S,6,H,W)

        qpos = observation["embodiment"]["joint"][: self.action_dim].to(self.device).float()
        return rgb, tac, qpos

    # --------------------------------------------------------------------------------
    @torch.no_grad()
    def eval(self, task, observation):
        # Cap the episode length for quick smoke runs (see self.max_steps).
        if self.max_steps and getattr(task.cfg, "step_lim", 0) > self.max_steps:
            task.cfg.step_lim = self.max_steps

        # encode_obs advances the per-sensor tactile frame history (live ~60Hz stream) so the
        # 6-channel current-step stack can span ~0.16s. Timing (live steps):
        #   - every `frame_stride` steps == one ~6Hz KEYFRAME: encode latents + append to the
        #     rolling context, so context spacing matches the ~6Hz training rate;
        #   - every `exec_steps` keyframes == one PLAN: run CEM (the heavy part) once and then
        #     open-loop execute the whole H-step plan before replanning. exec_steps defaults to
        #     `horizon` (execute all planned steps); set exec_steps=1 for pure receding-horizon MPC.
        rgb, tac, qpos = self.encode_obs(observation)
        plan_period = self.exec_steps * self.frame_stride   # live steps between CEM replans
        g = self._call_count % plan_period                  # live-step phase within the current plan

        if self._call_count % self.frame_stride == 0:        # at a keyframe -> grow the context
            s_cur = self.vision.encode(rgb)      # (1,1,16,12,20)
            t_cur = self.tactile.encode(tac)     # (1,1,S,196,768)
            self.s_hist.append(s_cur[:, 0])
            self.t_hist.append(t_cur[:, 0])      # tactile context = each step's OWN tactile latent
            if len(self.s_hist) > self.max_context:
                self.s_hist = self.s_hist[-self.max_context:]
                self.t_hist = self.t_hist[-self.max_context:]

        if g == 0:                                           # start of a plan window -> run CEM
            # Single-frame cold start: seed CEM imagination from ONLY the latest latent and let the
            # model autoregress (rollout window grows 1 -> 1+horizon, capped at max_context=9). This
            # matches training's sampling_loss (ctx=1) and val action_mse_eval — the only regime where
            # predicted-frame feedback is actually trained. Feeding the full rolling context here would
            # roll out from a long/sliding window the model never saw, putting CEM's cost out of dist.
            s_ctx = self.s_hist[-1].unsqueeze(1)      # (1,1,16,12,20)
            t_ctx = self.t_hist[-1].unsqueeze(1)      # (1,1,S,196,768)
            # CEM warm-start: seed from the previous plan (executed) so consecutive plans stay
            # coherent. On the first plan there is no prior, so fall back to the current qpos.
            if self._prev_plan is not None:
                mu_init = self._prev_plan
            else:
                mu_init = qpos.view(1, 1, self.action_dim).expand(self.horizon, self.action_chunk, -1)
            # Single-qpos (or prev-plan) seed: every (keyframe, chunk) position drifts from it, so
            # the init std ramps along BOTH the chunk and horizon axes.
            sigma_init = self._qpos_sigma_ramp(
                self.horizon, self.action_chunk, self.action_dim, self.qpos_sigma_step,
                self.frame_stride, device=self.device, chunk_accumulate=True,
            )
            best_action, _ = self._cem_plan(
                self.predictor, s_ctx, t_ctx, self.s_goal,
                horizon=self.horizon, action_chunk=self.action_chunk, action_dim=self.action_dim,
                particles=self.particles, iters=self.iters, elites=self.elites,
                max_context=self.max_context, device=self.device,
                mu_init=mu_init, sigma_init=sigma_init,
            )
            self._prev_plan = best_action.detach()
            self._plan = best_action.to(task.device).float()   # (H, action_chunk, action_dim)

        # Execute the plan open-loop across the window. The g-th live step maps to horizon-step
        # `kf` and, within it, chunk index `ci`. By the dataset convention chunk[i] = joint[k+i+1]
        # (consecutive ~60Hz commands, chunk[-1] ~= the next keyframe), so replaying the chunks in
        # order drives the arm through the planned keyframes instead of crawling to chunk[0] of one
        # step (bug #2) and uses all action_chunk commands the model was conditioned on (bug #3).
        kf = min(self.horizon - 1, g // self.frame_stride)
        sub = g % self.frame_stride
        ci = min(self.action_chunk - 1, (sub * self.action_chunk) // self.frame_stride)
        self._cur_target = self._plan[kf, ci]

        self._call_count += 1
        task.take_action(self._cur_target, action_type="qpos")


def _resize_hwc_local(img: np.ndarray, h: int, w: int) -> np.ndarray:
    if img.shape[0] != h or img.shape[1] != w:
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return img
