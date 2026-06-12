# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

From-scratch reproduction of **"Visuo-Tactile World Models"** (arXiv:2602.06001). No official
code exists; only the two pretrained encoders are reused. The world model predicts future
visual + tactile *latents* given an action sequence, and acts by CEM planning toward a goal
latent. It is a transition model, **not a controller** — there is no action head; actions come
out of planning.

## Environment & commands

The project uses a self-contained `uv` venv (`.venv/`) pinning torch 2.4.1 + cu121. Always
invoke Python as `.venv/bin/python` (or `.venv/bin/torchrun`). `XFORMERS_DISABLED=1` must be
set for nearly everything (the Sparsh backbones import xformers, which we stub out).

```bash
uv sync                                                   # install deps
bash scripts/smoke_test.sh                                # encoder checks -> few train steps -> rollout+CEM (fake data)

# Train (config selects dataset + tactile encoder; see configs/)
.venv/bin/python -m vtwm.train --config configs/univtac_lift_bottle_ada.yaml --resume auto
bash scripts/run_univtac_8gpu.sh                          # 8-GPU A100 DDP (configs/univtac_a100.yaml)
bash scripts/det_train_ada.sh                             # 6-GPU ada-24g DDP (Determined entrypoint)

# Imagination benchmark (rolls out under GROUND-TRUTH actions; PSNR vs frame-freeze baseline)
bash scripts/eval_univtac_lift_bottle.sh                  # wraps `python -m vtwm.eval`

# Open-loop CEM eval (does the planner recover demonstrated actions from a real window?)
bash scripts/openloop_eval_univtac_lift_bottle.sh        # wraps `python -m vtwm.openloop_eval`

# Closed-loop eval in UniVTAC Isaac Sim (CEM policy acting in the sim)
bash scripts/run_univtac_eval.sh --deploy-config VTWM/deploy_lift_bottle --task lift_bottle --log-to-file
bash scripts/det_eval_univtac_lift_bottle_ada.sh         # 2-GPU ada Determined entrypoint (handles Vulkan ICD setup)
```

There is no test suite or linter; `scripts/smoke_test.sh` is the de-facto integration test.
The single-pipeline checks `vtwm.check_encoders` / `vtwm.infer` run against fake data.

## Architecture

Pipeline: `Cosmos (vision, frozen) + Sparsh (touch) → Predictor (transition model) → (s_{k+1}, t_{k+1})`.

- **Vision encoder** (`vtwm/encoders/vision_cosmos.py`) — NVIDIA Cosmos-Tokenize1-CI16x16-360p
  `encoder.jit`, frozen, runs under `no_grad`. RGB `(B,3,192,320)` in `[-1,1]` → latent
  `(B,16,12,20)` = 240 tokens × 16-dim. `eval.py`/`openloop_eval.py` also need the matching
  `decoder.jit` to render latents back to RGB.
- **Tactile encoder** — config-selected via `encoder.tactile`:
  - `sparsh_gelsight` (`sparsh_gelsight.py`, Sparsh DINO ViT-base) for GelSight/UniVTAC; can be
    **fine-tuned** (`encoder.freeze_tactile: false`). Loads `third_party/sparsh-orig`.
  - `sparshx` (`tactile_sparshx.py`, Sparsh-X Digit-360) for the original/fake pipeline. Loads
    the Sparsh-X repo.
  - Per sensor: 6-channel `(B,6,224,224)` → `(B,196,768)`. **The two Sparsh repos both ship a
    `tactile_ssl` package that collides by name — never import both encoders in one process.**
    `vtwm/build.py` imports only the selected one lazily.
- **Predictor** (`vtwm/models/predictor.py`) — transformer. Each block: factorized
  spatio-temporal self-attention (spatial within a timestep, then *causal* temporal across
  timesteps) on sensory AND action tokens, then action-conditioning **cross-attention**
  (sensory queries attend to action keys), then MLPs. RoPE in all attention. `forward` does
  one-step prediction (position k predicts k+1); `rollout` autoregressively imagines `horizon`
  steps, capped by `max_context` (paper = 9).
- **Losses** (`vtwm/losses.py`) — `L_teacher + L_sampling`, both L1 over latents. Sampling loss
  starts from a **single** GT frame and imagines H steps (matching the cold start of
  inference/deploy). **Prediction targets are always `.detach()`'d** — critical when the tactile
  encoder is trainable, so the latent targets don't collapse.
- **CEM planner** (`vtwm/planning/cem.py`) — goal-conditioned, **vision-only** cost (L2 between
  final imagined visual latent and goal latent). Tactile enters only through the initial context.
  `mu_init`/`sigma_init` seed the search: default zero-mean/unit-std (for normalized deltas);
  pass current qpos broadcast + small sigma for an absolute joint-qpos action space.

## Config-driven design

Everything routes through `vtwm/build.py` (`build_vision_encoder`, `build_tactile_encoder`,
`build_dataset`) keyed off the OmegaConf YAML. `data.source` ∈ {`fake`, `univtac`};
`encoder.tactile` ∈ {`sparshx`, `sparsh_gelsight`}. So the dataset and tactile backbone are
swapped purely by config — the predictor/losses/CEM are shared.

Config families in `configs/`:
- `default.yaml` — fake data, Sparsh-X, smoke-scale (depth 4, both encoders frozen).
- `univtac*.yaml` — real UniVTAC HDF5 data, Sparsh-DINO (fine-tuned). `_a100` = 8-GPU training;
  `_*_ada` = ada-24g eval variants with Determined NAS mount paths.

`paths.*` point at external model checkpoints (Cosmos jit, Sparsh DINO safetensors, optional
Sparsh force-field decoder) — on the cluster these live under `/run/determined/NAS1/...`.

## Training details that bite

- `train.batch_size_per_gpu` is **per-GPU**; global batch = per_gpu × world_size (DDP only shards
  indices). DDP uses `find_unused_parameters=True`.
- Vision is encoded **outside** the DDP module (frozen, no_grad); `VTWMTrainModule`
  (`models/train_module.py`) wraps only the trainable tactile encoder + predictor so DDP sees
  the right params. Tactile encoder gets `train.tactile_lr_scale`× the predictor LR.
- Checkpoints are written to `out_dir/step_<n>/checkpoint.pt` (newest `keep_last` kept) plus a
  final `predictor.pt`. `--resume auto` restores the highest-step checkpoint (predictor +
  tactile + optimizer + step). A ckpt dict has keys `model`, `tactile`, `optimizer`, `step`, `config`.
- Precision: `train.precision: bf16` → autocast (no GradScaler; params stay fp32).
- Validation (rank 0): `val/*` imagination losses plus `val/action_mse` — a policy-quality proxy
  that CEM-plans toward the **episode's final** visual latent and MSEs the plan against the
  demonstrated actions. The world model has no action output, so this is how action quality is
  measured.

## UniVTAC dataset specifics (`vtwm/data/univtac_dataset.py`)

- One HDF5 file == one episode, **flat** layout; JPEG byte blobs decoded via cv2. Tactile groups
  are `{left,right}_gsmini` (older files: `{left,right}_tactile`) — both probed.
- 60Hz source: vision is downsampled by `frame_stride` (10 → ~6Hz, paper rate; T=9 ≈ 1.5s
  window). **Actions stay at source rate** — each window step yields an ACT-style chunk of
  `action_chunk` consecutive raw joint-qpos commands (`action_dim` default 8 = 7 arm + 1 gripper).
- Tactile is built **per timestep**: the 6-channel pair is the two most-recent stride-`tactile_stride`
  frames, background-subtracted (`frame - bg + 0.5`).
- See the memory notes (`.claude/.../memory/`) for non-obvious gotchas: dataset frames are stored
  R/B-swapped (live deploy obs must R/B-swap to match), and `insert_HDMI` data covers only the
  final 1.5cm insertion push (too narrow for CEM closed-loop).

## The UniVTAC benchmark (`third_party/UniVTAC/`)

The full **UniVTAC** Isaac Sim / Isaac Lab simulation benchmark (arXiv:2602.10093) is now
vendored in-tree under `third_party/UniVTAC/` (gitignored, `sys.path`-injected — not installed).
It generates scripted expert demos, trains visuotactile policies, and runs closed-loop in-sim
eval; tactile comes from a UIPC/TacEx deformable sim of GelSight-Mini sensors. **It has its own
`third_party/UniVTAC/CLAUDE.md`** — read that for the benchmark's tasks (`envs/`), drivers
(`scripts/`), Atom scripting DSL, HDF5 episode format, and `BasePolicy` interface. The VT-WM
world model plugs in as one policy under `policy/VTWM/`; everything below is that integration.

### Closed-loop deploy (`third_party/UniVTAC/policy/VTWM/deploy_policy.py`)

A `BasePolicy` wrapper run in-process by UniVTAC's Isaac Sim `scripts/eval_policy.py`. It
imports `vtwm` from the repo root (+ `.univtac_pydeps` for vendored pure-python deps), encodes
live obs **exactly mirroring `univtac_dataset.py`** (incl. the R/B swap), and CEM-plans toward a
cached goal latent (the encoded final head-camera frame of a reference expert episode, set in
`deploy_*.yml`).

**GPU split (Isaac Sim constraint):** Isaac's USD/Fabric scenegraph only supports the process's
`cuda:0`, so the **render GPU must be cuda:0**. `run_univtac_eval.sh` exposes
`CUDA_VISIBLE_DEVICES="RENDER_GPU,MODEL_GPU"` → render on cuda:0, VT-WM CEM inference on cuda:1
(`VTWM_DEVICE`). This avoids a single-card Vulkan-render + CUDA-inference driver deadlock. Setting
model==render GPU falls back to single-card and reintroduces the deadlock risk. The Determined
entrypoints (`det_eval_*`) additionally synthesize a Vulkan ICD pointing at `libEGL_nvidia.so.0`
because the system ICD fails `vkCreateInstance` in the headless container.

`third_party/` (UniVTAC, sparsh-orig, vjepa2, curobo, IsaacLab) is gitignored and added to
`sys.path` at runtime rather than installed. The UniVTAC episode HDF5 produced by the benchmark's
collection drivers is what `vtwm/data/univtac_dataset.py` consumes for training.
