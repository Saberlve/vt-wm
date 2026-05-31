# VT-WM — Visuo-Tactile World Model (reproduction)

From-scratch reproduction of **"Visuo-Tactile World Models"** (arXiv:2602.06001).
No official code exists; only the two pretrained encoders are reused.

```
Cosmos (vision, frozen) ┐
                        ├─► Predictor (transition model) ─► (s_{k+1}, t_{k+1})
Sparsh-X (touch, frozen)┘        ▲ action conditioning
```

## Components
- **Vision encoder** — NVIDIA Cosmos-Tokenize1-CI16x16-360p (`encoder.jit`), frozen.
  RGB `(B,3,192,320)` → latent `(B,16,12,20)` = 240 tokens × 16-dim per frame.
- **Tactile encoder** — Sparsh-X img base (`dit_base`, vanilla fusion), frozen.
  Per Digit 360 sensor: `(B,6,224,224)` → `(B,196,768)`. 4 sensors.
  Loaded directly from the cloned `third_party/sparsh-multisensory-touch` code
  (`vtwm/encoders/tactile_sparshx.py`), `strict=True`.
- **Predictor** (`vtwm/models/predictor.py`) — transformer with per-block factorized
  spatio-temporal self-attention + action-conditioning cross-attention, RoPE, causal
  temporal attention, modality output heads. Trained with teacher-forcing + autoregressive
  sampling L1 loss (`vtwm/losses.py`). CEM planner in `vtwm/planning/cem.py`.

## Setup
```bash
cd vt-wm
uv sync                # installs torch cu121 + deps
# Sparsh-X repo is already cloned under third_party/ (added to sys.path at runtime)
```

## Run the smoke test (fake data, Sparsh-X / Digit 360)
```bash
bash scripts/smoke_test.sh        # encoder checks -> few train steps -> rollout + CEM
```
Individual steps:
```bash
.venv/bin/python -m vtwm.check_encoders --config configs/default.yaml
.venv/bin/python -m vtwm.train         --config configs/default.yaml
.venv/bin/python -m vtwm.infer         --config configs/default.yaml --ckpt ./runs/smoke/predictor.pt
```

## Train on the ManiFeel dataset (GelSight / Sparsh-DINO)
ManiFeel is a GelSight visuo-tactile manipulation dataset (diffusion-policy zarr
ReplayBuffer). We encode its GelSight (Taxim-rendered) tactile images with the original
**Sparsh DINO ViT-base** (`facebook/sparsh-dino-base`), the in-domain choice for
GelSight/DIGIT (Sparsh-X is Digit-360-only). Config: `configs/manifeel.yaml`.

ManiFeel realities (vs the paper, handled by the loader): **2 GelSight sensors**
(left+right finger); exocentric camera key varies per task (`rgb_key` is a candidate
list `[front,side,wrist,wrist_2]`, first present); **action dim varies 6/7** across
tasks and is padded to 7 (`data.action_dim`); chunk size 1.

`configs/manifeel.yaml`'s `data.zarr_path` may be a single task store **or** the
`extracted/` root containing all task subdirs — the latter trains across all 9 tasks
(multi-task), skipping any incomplete/incompatible store.

```bash
# data: HF purdue-mars/manifeel (9 per-task .zip of zarr stores); unzip into extracted/, then:
.venv/bin/python -m vtwm.data.inspect_zarr <extracted_task_dir>      # inspect schema
.venv/bin/python -m vtwm.train --config configs/manifeel.yaml        # multi-task over all tasks
.venv/bin/python -m vtwm.train --config configs/manifeel.yaml --resume auto   # resume from last.pt
.venv/bin/python -m vtwm.infer --config configs/manifeel.yaml --ckpt ./runs/manifeel/predictor.pt
```

### Training features (configs/manifeel.yaml)
- **Fine-tune the tactile encoder**: `encoder.freeze_tactile: false` makes Sparsh-DINO
  trainable (paper fine-tunes the tactile encoder; Cosmos stays frozen) at
  `train.tactile_lr_scale`× the predictor LR. Prediction **targets are stop-gradient**
  (`losses.py`) so trainable-encoder targets don't collapse. Trainable params ≈ 96M
  (predictor 9.9M + Sparsh-DINO 86.3M).
- **Validation**: held-out episode split (`data.val_ratio`); eval every `train.eval_every`
  steps over `train.val_batches` batches → `val/*` metrics.
- **Checkpoint/resume**: `last.pt` written every `train.save_every`; `--resume auto`
  restores predictor + tactile encoder + optimizer + step.
- **wandb curves**: `wandb.enabled: true` logs `train/{loss,teacher,sampling,lr,grad_norm}`
  and `val/*`. `mode: online` needs network — enable the clash proxy first
  (`source ~/.bashrc; clashon`); use `mode: offline` otherwise (sync later with `wandb sync`).

Both pipelines share the predictor/losses/CEM; the tactile encoder and dataset are
selected by config (`encoder.tactile`, `data.source`) via `vtwm/build.py`. The two Sparsh
repos under `third_party/` both ship a `tactile_ssl` package, so only the selected encoder
is imported per process.

## Deliberate deviations from the paper (first runnable version)
See `configs/default.yaml`. These match what was agreed for "get it running first":
- Predictor depth **4** (paper: 12).
- **Both encoders frozen** (paper fine-tunes Sparsh-X; Cosmos frozen).
- Action representation unified to **7-dim** `[dx,dy,dz,droll,dpitch,dyaw,gripper]`
  (paper trains on quaternion+hand = 8-dim); `data.action_dim` is config-swappable.
- **Fake** data (`vtwm/data/fake_dataset.py`) with paper-correct raw shapes until a real
  Digit 360 + exocentric dataset is available.
- Smoke-scale `dim`, batch, steps, CEM particles.

To move toward the full paper: set `model.depth: 12`, raise `train.steps`/`warmup_steps`
to 80k/10k, swap in a real dataset, and (optionally) unfreeze Sparsh-X.
