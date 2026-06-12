# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**UniVTAC** — a tactile-aware Isaac Sim/Isaac Lab simulation benchmark for contact-rich robotic
manipulation (arXiv:2602.10093). It does three things: **generate** expert demonstrations via
scripted policies + cuRobo motion planning, **train** visuotactile policies, and **evaluate**
them in-sim. Tactile feedback comes from a UIPC-based deformable simulation (TacEx) of GelSight
Mini sensors mounted on the gripper.

**This is a vendored copy inside the parent `vt-wm` repo** (git root is `/home/wangshuxun/VLA/vt-wm`;
this directory is gitignored there). The parent project's world-model work lives under
`policy/VTWM/` and is documented in `../../CLAUDE.md` — that is a *separate concern*. This file
describes the UniVTAC simulation platform itself (data collection + closed-loop eval), which the
VTWM policy plugs into as just another `policy/`.

## Environment & commands

Requires a working **Isaac Sim 4.5 + Isaac Lab 2.1.1** install plus **TacEx built from
`third_party/TacEx`** (the modified local source — *not* the public TacEx) and **cuRobo**. See
`docs/Installation.md`. The conda env is `UniVTAC` (Python 3.10). Nothing here runs without the
Isaac stack on the machine.

```bash
# Collect demos: task_name = a module in envs/; config_name = a *.yml stem in task_config/
bash collect_data.sh ${task_name} ${config_name} ${gpu_id}        # e.g. lift_bottle demo 0
bash parallel_collect.sh ${task_name} ${config_name} ${gpu_id} [num_processes]   # multiprocessing → N Isaac apps

# Closed-loop eval: deploy_config = "${PolicyDir}/${deploy_yml_stem}" under policy/
bash eval_policy.sh ${task_name} ${task_config} ${deploy_config} ${gpu_id}       # e.g. lift_bottle demo ACT/deploy 0
bash parallel_eval.sh ${task_name} ${task_config} ${deploy_config} ${gpu_id} [num_processes] [total_num]
```

There is no test suite or linter. The `.sh` wrappers just set `CUDA_VISIBLE_DEVICES` and call the
real entry points: `scripts/collect_data.py`, `scripts/eval_policy.py`, and their
`parallel_*` siblings. `scripts/replay.py` replays a saved HDF5 episode; `scripts/convert.py`
converts collected data into per-policy training formats; `scripts/visualize.py` renders
episode videos.

## Architecture

Three top-level pieces — **`envs/` (tasks), `scripts/` (drivers), `policy/` (learned policies)** —
glued by config files and dynamic module import.

### Tasks (`envs/`)
Each manipulation task is a module (`lift_bottle.py`, `insert_HDMI.py`, `grasp_classify.py`, …)
that subclasses `BaseTask`/`BaseTaskCfg` from **`envs/_base_task.py`** (the ~1000-line core: an
Isaac Lab `UipcRLEnv` wrapper handling scene setup, stepping, observation capture, HDF5 saving,
video, and the `take_action` execution interface). A task subclass implements only the
task-specific hooks:
- `create_actors()` / `_reset_actors()` — spawn and randomize USD assets (from `assets/`, paths in `envs/_global.py`).
- `pre_move()` — scripted approach (grasp pose selection + a cuRobo `move`).
- `_play_once()` — the scripted expert demonstration (this is what gets recorded).
- `check_success()` / `check_mid_success()` / `check_early_stop()` — pose-based predicates.

Scripted policies are written in a small **Atom DSL** (`envs/utils/atom.py`, accessed as
`self.atom`): high-level primitives like `grasp_actor`, `place_actor`, `move_by_displacement`,
`open_gripper`/`close_gripper` that compile to collision-aware cuRobo trajectories. `self.move(...)`
executes one. Supporting utils: `actor.py` (USD actor + `ActorManager`, contact-point
registration), `transforms.py` (the `Pose` helper with `.add_bias`/`.add_offset`/`.rebase`/
`.to_transformation_matrix`), `data.py` (`HDF5Handler` — the canonical reader/writer for the
episode format), `atom.py`. Robot + sensors live in `envs/robot/` (Franka cfg + `curobo_planner.py`)
and `envs/sensors/` (`camera.py`, `tactile.py`).

### Eval flow (`scripts/eval_policy.py`)
`AppLauncher` **must** start the Omniverse app before any `omni.isaac`/`isaaclab` import — hence
the heavy top-of-file arg parsing. `main()` then: loads `task_config/<cfg>.yml` and
`policy/<deploy>.yml`, **dynamically imports** `envs.<task_name>` and `policy.<policy_name>`
(`policy_name` comes from the deploy yml), instantiates `task_module.Task(cfg, mode='eval')` and
`policy_module.Policy(deploy_config)`. The loop, per seed: with `--expert_check`, first runs the
scripted expert and skips seeds the script itself can't solve (cached in `seeds.json`); then resets,
calls `policy.reset()`, and repeatedly `policy.eval(task, observation)` until `step_lim`, success,
or `check_early_stop()`. Results + videos land in `eval_result/<policy>/<task>/<deploy>/<timestamp>/`.

### Policies (`policy/`)
Each is a self-contained module implementing the `BasePolicy` interface
(`policy/_base_policy.py`): `__init__(args)` (args = the deploy yml merged with runtime
`task_name`/`task_config`), `encode_obs`, `eval(task, observation)`, `reset`, `close`. Inside
`eval`, the policy calls **`task.take_action(action, action_type=...)`** where `action_type` ∈
`{'qpos'` (8 = 7 arm + 1 gripper)`, 'ee'` (8 = pos3 + quat4 + grip)`, 'delta_ee'` (7)`}`. Baselines:
`ACT/` (Action Chunking Transformer, with/without tactile — has its own `train_config_*.yml`,
`process_data.py`, `train.sh`), `Ablation/`, `ViTAL/` (CLIP-pretrained VT encoders), and `VTWM/`
(the parent project's world-model planner). To add a policy, drop a dir under `policy/` with
`deploy_policy.py` + `deploy.yml` (`policy_name` field must equal the dir name) — see `docs/Deploy.md`.

## Config & conventions

- **`task_config/*.yml`** configure *data collection / sim* (sensor type, save/video/render
  frequency, texture randomization, episode count, observation modalities) — read by both collect
  and eval drivers. `demo.yml` is the default; `contact.yml` for the `collect` pretraining task.
- **`policy/<dir>/deploy.yml`** configure a policy at eval time; arbitrary custom fields are passed
  straight through to `Policy.__init__`.
- **`policy/task_settings.json`** maps each task to its eval `camera_type` (`head`/`all`) and `downsample`.
- **Sensor support:** collection/eval are currently GelSight-Mini (`gsmini`) only despite `gf225`/
  `xensews` appearing in configs (see README TODO).

## Data format

One HDF5 file per episode under `data/${config_name}/${task_name}/hdf5/`, plus `video/`,
`metadata.json` (step counts/timing/success), `suc_map.txt` and `scene/` (collection auxiliaries).
Per-step contents: `observation.{head,wrist}.rgb` `(270,480,3)`, `tactile.{left,right}_tactile.{rgb,
depth,marker,rgb_marker,pose}`, `embodiment.{ee(7),joint(9)}`, plus actor poses and `step`/`atom`
metadata. Read/write it through `HDF5Handler` in `envs/utils/data.py`.

## Gotchas

- **Isaac Sim is single-`cuda:0` for rendering** (USD/Fabric scenegraph). The parent VTWM deploy
  works around a Vulkan-render + CUDA-inference deadlock by splitting GPUs — see `../../CLAUDE.md`.
- `parallel_collect` uses Python `multiprocessing`, so it launches **multiple full Isaac Sim apps
  simultaneously** on one GPU — watch VRAM.
- Collection auto-skips failed seeds and resumes from `suc_map.txt`/`seeds.json`; an "episode" means
  a *successful* one (`episode_num` counts successes, not attempts).
- `convert.py` is large and per-policy; regenerate training data through it rather than hand-parsing HDF5.
