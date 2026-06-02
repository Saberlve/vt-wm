"""No-simulator smoke test for the VTWM closed-loop policy.

Exercises the full policy path that does NOT need Isaac Sim: load config + encoders +
predictor checkpoint, build the goal latent, then run reset() + a few eval() steps against a
synthetic observation and a mock task. Verifies an 8-dim joint qpos is produced each step.

Run from the UniVTAC checkout with vt-wm on PYTHONPATH, e.g.:
  cd third_party/UniVTAC
  PYTHONPATH=<vt-wm-root>:<vt-wm-root>/.univtac_pydeps \
    /home/wangshuxun/miniconda3/envs/UniVTAC/bin/python ../../scripts/vtwm_policy_smoke.py
"""
import os
import importlib
import yaml
import torch
from pathlib import Path


class MockTask:
    def __init__(self, device):
        self.device = device
        self.calls = []

    def take_action(self, action, action_type="qpos", **kw):
        self.calls.append((action, action_type))
        assert action.shape[-1] == 8, f"expected 8-dim qpos, got {tuple(action.shape)}"
        assert str(action.device).startswith(self.device.split(":")[0])
        return True, False


def fake_obs(device):
    j = torch.zeros(9)
    j[:7] = torch.tensor([0.0, -0.3, 0.0, -1.8, 0.0, 1.5, 0.7])  # plausible arm pose
    j[7] = 0.04
    return {
        "observation": {"head": {"rgb": torch.randint(0, 256, (270, 480, 3)).float().to(device)}},
        "tactile": {
            "left_gsmini": {"rgb_marker": torch.randint(0, 256, (240, 320, 3)).float().to(device)},
            "right_gsmini": {"rgb_marker": torch.randint(0, 256, (240, 320, 3)).float().to(device)},
        },
        "embodiment": {"joint": j.to(device)},
    }


def main():
    univtac = Path(__file__).resolve().parents[1] / "third_party" / "UniVTAC"
    deploy_yml = univtac / "policy" / "VTWM" / "deploy.yml"
    args = yaml.safe_load(deploy_yml.read_text())
    args.update({"task_name": "grasp_classify", "task_config": "demo"})
    # No-sim test: no Isaac Sim renderer to contend with, so the two-GPU split is irrelevant
    # here. Pin to a single visible card (cuda:0 by default) instead of the deploy.yml cuda:1,
    # so this runs on a one-GPU box. VTWM_DEVICE still overrides if you want a specific card.
    args["device"] = os.environ.get("VTWM_DEVICE", "cuda:0")

    mod = importlib.import_module("policy.VTWM")  # requires cwd/sys.path == UniVTAC root
    policy = mod.Policy(args)
    device = policy.device

    task = MockTask(device)
    policy.reset()
    for step in range(3):
        policy.eval(task, fake_obs(device))
        a = task.calls[-1][0]
        print(f"[smoke] step {step}: qpos action {tuple(a.shape)} "
              f"min={a.min().item():.3f} max={a.max().item():.3f}")
    print(f"[smoke] OK — {len(task.calls)} actions taken, context len={len(policy.s_hist)}")


if __name__ == "__main__":
    main()
