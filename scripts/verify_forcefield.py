"""Smoke-test the Sparsh GelSight force-field decoder on a real tactile frame.

Loads our existing Sparsh-DINO encoder, registers DPT hooks on blocks [2,5,8,11],
loads the downloaded force-field decoder weights, and decodes one GT tactile frame
from the lift_bottle dataset into a (normal, shear) force field. Saves a side-by-side
PNG so we can confirm the decoder produces a sensible physical readout.
"""
import os
import sys

import numpy as np
import torch
import cv2
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# tactile_ssl.downstream_task.__init__ pulls in dinov2 -> xformers (unused when
# XFORMERS_DISABLED=1). Stub it so the force-field decoder import doesn't fail.
os.environ.setdefault("XFORMERS_DISABLED", "1")
import types  # noqa: E402
if "xformers" not in sys.modules:
    _xf = types.ModuleType("xformers")
    _ops = types.ModuleType("xformers.ops")
    _ops.fmha = types.ModuleType("xformers.ops.fmha")
    _xf.ops = _ops
    sys.modules["xformers"] = _xf
    sys.modules["xformers.ops"] = _ops
    sys.modules["xformers.ops.fmha"] = _ops.fmha

from vtwm.build import build_dataset, build_tactile_encoder  # noqa: E402

DECODER_CKPT = (
    "/run/determined/NAS1/public/wangshuxun/models/sparsh/"
    "sparsh-gelsight-forcefield-decoder/gelsight_t1_forcefield_dino_vitbase_bg/"
    "checkpoints/epoch-0021.pth"
)
HOOKS = [2, 5, 8, 11]
_REPO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "third_party", "sparsh-orig")


def _import_forcefield_decoder():
    """Import ForceFieldDecoder without running the downstream_task package __init__
    (which pulls in slip/pose/grasp modules needing sklearn/xformers we don't have).

    Registering a stub package module with the right __path__ makes Python import the
    forcefield_sl submodule directly from disk instead of executing __init__.py.
    """
    import importlib
    import tactile_ssl  # noqa: F401
    pkg = "tactile_ssl.downstream_task"
    if pkg not in sys.modules:
        d = os.path.join(_REPO, "tactile_ssl", "downstream_task")
        m = types.ModuleType(pkg)
        m.__path__ = [d]
        m.__package__ = pkg
        sys.modules[pkg] = m
    mod = importlib.import_module(f"{pkg}.forcefield_sl")
    return mod.ForceFieldDecoder


def main():
    cfg = OmegaConf.load("configs/univtac_lift_bottle_ada.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.device = device

    tactile = build_tactile_encoder(cfg, device)
    enc = tactile.model  # vit_base, in_chans=6, num_register_tokens=1
    enc.eval()

    # DPT hooks: capture each block's full token output.
    acts = {}

    def mk(name):
        def hook(_m, _i, out):
            acts[name] = out
        return hook

    for h in HOOKS:
        enc.blocks[h].register_forward_hook(mk(f"t{h}"))

    # Build the force-field decoder and load weights (strip model_task. prefix).
    ForceFieldDecoderSL = _import_forcefield_decoder()
    decoder = ForceFieldDecoderSL(embed_dim="base").to(device).eval()
    sd = torch.load(DECODER_CKPT, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    sd = {k[len("model_task."):]: v for k, v in sd.items() if k.startswith("model_task.")}
    missing, unexpected = decoder.load_state_dict(sd, strict=False)
    print(f"[decoder] loaded; missing={len(missing)} unexpected={len(unexpected)}")
    assert not missing, missing[:8]

    # Grab one real tactile frame from the dataset.
    ds = build_dataset(cfg, val=True)
    item = ds[0]
    tac = item["tactile"]  # (T, S, 6, 224, 224)
    print(f"[data] tactile item shape {tuple(tac.shape)}")
    x = tac[0, 0].unsqueeze(0).to(device)  # (1, 6, 224, 224) frame 0, sensor 0

    from torchvision.utils import flow_to_image

    def decode(hook_map):
        out = decoder({k: v.clone() for k, v in hook_map.items()}, mode="normal_shear")
        return out["normal"], out["shear"]

    with torch.no_grad():
        acts.clear()
        feats = enc.forward_features(x)
        final = feats["x_norm_patchtokens"]            # (1,196,768) what the world model uses
        proper = {f"t{h}": acts[f"t{h}"] for h in HOOKS}
        # DEGENERATE path: feed the final-layer latent to all 4 DPT hooks (what we'd be
        # forced to do for the predicted latent, which lacks intermediate activations).
        degen = {f"t{h}": final for h in HOOKS}
        n_p, s_p = decode(proper)
        n_d, s_d = decode(degen)
    nd_corr = float(torch.corrcoef(torch.stack([n_p.flatten(), n_d.flatten()]))[0, 1])
    print(f"[proper ] normal range[{float(n_p.min()):.3f},{float(n_p.max()):.3f}] "
          f"shear range[{float(s_p.min()):.2f},{float(s_p.max()):.2f}]")
    print(f"[degen  ] normal range[{float(n_d.min()):.3f},{float(n_d.max()):.3f}] "
          f"shear range[{float(s_d.min()):.2f},{float(s_d.max()):.2f}]")
    print(f"[compare] normal-field corr(proper, degenerate) = {nd_corr:.3f}")

    def to_img(t):  # (3,H,W) any range -> uint8 HWC RGB
        t = t.detach().float().cpu()
        t = (t - t.min()) / (t.max() - t.min() + 1e-8)
        return (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    def nrm_img(n):
        n = n[0].repeat(3, 1, 1)
        n = (n - n.min()) / (n.max() - n.min() + 1e-8)
        h = cv2.applyColorMap((n[0].cpu().numpy() * 255).astype(np.uint8), cv2.COLORMAP_JET)
        return cv2.cvtColor(h, cv2.COLOR_BGR2RGB)

    def shr_img(s):
        return flow_to_image(s[0].cpu()).permute(1, 2, 0).numpy().astype(np.uint8)

    inp = to_img(x[0, 3:6])                 # current frame = last 3 of 6 channels
    H = inp.shape[0]
    row_proper = [cv2.resize(p, (H, H)) for p in (inp, nrm_img(n_p), shr_img(s_p))]
    row_degen = [cv2.resize(p, (H, H)) for p in (inp, nrm_img(n_d), shr_img(s_d))]
    grid = np.concatenate([np.concatenate(row_proper, axis=1),
                           np.concatenate(row_degen, axis=1)], axis=0)
    os.makedirs("eval_out", exist_ok=True)
    out_path = "eval_out/forcefield_smoke.png"
    cv2.imwrite(out_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"[done] wrote {out_path}  rows: [proper 4-hook] / [degenerate final-only]"
          f" ; cols: input | normal | shear")


if __name__ == "__main__":
    main()
