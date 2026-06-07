"""Config-driven builders for the dataset and tactile encoder.

The two tactile encoders live in repos that both ship a `tactile_ssl` package, so we
import only the one selected by config (never both in the same process).
"""
from __future__ import annotations

from torch.utils.data import Dataset

from vtwm.encoders.vision_cosmos import CosmosVisionEncoder


def build_vision_encoder(cfg, device):
    return CosmosVisionEncoder(cfg.paths.cosmos_encoder_jit, device=device)


def build_tactile_encoder(cfg, device):
    kind = cfg.encoder.tactile
    freeze = cfg.encoder.get("freeze_tactile", True)
    if kind == "sparshx":
        from vtwm.encoders.tactile_sparshx import SparshXTactileEncoder

        return SparshXTactileEncoder(cfg.paths.sparshx_ckpt, device=device)
    if kind == "sparsh_gelsight":
        from vtwm.encoders.sparsh_gelsight import SparshGelSightEncoder

        return SparshGelSightEncoder(cfg.paths.sparsh_dino_ckpt, device=device, freeze=freeze)
    raise ValueError(f"unknown tactile encoder: {kind}")


def build_dataset(cfg, val: bool = False) -> Dataset:
    src = cfg.data.source
    if src == "fake":
        from vtwm.data.fake_dataset import FakeVisuoTactileDataset

        return FakeVisuoTactileDataset(
            length=cfg.data.dataset_len, T=cfg.data.T, num_sensors=cfg.data.num_sensors,
            action_chunk=cfg.data.action_chunk, action_dim=cfg.data.action_dim,
            rgb_hw=cfg.data.rgb_hw, tactile_hw=cfg.data.tactile_hw, seed=cfg.train.seed,
        )
    if src == "manifeel":
        from vtwm.data.manifeel_dataset import make_manifeel_dataset

        return make_manifeel_dataset(
            zarr_path=cfg.data.zarr_path, val=val, T=cfg.data.T,
            rgb_key=cfg.data.rgb_key, tactile_keys=list(cfg.data.tactile_keys),
            action_key=cfg.data.action_key, rgb_hw=cfg.data.rgb_hw, tactile_hw=cfg.data.tactile_hw,
            tactile_num_frames=cfg.data.tactile_num_frames, tactile_stride=cfg.data.tactile_stride,
            action_dim=cfg.data.action_dim, val_ratio=cfg.data.get("val_ratio", 0.02), seed=cfg.train.seed,
        )
    if src == "univtac":
        from vtwm.data.univtac_dataset import make_univtac_dataset

        return make_univtac_dataset(
            data_root=cfg.data.data_root, val=val, T=cfg.data.T,
            frame_stride=cfg.data.get("frame_stride", 1),
            camera=cfg.data.get("camera", "head"),
            tactile_keys=(list(cfg.data.tactile_keys) if cfg.data.get("tactile_keys", None) else None),
            tactile_image=cfg.data.get("tactile_image", "rgb_marker"),
            action_dim=cfg.data.action_dim, action_chunk=cfg.data.action_chunk,
            rgb_hw=cfg.data.rgb_hw, tactile_hw=cfg.data.tactile_hw,
            tactile_num_frames=cfg.data.tactile_num_frames, tactile_stride=cfg.data.tactile_stride,
            val_ratio=cfg.data.get("val_ratio", 0.05), seed=cfg.train.seed,
            max_episodes=cfg.data.get("max_episodes", None),
        )
    raise ValueError(f"unknown data source: {src}")
