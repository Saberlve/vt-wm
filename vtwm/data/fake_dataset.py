"""Fake visuo-tactile dataset with paper-correct raw shapes.

Until a real Digit 360 + exocentric dataset is available, this yields random raw
observations so the full pipeline (frozen encoders -> predictor -> losses) can be
exercised end-to-end. Each item is one trajectory window of T frames at 6 fps.
"""
from __future__ import annotations

from typing import Dict

import torch
from torch.utils.data import Dataset


class FakeVisuoTactileDataset(Dataset):
    def __init__(
        self,
        length: int = 64,
        T: int = 9,
        num_sensors: int = 4,
        action_chunk: int = 5,
        action_dim: int = 7,
        rgb_hw=(192, 320),
        tactile_hw=(224, 224),
        seed: int = 0,
    ):
        self.length = length
        self.T = T
        self.num_sensors = num_sensors
        self.action_chunk = action_chunk
        self.action_dim = action_dim
        self.rgb_hw = tuple(rgb_hw)
        self.tactile_hw = tuple(tactile_hw)
        self.seed = seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        g = torch.Generator().manual_seed(self.seed * 100003 + idx)
        H, W = self.rgb_hw
        th, tw = self.tactile_hw
        # rgb in [0,1] (encoder maps to [-1,1]); tactile raw-ish; actions small deltas.
        rgb = torch.rand(self.T, 3, H, W, generator=g)
        tactile = torch.rand(self.T, self.num_sensors, 6, th, tw, generator=g)
        action = 0.05 * torch.randn(self.T, self.action_chunk, self.action_dim, generator=g)
        return {"rgb": rgb, "tactile": tactile, "action": action}
