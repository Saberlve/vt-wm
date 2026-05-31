"""ManiFeel visuo-tactile dataset for VT-WM.

Reads a ManiFeel task store (diffusion-policy ReplayBuffer zarr) and yields trajectory
windows shaped like the rest of the pipeline expects, but adapted to ManiFeel's setup
(1 GelSight tactile sensor, single exocentric camera, 6-dim action):

  rgb     : (T, 3, *rgb_hw)            exocentric `front` camera, [0,1]
  tactile : (T, 1, 6, *tactile_hw)     GelSight `right_tactile_camera_taxim`, Sparsh-style
                                       6-channel = two stride-`tactile_stride` frames,
                                       background-subtracted (compute_diff, offset 0.5)
  action  : (T, 1, 6)                  per-step 6-dim action (chunk size 1)

Vision is resized to `rgb_hw` (default 192x320) to reuse the Cosmos latent geometry.
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

from torch.utils.data import ConcatDataset

from .inspect_zarr import find_zarr_root


def find_task_zarrs(root: str) -> List[str]:
    """Return the list of per-task zarr roots under `root` (each a ManiFeel task store).

    If `root` is itself a single zarr store, returns just [root].
    """
    if os.path.isdir(root) and os.path.exists(os.path.join(root, ".zgroup")):
        return [root]
    tasks = []
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            continue
        zr = find_zarr_root(d)
        if os.path.exists(os.path.join(zr, ".zgroup")):
            tasks.append(zr)
    return tasks


def _to_float_hwc(arr: np.ndarray) -> np.ndarray:
    """Return HWC float image in [0,1] from a stored uint8/float frame."""
    x = np.asarray(arr)
    if x.dtype == np.uint8:
        x = x.astype(np.float32) / 255.0
    else:
        x = x.astype(np.float32)
        if x.max() > 1.5:  # stored as 0..255 floats
            x = x / 255.0
    return x


def _resize_hwc(img: np.ndarray, h: int, w: int) -> np.ndarray:
    import cv2

    if img.shape[0] != h or img.shape[1] != w:
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return img


class ManiFeelDataset(Dataset):
    def __init__(
        self,
        zarr_path: str,
        T: int = 9,
        rgb_key=("front", "side", "wrist", "wrist_2"),
        tactile_keys=("left_tactile_camera_taxim", "right_tactile_camera_taxim"),
        action_key: str = "action",
        rgb_hw=(192, 320),
        tactile_hw=(224, 224),
        tactile_num_frames: int = 2,
        tactile_stride: int = 5,
        remove_bg: bool = True,
        action_dim: int = None,
        val: bool = False,
        val_ratio: float = 0.02,
        seed: int = 42,
        task_name: str = None,
    ):
        self.task_name = task_name or os.path.basename(os.path.normpath(zarr_path))
        root_path = find_zarr_root(zarr_path)
        assert os.path.isdir(root_path), f"not a zarr dir: {root_path}"
        self.root = zarr.open(root_path, "r")
        self.data = self.root["data"]
        # rgb_key may be a single key or a candidate list; pick the first present.
        rgb_candidates = [rgb_key] if isinstance(rgb_key, str) else list(rgb_key)
        rgb_key = next((k for k in rgb_candidates if k in self.data), None)
        assert rgb_key is not None, f"none of {rgb_candidates} in {list(self.data.keys())}"
        self.tactile_keys = list(tactile_keys)
        for tk in self.tactile_keys:
            assert tk in self.data, f"{tk} not in {list(self.data.keys())}"

        ends = np.asarray(self.root["meta"]["episode_ends"][:]).astype(np.int64)
        starts = np.concatenate([[0], ends[:-1]])
        self.episodes: List[Tuple[int, int]] = list(zip(starts.tolist(), ends.tolist()))

        # train/val split over episodes
        rng = np.random.default_rng(seed)
        n = len(self.episodes)
        n_val = max(1, int(round(n * val_ratio))) if val_ratio > 0 else 0
        perm = rng.permutation(n)
        val_idx = set(perm[:n_val].tolist())
        keep = [i for i in range(n) if (i in val_idx) == val]

        self.T = T
        self.rgb_key, self.action_key = rgb_key, action_key
        self.rgb_hw = tuple(rgb_hw)
        self.tactile_hw = tuple(tactile_hw)
        self.tactile_num_frames = tactile_num_frames
        self.tactile_stride = tactile_stride
        self.remove_bg = remove_bg
        # Tasks vary between 6- and 7-dim actions; pad/truncate to a fixed width so the
        # predictor's action projector is consistent across multi-task batches.
        self.action_dim = action_dim

        # window index: (episode_idx, start_frame) with a full T-window inside the episode
        self.index: List[Tuple[int, int]] = []
        for ei in keep:
            s, e = self.episodes[ei]
            length = e - s
            for st in range(0, max(0, length - T + 1)):
                self.index.append((ei, s + st))
        if not self.index:  # episodes shorter than T -> at least one padded window each
            for ei in keep:
                self.index.append((ei, self.episodes[ei][0]))

    def __len__(self) -> int:
        return len(self.index)

    def _tactile_frame_6ch(self, key: str, ep_start: int, k: int, bg: np.ndarray) -> np.ndarray:
        """Build the Sparsh-style multi-frame, background-subtracted tactile input at step k."""
        frames = []
        for j in range(self.tactile_num_frames):
            idx = max(ep_start, k - j * self.tactile_stride)
            img = _to_float_hwc(self.data[key][idx])  # HWC [0,1]
            if self.remove_bg:
                img = np.clip(img - bg + 0.5, 0.0, 1.0)
            img = _resize_hwc(img, *self.tactile_hw)
            frames.append(np.moveaxis(img, -1, 0))  # CHW
        return np.concatenate(frames, axis=0).astype(np.float32)  # (3*num_frames, H, W)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        ei, start = self.index[i]
        ep_start, ep_end = self.episodes[ei]
        T = self.T
        idxs = [min(ep_end - 1, start + t) for t in range(T)]  # clamp-pad short episodes

        # vision
        rgb = []
        for k in idxs:
            img = _to_float_hwc(self.data[self.rgb_key][k])
            img = _resize_hwc(img, *self.rgb_hw)
            rgb.append(np.moveaxis(img, -1, 0))
        rgb = np.stack(rgb, 0).astype(np.float32)  # (T,3,H,W)

        # tactile (S sensors: left + right GelSight) -> (T, S, 6, H, W)
        tac_sensors = []
        for tk in self.tactile_keys:
            bg = _to_float_hwc(self.data[tk][ep_start]) if self.remove_bg else None
            frames = np.stack([self._tactile_frame_6ch(tk, ep_start, k, bg) for k in idxs], 0)  # (T,6,H,W)
            tac_sensors.append(frames)
        tac = np.stack(tac_sensors, axis=1)  # (T, S, 6, H, W)

        # action (chunk size 1), padded/truncated to a fixed width across tasks
        act = np.stack([np.asarray(self.data[self.action_key][k], dtype=np.float32) for k in idxs], 0)
        if self.action_dim is not None:
            raw = act.shape[-1]
            if raw < self.action_dim:
                act = np.pad(act, ((0, 0), (0, self.action_dim - raw)))
            elif raw > self.action_dim:
                act = act[:, : self.action_dim]
        act = act[:, None]  # (T,1,A)

        return {
            "rgb": torch.from_numpy(rgb),
            "tactile": torch.from_numpy(tac),
            "action": torch.from_numpy(act),
            # metadata (ignored by training, used by eval to label videos)
            "task": self.task_name,
            "episode": int(ei),
            "frame0": int(start),
            "cam": self.rgb_key,
        }


def make_manifeel_dataset(zarr_path: str, val: bool = False, **kwargs):
    """Build a single- or multi-task ManiFeel dataset.

    `zarr_path` may be a single task store or a root dir containing several task
    subdirs (e.g. the `extracted/` folder); in the latter case all tasks are
    concatenated for multi-task training.
    """
    tasks = find_task_zarrs(zarr_path)
    assert tasks, f"no ManiFeel task zarr stores found under {zarr_path}"
    if len(tasks) == 1:
        return ManiFeelDataset(tasks[0], val=val, **kwargs)
    # Skip stores that are incomplete (extraction in progress) or have an incompatible
    # schema (e.g. missing a camera/tactile key), so multi-task training is robust.
    subsets, used = [], []
    root = os.path.normpath(zarr_path)
    for t in tasks:
        try:
            # name the task by its sub-path under the dataset root (the task dir is itself
            # the zarr store here, so basename(dirname) would collapse to the root name).
            rel = os.path.relpath(os.path.normpath(t), root)
            tname = rel.split(os.sep)[0] if rel not in (".", "") else os.path.basename(root)
            subsets.append(ManiFeelDataset(t, val=val, task_name=tname, **kwargs))
            used.append(t)
        except Exception as e:  # noqa: BLE001
            print(f"[manifeel] skip {os.path.basename(t)}: {type(e).__name__}: {str(e)[:80]}")
    assert subsets, f"no usable ManiFeel tasks under {zarr_path}"
    ds = ConcatDataset(subsets)
    ds.task_paths = used  # for introspection/logging
    return ds
