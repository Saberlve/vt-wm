"""UniVTAC visuo-tactile dataset for VT-WM.

Reads UniVTAC episode HDF5 files (the `byml/UniVTAC` benchmark, GelSight Mini sensors)
and yields trajectory windows shaped exactly like the rest of the VT-WM pipeline expects
(same contract as `manifeel_dataset.ManiFeelDataset`):

  rgb     : (T, 3, *rgb_hw)            head camera RGB, [0,1]
  tactile : (T, 2, 6, *tactile_hw)     left+right GelSight `rgb_marker`, Sparsh-style
                                       6-channel = two stride-`tactile_stride` frames,
                                       background-subtracted (frame - bg + 0.5)
  action  : (T, 1, action_dim)         next-step joint qpos command (chunk size 1)

On-disk layout (one file == one episode, FLAT not nested), each field a length-T array:

  step              (T,)  int64          sim step counter
  observation/head/rgb   (T,)  |S<n>     JPEG-encoded bytes  -> cv2.imdecode -> (270,480,3) BGR
  observation/wrist/rgb  (T,)  |S<n>     JPEG-encoded bytes
  tactile/{left,right}_gsmini/rgb_marker (T,) |S<n>  JPEG bytes -> (240,320,3)
  tactile/{left,right}_gsmini/{rgb,depth,marker,pose}
  embodiment/joint  (T, 9)  float32      7 arm + 2 finger
  embodiment/ee     (T, 7)  float32      pose
  actor/<obj>       (T, 7), atom/{id,tag}

(Older/contact-collection files may use `left_tactile`/`right_tactile` instead of
`*_gsmini`; both are probed.)

Action follows UniVTAC's own convention (policy/ACT/process_data.py): the commanded
joint qpos for the *next* step, padded/truncated to `action_dim` (default 8 = 7 arm + 1 gripper).
"""
from __future__ import annotations

import glob
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

# Candidate tactile group names, in priority order (file picks whichever is present).
_TACTILE_NAME_SETS = (
    ("left_gsmini", "right_gsmini"),
    ("left_tactile", "right_tactile"),
)


def _decode_image(raw) -> np.ndarray:
    """Decode a JPEG/PNG byte blob (stored as np.bytes_) to an RGB float image in [0,1]."""
    import cv2

    buf = np.frombuffer(bytes(raw), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # BGR uint8 (H,W,3)
    if img is None:
        raise ValueError("cv2.imdecode failed on a UniVTAC image blob")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def _resize_hwc(img: np.ndarray, h: int, w: int) -> np.ndarray:
    import cv2

    if img.shape[0] != h or img.shape[1] != w:
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return img


def find_task_dirs(root: str) -> List[Tuple[str, str]]:
    """Return [(task_name, episode_dir)] for UniVTAC episode stores under `root`.

    Accepts either a directory of `*.hdf5` files (a single `{task}/{config}` store) or a
    parent containing several `{task}/{config}` subdirs (multi-task).
    """
    root = os.path.normpath(root)
    if glob.glob(os.path.join(root, "*.hdf5")):
        # `root` is itself an episode dir; name the task after its parent (the task dir).
        return [(os.path.basename(os.path.dirname(root)) or os.path.basename(root), root)]
    found: List[Tuple[str, str]] = []
    for task in sorted(os.listdir(root)):
        tdir = os.path.join(root, task)
        if not os.path.isdir(tdir):
            continue
        # episodes may sit directly under the task dir or under a config subdir (e.g. `clean`)
        if glob.glob(os.path.join(tdir, "*.hdf5")):
            found.append((task, tdir))
            continue
        for cfg in sorted(os.listdir(tdir)):
            cdir = os.path.join(tdir, cfg)
            if os.path.isdir(cdir) and glob.glob(os.path.join(cdir, "*.hdf5")):
                found.append((task, cdir))
    return found


class UniVTACDataset(Dataset):
    def __init__(
        self,
        episode_dir: str,
        T: int = 9,
        camera: str = "head",
        tactile_keys: Optional[Tuple[str, ...]] = None,
        tactile_image: str = "rgb_marker",
        action_dim: int = 8,
        rgb_hw=(192, 320),
        tactile_hw=(224, 224),
        tactile_num_frames: int = 2,
        tactile_stride: int = 5,
        remove_bg: bool = True,
        val: bool = False,
        val_ratio: float = 0.05,
        seed: int = 42,
        task_name: str = None,
        max_episodes: int = None,
    ):
        import h5py

        self.task_name = task_name or os.path.basename(os.path.dirname(os.path.normpath(episode_dir)))
        files = sorted(
            glob.glob(os.path.join(episode_dir, "*.hdf5")),
            key=lambda x: int("".join(c for c in os.path.basename(x).split(".")[0] if c.isdigit()) or 0),
        )
        files = [f for f in files if os.path.getsize(f) > 0]  # drop zero-byte / partial files
        assert files, f"no episode hdf5 files under {episode_dir}"
        if max_episodes is not None:
            files = files[:max_episodes]

        self.camera = camera
        self.tactile_image = tactile_image
        self.action_dim = action_dim
        self.rgb_hw = tuple(rgb_hw)
        self.tactile_hw = tuple(tactile_hw)
        self.tactile_num_frames = tactile_num_frames
        self.tactile_stride = tactile_stride
        self.remove_bg = remove_bg
        self.T = T
        self._h5: Dict[str, "h5py.File"] = {}  # lazily-opened per-worker file handles

        # episode lengths (read once up front) + resolve which schema this store uses
        self.episodes: List[Tuple[str, int]] = []  # (path, n_steps)
        for f in files:
            with h5py.File(f, "r") as h:
                self.episodes.append((f, int(h["step"].shape[0])))

        with h5py.File(self.episodes[0][0], "r") as h:
            # resolve tactile group names: explicit override, else first present candidate set
            if tactile_keys is not None:
                cand = [tuple(tactile_keys)]
            else:
                cand = list(_TACTILE_NAME_SETS)
            self.tactile_keys: List[str] = []
            self.has_tactile = False
            for names in cand:
                if all(f"tactile/{n}/{self.tactile_image}" in h for n in names):
                    self.tactile_keys = list(names)
                    self.has_tactile = True
                    break
            if not self.has_tactile:
                # keep the requested/ default sensor count for a zero-filled tactile tensor
                self.tactile_keys = list(tactile_keys or _TACTILE_NAME_SETS[0])
                print(f"[univtac] WARNING: '{self.task_name}' has no tactile "
                      f"({self.tactile_image}); emitting ZERO tactile.")
            assert f"observation/{self.camera}/rgb" in h, \
                f"camera '{self.camera}' not in {self.episodes[0][0]}"

        # train/val split over episodes
        rng = np.random.default_rng(seed)
        n_ep = len(self.episodes)
        n_val = max(1, int(round(n_ep * val_ratio))) if val_ratio > 0 else 0
        perm = rng.permutation(n_ep)
        val_idx = set(perm[:n_val].tolist())
        keep = [i for i in range(n_ep) if (i in val_idx) == val]

        # window index: (episode_idx, start_frame) with a full T-window inside the episode
        self.index: List[Tuple[int, int]] = []
        for ei in keep:
            _, length = self.episodes[ei]
            for st in range(0, max(1, length - T + 1)):
                self.index.append((ei, st))

    def __len__(self) -> int:
        return len(self.index)

    def _file(self, path: str):
        import h5py

        h = self._h5.get(path)
        if h is None:
            h = h5py.File(path, "r")
            self._h5[path] = h
        return h

    def _rgb_at(self, h, k: int) -> np.ndarray:
        img = _decode_image(h[f"observation/{self.camera}/rgb"][k])
        img = _resize_hwc(img, *self.rgb_hw)
        return np.moveaxis(img, -1, 0)  # CHW

    def _tac_frame_6ch(self, h, key: str, ep_start: int, k: int, bg: np.ndarray) -> np.ndarray:
        frames = []
        for j in range(self.tactile_num_frames):
            idx = max(ep_start, k - j * self.tactile_stride)
            img = _decode_image(h[f"tactile/{key}/{self.tactile_image}"][idx])
            if self.remove_bg:
                img = np.clip(img - bg + 0.5, 0.0, 1.0)
            img = _resize_hwc(img, *self.tactile_hw)
            frames.append(np.moveaxis(img, -1, 0))  # CHW
        return np.concatenate(frames, axis=0).astype(np.float32)  # (3*num_frames,H,W)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        ei, start = self.index[i]
        path, length = self.episodes[ei]
        h = self._file(path)
        T = self.T
        idxs = [min(length - 1, start + t) for t in range(T)]  # clamp-pad short episodes
        ep_start = 0

        # vision (head camera) -> (T,3,H,W)
        rgb = np.stack([self._rgb_at(h, k) for k in idxs], 0).astype(np.float32)

        # tactile: S sensors -> (T, S, 6, H, W); zero-filled when the task has no tactile.
        ch = 3 * self.tactile_num_frames
        if self.has_tactile:
            tac_sensors = []
            for key in self.tactile_keys:
                bg = _decode_image(h[f"tactile/{key}/{self.tactile_image}"][ep_start]) \
                    if self.remove_bg else None
                frames = np.stack([self._tac_frame_6ch(h, key, ep_start, k, bg) for k in idxs], 0)
                tac_sensors.append(frames)  # (T,6,H,W)
            tac = np.stack(tac_sensors, axis=1)  # (T,S,6,H,W)
        else:
            tac = np.zeros((T, len(self.tactile_keys), ch, *self.tactile_hw), dtype=np.float32)

        # action: commanded next-step joint qpos, padded/truncated to action_dim (chunk size 1)
        joint = np.asarray(h["embodiment/joint"], dtype=np.float32)  # (length, J)
        act = []
        for k in idxs:
            nxt = min(length - 1, k + 1)
            j = joint[nxt]
            if j.shape[0] < self.action_dim:
                j = np.pad(j, (0, self.action_dim - j.shape[0]))
            else:
                j = j[: self.action_dim]
            act.append(j)
        act = np.stack(act, 0)[:, None].astype(np.float32)  # (T,1,A)

        return {
            "rgb": torch.from_numpy(rgb),
            "tactile": torch.from_numpy(tac),
            "action": torch.from_numpy(act),
            "task": self.task_name,
            "episode": int(ei),
            "frame0": int(start),
            "cam": self.camera,
        }


def make_univtac_dataset(data_root: str, val: bool = False, **kwargs):
    """Build a single- or multi-task UniVTAC dataset.

    `data_root` may be a single episode dir (e.g. `.../grasp_classify/clean`) or a parent
    containing several `{task}/{config}` stores; in the latter case all tasks are
    concatenated for multi-task training.
    """
    tasks = find_task_dirs(data_root)
    assert tasks, f"no UniVTAC episode stores found under {data_root}"
    if len(tasks) == 1:
        name, edir = tasks[0]
        return UniVTACDataset(edir, val=val, task_name=name, **kwargs)
    subsets, used = [], []
    for name, edir in tasks:
        try:
            subsets.append(UniVTACDataset(edir, val=val, task_name=name, **kwargs))
            used.append((name, edir))
        except Exception as e:  # noqa: BLE001
            print(f"[univtac] skip {name}: {type(e).__name__}: {str(e)[:80]}")
    assert subsets, f"no usable UniVTAC tasks under {data_root}"
    ds = ConcatDataset(subsets)
    ds.task_paths = used
    return ds
