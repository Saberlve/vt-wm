"""Dump the structure of a ManiFeel (diffusion-policy ReplayBuffer) zarr store.

Usage: python -m vtwm.data.inspect_zarr <path-to-extracted-task-dir-or-zarr>
"""
from __future__ import annotations

import sys

import numpy as np
import zarr


def find_zarr_root(path: str) -> str:
    import os

    if os.path.isdir(path) and os.path.exists(os.path.join(path, ".zgroup")):
        return path
    for root, dirs, files in os.walk(path):
        if ".zgroup" in files and ("data" in dirs or "meta" in dirs):
            return root
        for d in dirs:
            if d.endswith(".zarr"):
                return os.path.join(root, d)
    return path


def main():
    path = find_zarr_root(sys.argv[1])
    print("zarr root:", path)
    root = zarr.open(path, "r")
    print("top-level groups/arrays:", list(root.keys()))
    if "data" in root:
        print("\n[data] arrays:")
        for k in root["data"].keys():
            a = root["data"][k]
            print(f"  {k:34s} shape={a.shape} dtype={a.dtype} chunks={a.chunks}")
            sample = np.asarray(a[0])
            print(f"      first-elem: shape={sample.shape} dtype={sample.dtype} "
                  f"min={sample.min():.3f} max={sample.max():.3f}")
    if "meta" in root:
        print("\n[meta] arrays:")
        for k in root["meta"].keys():
            a = root["meta"][k]
            arr = np.asarray(a[:])
            print(f"  {k:20s} shape={a.shape} dtype={a.dtype} "
                  f"head={arr[:5].tolist()} n_episodes={len(arr)} total_steps={arr[-1] if len(arr) else 0}")


if __name__ == "__main__":
    main()
