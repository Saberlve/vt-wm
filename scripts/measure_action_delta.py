#!/usr/bin/env python
"""Measure per-dim per-step |Δqpos| on a UniVTAC task to ground the CEM init std.

The CEM action space is ABSOLUTE joint qpos and the search is seeded at the current pose, so the
right per-position init std is (per-dim per-step delta) x (cumulative source steps between the seed
and that action position). This script reports that per-dim per-step delta from the raw HDF5
`embodiment/joint` stream (the same array `univtac_dataset.py` chunks into actions).

Usage:
    XFORMERS_DISABLED=1 .venv/bin/python scripts/measure_action_delta.py \
        --data_root /run/determined/NAS1/public/wangshuxun/UniVTAC/lift_bottle/clean \
        --action_dim 8 --frame_stride 10
"""
import argparse
import glob
import os

import h5py
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="dir of *.hdf5 episodes (or a parent of task/config dirs)")
    ap.add_argument("--action_dim", type=int, default=8)
    ap.add_argument("--frame_stride", type=int, default=10, help="keyframe stride (matches data.frame_stride)")
    ap.add_argument("--max_episodes", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_root, "**", "*.hdf5"), recursive=True))
    if args.max_episodes:
        files = files[: args.max_episodes]
    if not files:
        raise SystemExit(f"no *.hdf5 under {args.data_root}")

    A, S = args.action_dim, args.frame_stride
    d1, dS, lens = [], [], []
    for f in files:
        with h5py.File(f, "r") as h:
            j = np.asarray(h["embodiment/joint"], dtype=np.float32)[:, :A]
        lens.append(len(j))
        if len(j) > 1:
            d1.append(np.abs(j[1:] - j[:-1]))
        if len(j) > S:
            dS.append(np.abs(j[S:] - j[:-S]))
    d1 = np.concatenate(d1, 0)
    dS = np.concatenate(dS, 0)

    np.set_printoptions(precision=5, suppress=True)
    print(f"episodes={len(files)}  mean_len={np.mean(lens):.0f}")
    print(f"\nper-step (stride=1) |delta| per dim:")
    print("  mean  ", d1.mean(0))
    print("  p90   ", np.percentile(d1, 90, 0))
    print(f"\nkeyframe (stride={S}) |delta| per dim:")
    print("  mean  ", dS.mean(0))
    print("  ratio keyframe/per-step (should ~= stride if growth is linear):")
    print("        ", dS.mean(0) / np.maximum(d1.mean(0), 1e-9))
    arm_mean = d1[:, : max(1, A - 1)].mean()   # exclude the (near-static) gripper dim
    print("\nSuggested val_plan_sigma_step (scalar per-step delta for the CEM ramp):")
    print(f"  arm-mean = {arm_mean:.5f}   (all-dim mean = {d1.mean():.5f})")


if __name__ == "__main__":
    main()
