"""Offline contact-force visualization for episodes collected with `record_force: true`.

Reads the NPZ sidecar written by envs/utils/contact_force.py (plus the episode HDF5
for the recorded tactile RGB) and renders:
  - 2D: per-sensor normal-pressure heatmaps (JET, redder = more force) beside the
    recorded GelSight RGB, as an mp4.
  - 3D: a point-cloud video of gelpads + actors with contacting vertices colored
    by per-vertex |F|.
  - --calib_check: static-equilibrium sanity check of the Newton conversion
    (resting bottle: sum of contact F_z vs m*g).

No Isaac imports — runnable in any env with numpy/cv2/h5py/matplotlib.

Usage:
  python scripts/visualize_force.py --data_root data/lift_bottle/force --seed 0 --mode both
"""

import json
import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm


# ---------------------------------------------------------------- data access

def quat_to_mat(q):
    """(w,x,y,z) quaternion -> 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def pose_to_mat(pose7):
    T = np.eye(4)
    T[:3, :3] = quat_to_mat(pose7[3:7])
    T[:3, 3] = pose7[:3]
    return T


class ForceEpisode:
    def __init__(self, npz_path, hdf5_path=None):
        self.npz = np.load(npz_path, allow_pickle=True)
        self.meta = json.loads(str(self.npz['meta']))
        self.steps = self.npz['steps']
        self.atom_tags = self.npz['atom_tag']
        self.in_pre_move = self.npz['in_pre_move']
        self.T = len(self.steps)
        self.sensors = list(self.meta['sensors'].keys())
        self.objects = list(self.meta['objects'].keys())

        self.h5 = None
        self.h5_steps = None
        if hdf5_path is not None and Path(hdf5_path).exists():
            self.h5 = h5py.File(hdf5_path, 'r')
            self.h5_steps = self.h5['step'][()]

    def block(self, name, t):
        """Sparse contact block of object `name` at recorded index t."""
        ptr = self.npz[f'{name}/ptr']
        a, b = int(ptr[t]), int(ptr[t + 1])
        if a == b:
            return None
        return {
            'vidx': self.npz[f'{name}/vidx'][a:b],
            'force': self.npz[f'{name}/force'][a:b].astype(np.float64),
            'pos': self.npz[f'{name}/pos'][a:b].astype(np.float64),
        }

    def world_to_local(self, sensor, t):
        """world -> gelpad local frame at recorded index t.

        attach_to_init already maps the attach-body frame into the gel local
        frame (the `origin_pts` frame of VisualTactileSensor.setup), so the
        full chain is world -> attach body -> gel local.
        """
        A2I = self.npz[f'meta/gel_{sensor}/attach_to_init']
        pose = self.npz[f'summary/{sensor}/pose'][t].astype(np.float64)
        return A2I @ np.linalg.inv(pose_to_mat(pose))

    def rest_local(self, name):
        """Rest vertices of `name` in its local frame."""
        rest = self.npz[f'meta/{name}/rest_verts'].astype(np.float64)
        if name.startswith('gel_'):
            # rest_verts are world-frame at recorder init; the robot has not
            # moved by the first recorded step, so its pose gives the transform
            L0 = self.world_to_local(name[len('gel_'):], 0)
            rest = rest @ L0[:3, :3].T + L0[:3, 3]
        return rest

    def tactile_rgb(self, sensor, t):
        """Recorded tactile RGB (decoded) nearest to recorded index t, or None."""
        if self.h5 is None:
            return None
        i = int(np.argmin(np.abs(self.h5_steps - self.steps[t])))
        buf = self.h5[f'tactile/{sensor}/rgb'][i]
        arr = np.frombuffer(buf if isinstance(buf, bytes) else buf.tobytes(), np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# ---------------------------------------------------------------- 2D heatmaps

def jet_heatmap(arr2d, vmax, size=None):
    norm = np.clip(arr2d / max(vmax, 1e-9), 0.0, 1.0)
    img = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    if size is not None:
        img = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)
    return img


def pressure_map(ep: ForceEpisode, sensor, t, shape, splat_grid=(12, 16)):
    """Rebuild the sensor-plane normal pressure map (Pa) at resolution `shape`.

    Point forces are splatted on a coarse grid whose cell size matches the gel
    mesh vertex spacing (so force / cell-area is a physical nodal pressure),
    then bilinearly upsampled — splatting directly into fine pixels would
    concentrate each nodal force into one tiny pixel and blow up the pressure.
    """
    sm = ep.meta['sensors'][sensor]
    h, w = shape
    sh, sw = splat_grid
    block = ep.block(f'gel_{sensor}', t)
    if block is None:
        return np.zeros((h, w), np.float64)
    M = ep.world_to_local(sensor, t)
    f_l = block['force'] @ M[:3, :3].T
    p_l = block['pos'] @ M[:3, :3].T + M[:3, 3]
    rest = ep.rest_local(f'gel_{sensor}')
    lo, hi = rest.min(axis=0), rest.max(axis=0)
    xa, ya, na = sm['x_axis'], sm['y_axis'], sm['normal_axis']
    f_n = np.maximum(-sm['outer_sign'] * f_l[:, na], 0.0)
    ix = np.clip(np.round((p_l[:, xa] - lo[xa]) / max(hi[xa] - lo[xa], 1e-9) * (sw - 1)), 0, sw - 1).astype(int)
    iy = np.clip(np.round((p_l[:, ya] - lo[ya]) / max(hi[ya] - lo[ya], 1e-9) * (sh - 1)), 0, sh - 1).astype(int)
    coarse = np.zeros((sh, sw), np.float64)
    np.add.at(coarse, (iy, ix), f_n)
    cell_area = (hi[xa] - lo[xa]) / sw * (hi[ya] - lo[ya]) / sh
    coarse /= max(cell_area, 1e-12)
    out = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_LINEAR)
    k = max(3, 2 * int(min(h, w) / min(sh, sw)) + 1)
    return cv2.GaussianBlur(out, (k, k), 0)


def annotate(img, lines, color=(255, 255, 255)):
    for i, line in enumerate(lines):
        cv2.putText(img, line, (6, 18 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return img


def render_2d(ep: ForceEpisode, out_dir: Path, indices, fps, vmax_arg):
    grid = (120, 160)  # (h, w) splat resolution
    panel = (320, 240)  # upscaled panel size (w, h)

    if vmax_arg == 'auto':
        sample = indices[:: max(1, len(indices) // 50)]
        vals = [pressure_map(ep, s, t, grid).max() for s in ep.sensors for t in sample]
        vmax = float(np.percentile([v for v in vals if v > 0] or [1.0], 99))
    else:
        vmax = float(vmax_arg)
    print(f'[2d] pressure vmax = {vmax:.3e} Pa')

    n_panels = 2 * len(ep.sensors)
    size = (panel[0] * n_panels, panel[1])
    writer = cv2.VideoWriter(str(out_dir / 'force_2d.mp4'),
                             cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for t in indices:
        cols = []
        for sensor in ep.sensors:
            rgb = ep.tactile_rgb(sensor, t)
            if rgb is None:
                rgb = np.zeros((panel[1], panel[0], 3), np.uint8)
            rgb = cv2.resize(rgb, panel)
            heat = jet_heatmap(pressure_map(ep, sensor, t, grid), vmax, panel)
            total = ep.npz[f'summary/{sensor}/total_force_w'][t]
            annotate(rgb, [f'{sensor} rgb'])
            annotate(heat, [f'{sensor} pressure',
                            f'|F|={np.linalg.norm(total):.2f} N'])
            cols += [rgb, heat]
        frame = np.concatenate(cols, axis=1)
        annotate(frame, ['', '', '',
                         f'step {int(ep.steps[t])}  atom={ep.atom_tags[t]}'
                         f'{"  [pre_move]" if ep.in_pre_move[t] else ""}'])
        writer.write(frame)
    writer.release()
    print(f'[2d] wrote {out_dir / "force_2d.mp4"} ({len(indices)} frames)')


# ---------------------------------------------------------------- 3D pointcloud

def render_3d(ep: ForceEpisode, out_dir: Path, indices, fps, vmax_arg, max_ctx_pts=1200,
              exclude=()):
    objects = [n for n in ep.objects if n not in exclude]
    # context clouds in local/rest frames + how to place them per step
    ctx = {}
    for name in objects:
        rest = ep.npz[f'meta/{name}/rest_verts'].astype(np.float64)
        if rest.shape[0] > max_ctx_pts:
            rest = rest[np.linspace(0, rest.shape[0] - 1, max_ctx_pts).astype(int)]
        ctx[name] = rest

    def ctx_world(name, t):
        rest = ctx[name]
        if f'{name}/transform' in ep.npz:  # ABD actor: rest-local + per-step transform
            T = ep.npz[f'{name}/transform'][t].astype(np.float64)
            return rest @ T[:3, :3].T + T[:3, 3]
        if name.startswith('gel_'):  # FEM gelpad: rigid approx via attach pose
            sensor = name[len('gel_'):]
            L0 = ep.world_to_local(sensor, 0)
            Li = np.linalg.inv(ep.world_to_local(sensor, t))  # local->world
            local = rest @ L0[:3, :3].T + L0[:3, 3]
            return local @ Li[:3, :3].T + Li[:3, 3]
        return rest

    if vmax_arg == 'auto':
        # scale off the gelpads + grasped object; static scenery (wall) carries
        # huge ground-reaction forces that would wash out the interesting range
        focus = [n for n in objects if n.startswith('gel_') or n == 'bottle'] or objects
        mags = []
        for name in focus:
            f = ep.npz[f'{name}/force']
            if f.shape[0]:
                mags.append(np.linalg.norm(f, axis=1))
        vmax = float(np.percentile(np.concatenate(mags), 99)) if mags else 1.0
    else:
        vmax = float(vmax_arg)
    print(f'[3d] per-vertex |F| vmax = {vmax:.3e} N')

    # fixed axis bounds over the episode (sampled)
    pts = np.concatenate([ctx_world(n, t) for n in objects
                          for t in indices[:: max(1, len(indices) // 10)]])
    center, span = pts.mean(axis=0), max(pts.max(axis=0) - pts.min(axis=0)) * 0.6

    writer = None
    for t in indices:
        fig = plt.figure(figsize=(7, 6), dpi=100)
        ax = fig.add_subplot(projection='3d')
        for name in objects:
            p = ctx_world(name, t)
            ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=3, c='0.6', alpha=0.5, linewidths=0)
        sc = None
        for name in objects:
            block = ep.block(name, t)
            if block is None:
                continue
            mag = np.linalg.norm(block['force'], axis=1)
            sc = ax.scatter(block['pos'][:, 0], block['pos'][:, 1], block['pos'][:, 2],
                            s=30, c=np.clip(mag / vmax, 0, 1), cmap=cm.jet,
                            vmin=0, vmax=1, linewidths=0, depthshade=False)
        if sc is not None:
            cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
            cb.set_label(f'|F| / {vmax:.2g} N')
        ax.set_xlim(center[0] - span, center[0] + span)
        ax.set_ylim(center[1] - span, center[1] + span)
        ax.set_zlim(center[2] - span, center[2] + span)
        ax.view_init(elev=22, azim=-60)
        ax.set_title(f'step {int(ep.steps[t])}  atom={ep.atom_tags[t]}'
                     f'{"  [pre_move]" if ep.in_pre_move[t] else ""}')
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        plt.close(fig)

        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if writer is None:
            writer = cv2.VideoWriter(str(out_dir / 'force_3d.mp4'),
                                     cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                     (frame.shape[1], frame.shape[0]))
        writer.write(frame)
    if writer is not None:
        writer.release()
    print(f'[3d] wrote {out_dir / "force_3d.mp4"} ({len(indices)} frames)')


# ---------------------------------------------------------------- calibration

def calib_check(ep: ForceEpisode):
    g = 9.81
    print('=== static-equilibrium calibration check ===')
    for name in ep.objects:
        if name.startswith('gel_') or name not in ep.meta['objects']:
            continue
        info = ep.meta['objects'][name]
        mass = info.get('mass_density', 0.0) * info.get('volume_m3', 0.0)
        if mass <= 0:
            continue
        # steps where the object is in contact but no gelpad is (resting state)
        gel_free = np.ones(ep.T, bool)
        for s in ep.sensors:
            ptr = ep.npz[f'gel_{s}/ptr']
            gel_free &= (ptr[1:] == ptr[:-1])
        fz = []
        for t in np.nonzero(gel_free)[0]:
            block = ep.block(name, t)
            if block is not None:
                fz.append(block['force'][:, 2].sum())
        if fz:
            fz = np.asarray(fz)
            print(f'{name}: m*g = {mass * g:8.3f} N | resting sum F_z: '
                  f'median {np.median(fz):8.3f} N over {len(fz)} steps '
                  f'(ratio {np.median(fz) / (mass * g):.3f})')
        else:
            print(f'{name}: no gel-free contact steps found')

    # grasp steady state: both pads in contact
    both = np.ones(ep.T, bool)
    for s in ep.sensors:
        ptr = ep.npz[f'gel_{s}/ptr']
        both &= (ptr[1:] > ptr[:-1])
    idx = np.nonzero(both)[0]
    if len(idx):
        print(f'--- grasp steady state ({len(idx)} steps with both pads in contact) ---')
        for s in ep.sensors:
            f = ep.npz[f'summary/{s}/total_force_w'][idx]
            print(f'{s}: mean total F_w = {f.mean(axis=0).round(3)} N, '
                  f'mean |F| = {np.linalg.norm(f, axis=1).mean():.3f} N')


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data_root', required=True, help='episode root, e.g. data/lift_bottle/force')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--mode', choices=['2d', '3d', 'both'], default='both')
    ap.add_argument('--fps', type=int, default=10)
    ap.add_argument('--vmax', default='auto')
    ap.add_argument('--steps', default=None, help='recorded-index range a:b')
    ap.add_argument('--stride', type=int, default=2, help='render every Nth recorded step')
    ap.add_argument('--out', default=None)
    ap.add_argument('--exclude', default='', help='comma-separated objects to hide in 3d (e.g. wall)')
    ap.add_argument('--calib_check', action='store_true')
    args = ap.parse_args()

    root = Path(args.data_root)
    npz_path = root / 'force' / f'{args.seed}.npz'
    hdf5_path = root / 'hdf5' / f'{args.seed}.hdf5'
    assert npz_path.exists(), f'missing {npz_path}'
    ep = ForceEpisode(npz_path, hdf5_path)
    print(f'loaded {npz_path}: {ep.T} recorded steps, sensors={ep.sensors}, objects={ep.objects}')

    if args.calib_check:
        calib_check(ep)
        return

    indices = np.arange(ep.T)
    if args.steps:
        a, b = (int(x) if x else None for x in args.steps.split(':'))
        indices = indices[a:b]
    indices = indices[::args.stride]

    out_dir = Path(args.out) if args.out else root / 'force_vis' / str(args.seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ('2d', 'both'):
        render_2d(ep, out_dir, indices, args.fps, args.vmax)
    if args.mode in ('3d', 'both'):
        exclude = tuple(x for x in args.exclude.split(',') if x)
        render_3d(ep, out_dir, indices, args.fps, args.vmax, exclude=exclude)


if __name__ == '__main__':
    main()
