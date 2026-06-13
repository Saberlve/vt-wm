"""Convert recorded contact-force data into GelSight tactile RGB images (offline).

Pipeline (physics-based, no learning):
  per-vertex contact forces (NPZ sidecar from `record_force: true`)
    -> sensor-plane normal pressure map p (Pa)            [splat in gelpad local frame]
    -> indentation map delta = p * h / E (Winkler elastic foundation; h = gel
       thickness, E = gel Young's modulus from the sim config)
    -> object height map (mm) + press_depth               [Taxim convention]
    -> Taxim optical simulation -> tactile RGB

`--source depth` is a sanity path: it renders the *recorded* gel height maps
through the same offline Taxim renderer, validating the renderer + conventions
independently of the force-derived maps.

The Taxim torch backend is loaded standalone (the parent `tacex` package needs
omni/Isaac; the `gpu_taxim/sim` subpackage does not).

Usage:
  python scripts/force_to_gelsight.py --data_root data/lift_bottle/force --seed 0 --source depth
  python scripts/force_to_gelsight.py --data_root data/lift_bottle/force --seed 0 --source force
"""

import sys
import csv
import json
import argparse
import importlib.util
from pathlib import Path

import cv2
import h5py
import torch
import numpy as np

UNIVTAC_ROOT = Path(__file__).resolve().parent.parent
SIM_DIR = (UNIVTAC_ROOT / 'third_party/TacEx/source/tacex/tacex/'
           'simulation_approaches/gpu_taxim/sim')
CALIB_DIR = (UNIVTAC_ROOT / 'third_party/TacEx/source/tacex_assets/tacex_assets/'
             'data/Sensors/GelSight_Mini/calibs/640x480')

# GelSight-Mini optical constants (tacex_assets gsmini_cfg.py)
GELPAD_HEIGHT_MM = 4.5
GEL_TO_CAMERA_MIN_MM = 24.0

sys.path.insert(0, str(UNIVTAC_ROOT / 'scripts'))
from visualize_force import ForceEpisode, pressure_map, jet_heatmap, annotate  # noqa: E402


def load_taxim(device='cuda'):
    spec = importlib.util.spec_from_file_location(
        'taxim_sim_offline', SIM_DIR / '__init__.py',
        submodule_search_locations=[str(SIM_DIR)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules['taxim_sim_offline'] = mod
    spec.loader.exec_module(mod)
    return mod.Taxim(calib_folder=CALIB_DIR, backend='torch', device=device)


def indentation_from_height_map(hm_mm):
    """Same formula as TaximSimulator.compute_indentation_depth (mm)."""
    dist = hm_mm.min() - GEL_TO_CAMERA_MIN_MM
    dist = max(dist, 0.0)
    return GELPAD_HEIGHT_MM - dist if dist <= GELPAD_HEIGHT_MM else 0.0


def to_uint8(img):
    if img.dtype != np.uint8:
        img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
    return img


class Converter:
    def __init__(self, ep: ForceEpisode, taxim, sensor, device='cuda'):
        self.ep = ep
        self.taxim = taxim
        self.sensor = sensor
        self.device = device
        sm = ep.meta['sensors'][sensor]
        self.res = tuple(sm['resolution'])          # (w, h)
        self.shape = (self.res[1], self.res[0])     # (h, w)
        self.youngs = sm['youngs_pa']
        self.thickness = sm['thickness_m']
        self.flipx = False
        self.flipy = False

    # -- recorded ground truth -------------------------------------------------

    def gt_depth(self, t):
        """Recorded gel height map (mm) at the hdf5 frame nearest recorded index t."""
        i = int(np.argmin(np.abs(self.ep.h5_steps - self.ep.steps[t])))
        return self.ep.h5[f'tactile/{self.sensor}/depth'][i].astype(np.float64)

    def gt_indentation_map(self, t, background):
        return np.maximum(background - self.gt_depth(t), 0.0)

    # -- force path -------------------------------------------------------------

    def delta_map_mm(self, t):
        """Indentation map (mm) on the sensor grid from contact forces.

        Saturating elastic-foundation law: delta = h * p / (p + E). For p << E
        this is the linear Winkler model delta = p*h/E; for large p it saturates
        at the gel thickness h (the measured grasp pressures far exceed E, where
        the linear model would predict indentations many times the pad height).
        """
        p = pressure_map(self.ep, self.sensor, t, self.shape)  # Pa
        if self.flipx:
            p = p[:, ::-1]
        if self.flipy:
            p = p[::-1, :]
        delta = np.ascontiguousarray(self.thickness * p / (p + self.youngs) * 1000.0)  # m -> mm
        # smooth at the gel-mesh vertex-spacing scale (~20 px) to remove nodal lumps
        return cv2.GaussianBlur(delta, (31, 31), 0)

    def auto_orient(self, indices, background):
        """Pick the x/y flips maximizing correlation with the recorded indentation."""
        scored = []
        for t in indices:
            gt = self.gt_indentation_map(t, background)
            if gt.max() < 0.05:
                continue
            base = self.delta_map_mm(t)
            if base.max() <= 0:
                continue
            scored.append((gt, base))
            if len(scored) >= 10:
                break
        if not scored:
            return
        best, best_cfg = -2.0, (False, False)
        for fx in (False, True):
            for fy in (False, True):
                cs = []
                for gt, base in scored:
                    d = base
                    if fx:
                        d = d[:, ::-1]
                    if fy:
                        d = d[::-1, :]
                    c = np.corrcoef(gt.ravel(), d.ravel())[0, 1]
                    if np.isfinite(c):
                        cs.append(c)
                score = np.mean(cs) if cs else -2.0
                if score > best:
                    best, best_cfg = score, (fx, fy)
        self.flipx, self.flipy = best_cfg
        print(f'[{self.sensor}] auto-orient: flipx={self.flipx} flipy={self.flipy} '
              f'(corr {best:.3f})')

    def render_from_force(self, t):
        delta = self.delta_map_mm(t)
        press = float(delta.max())
        if press <= 1e-4:
            rgb = self.taxim.background_img
            rgb = rgb.permute(1, 2, 0).cpu().numpy()
            return to_uint8(cv2.resize(rgb, self.res)), delta
        # Taxim object height map: smaller = closer to camera; offset is
        # irrelevant because press_depth re-shifts by the map minimum.
        hm = np.where(delta > 1e-4, -delta, 1.0)
        rgb = self.taxim.render(hm, with_shadow=False, press_depth=self._press(press))
        return to_uint8(cv2.resize(rgb, self.res)), delta

    def render_from_depth(self, t):
        hm = self.gt_depth(t)
        press = indentation_from_height_map(hm)
        rgb = self.taxim.render(hm, with_shadow=False,
                                press_depth=self._press(press) if press > 0 else None)
        return to_uint8(cv2.resize(rgb, self.res))

    def _press(self, press_mm: float):
        # TaximTorch expects a per-batch tensor, not a python float
        return torch.tensor([press_mm], dtype=torch.float32, device=self.device)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--sensor', default='both', help='sensor name or "both"')
    ap.add_argument('--source', choices=['force', 'depth'], default='force')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--fps', type=int, default=10)
    ap.add_argument('--stride', type=int, default=1, help='stride over hdf5-aligned steps')
    ap.add_argument('--flipx', action='store_true', help='force x flip (skip auto-orient)')
    ap.add_argument('--flipy', action='store_true', help='force y flip (skip auto-orient)')
    ap.add_argument('--no_auto_orient', action='store_true')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    root = Path(args.data_root)
    npz_path = root / 'force' / f'{args.seed}.npz'
    hdf5_path = root / 'hdf5' / f'{args.seed}.hdf5'
    assert npz_path.exists(), f'missing {npz_path}'
    assert hdf5_path.exists(), f'missing {hdf5_path} (needed for GT comparison)'
    ep = ForceEpisode(npz_path, hdf5_path)

    sensors = ep.sensors if args.sensor == 'both' else [args.sensor]
    out_dir = Path(args.out) if args.out else root / 'force2gel' / str(args.seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    print('loading Taxim (torch backend)...')
    taxim = load_taxim(args.device)

    # recorded indices aligned to saved hdf5 frames (those have GT rgb/depth)
    step_to_t = {int(s): t for t, s in enumerate(ep.steps)}
    aligned = [step_to_t[int(s)] for s in ep.h5_steps if int(s) in step_to_t]
    aligned = aligned[::args.stride]
    print(f'{len(aligned)} recorded steps aligned with saved frames')

    for sensor in sensors:
        conv = Converter(ep, taxim, sensor, args.device)
        depth_ds = ep.h5[f'tactile/{sensor}/depth']
        background = float(np.median([depth_ds[i].max() for i in
                                      range(0, depth_ds.shape[0], max(1, depth_ds.shape[0] // 20))]))
        bg_map = np.full(conv.shape, background)

        if args.source == 'force':
            conv.flipx, conv.flipy = args.flipx, args.flipy
            if not (args.flipx or args.flipy or args.no_auto_orient):
                conv.auto_orient(aligned, bg_map)

        panel = conv.res
        n_cols = 4 if args.source == 'force' else 2
        writer = cv2.VideoWriter(str(out_dir / f'{sensor}_{args.source}.mp4'),
                                 cv2.VideoWriter_fourcc(*'mp4v'), args.fps,
                                 (panel[0] * n_cols, panel[1]))
        metrics = []
        for k, t in enumerate(aligned):
            gt_rgb_bgr = ep.tactile_rgb(sensor, t)
            gt_ind = conv.gt_indentation_map(t, bg_map)
            if args.source == 'depth':
                sim_rgb = conv.render_from_depth(t)
                sim_bgr = cv2.cvtColor(sim_rgb, cv2.COLOR_RGB2BGR)
                frame = np.concatenate([
                    annotate(gt_rgb_bgr.copy(), [f'{sensor} recorded rgb']),
                    annotate(sim_bgr, ['taxim(recorded depth)']),
                ], axis=1)
                contact = gt_ind > 0.05
                rgb_diff = (np.abs(gt_rgb_bgr.astype(float) - sim_bgr.astype(float))
                            [contact].mean() if contact.any() else 0.0)
                metrics.append({'step': int(ep.steps[t]), 'rgb_l1_contact': rgb_diff})
            else:
                sim_rgb, delta = conv.render_from_force(t)
                sim_bgr = cv2.cvtColor(sim_rgb, cv2.COLOR_RGB2BGR)
                vmax = max(gt_ind.max(), delta.max(), 1e-6)
                frame = np.concatenate([
                    annotate(gt_rgb_bgr.copy(), [f'{sensor} recorded rgb']),
                    annotate(sim_bgr, ['force -> taxim rgb']),
                    annotate(jet_heatmap(gt_ind, vmax, panel), ['recorded indentation']),
                    annotate(jet_heatmap(delta, vmax, panel), ['force indentation (elastic fnd.)']),
                ], axis=1)
                union = (gt_ind > 0.05) | (delta > 0.05)
                corr = (np.corrcoef(gt_ind.ravel(), delta.ravel())[0, 1]
                        if union.any() else np.nan)
                metrics.append({
                    'step': int(ep.steps[t]),
                    'ind_corr': float(corr) if np.isfinite(corr) else 0.0,
                    'ind_l1_mm': float(np.abs(gt_ind - delta)[union].mean()) if union.any() else 0.0,
                    'gt_max_mm': float(gt_ind.max()),
                    'force_max_mm': float(delta.max()),
                })
            annotate(frame, ['', '', f'step {int(ep.steps[t])} atom={ep.atom_tags[t]}'])
            writer.write(frame)
            if k % max(1, len(aligned) // 8) == 0:
                cv2.imwrite(str(out_dir / f'{sensor}_{args.source}_{int(ep.steps[t]):05d}.png'), frame)
        writer.release()

        csv_path = out_dir / f'{sensor}_{args.source}_metrics.csv'
        if metrics:
            with open(csv_path, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(metrics[0].keys()))
                w.writeheader()
                w.writerows(metrics)
            keys = [k for k in metrics[0] if k != 'step']
            means = {k: float(np.mean([m[k] for m in metrics])) for k in keys}
            print(f'[{sensor}] {len(metrics)} frames -> {out_dir / f"{sensor}_{args.source}.mp4"}')
            print(f'[{sensor}] mean metrics: {json.dumps(means, indent=None)}')


if __name__ == '__main__':
    main()
