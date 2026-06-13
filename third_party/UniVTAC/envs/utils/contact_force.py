"""Per-vertex contact force extraction from the UIPC solver.

The vendored libuipc exposes ``ContactSystemFeature`` whose ``contact_gradient``
returns the sparse per-vertex gradient of the IPC contact potential (doublets of
global vertex index + Vector3). All contact kernels scale the barrier by
``kappa * dt * dt``, so the physical contact force on a vertex is
``F = -grad / dt^2`` (Newtons, world frame). Both the FEM gelpads and the
affine-body actors (bottle/wall) live in the same UIPC world, so one query
covers forces on the gripper gel and on the manipulated objects.
"""

import json
import time
import numpy as np
import torch
from pathlib import Path

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .._base_task import BaseTask


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class ContactForceExtractor:
    """Queries libuipc's ContactSystemFeature and maps global vertex indices to objects."""

    def __init__(self, task: 'BaseTask'):
        self.task = task
        self.uipc_sim = task.uipc_sim
        self.dt = float(task.cfg.uipc_sim.dt)
        self._feature = None
        self._grad_geo = None
        self._ranges = None  # list of (lo, hi, name, obj, kind)
        self.disabled = False

    # ---- lazy setup -------------------------------------------------------

    def _build_feature(self):
        from uipc.core import ContactSystemFeature
        from uipc import geometry as uipc_geometry
        world = self.uipc_sim.world
        feature = None
        try:
            feature = world.features().find(ContactSystemFeature)
        except TypeError:
            feature = world.features().find(ContactSystemFeature.FeatureName)
        if feature is None:
            raise RuntimeError('ContactSystemFeature not found on UIPC world')
        self._feature = feature
        self._grad_geo = uipc_geometry.Geometry()

    def _build_ranges(self):
        offsets = self.uipc_sim._system_vertex_offsets["uipc::backend::cuda::GlobalVertexManager"]
        # resolve human-readable names: actors by scene registry, gelpads via the tactile manager
        names = {}
        for name, actor in getattr(self.task.scene, 'uipc_objects', {}).items():
            names[id(actor)] = name
        for name, tact in self.task._tactile_manager.tactiles.items():
            names[id(tact.gelpad)] = f'gel_{name}'

        ranges = []
        for obj in self.uipc_sim.uipc_objects:
            lo = int(offsets[obj.global_system_id])
            hi = lo + int(obj._vertex_count)
            name = names.get(id(obj), Path(obj.cfg.prim_path).name)
            kind = 'abd' if obj._system_name.endswith('AffineBodyDynamics') else 'fem'
            ranges.append((lo, hi, name, obj, kind))
        self._ranges = ranges

    # ---- accessors --------------------------------------------------------

    @property
    def ranges(self):
        if self._ranges is None:
            self._build_ranges()
        return self._ranges

    def object_positions(self, obj, kind):
        """Rest-frame positions for ABD objects, world-frame for FEM."""
        geo = obj.geo_slot_list[0].geometry()
        return geo.positions().view().reshape(-1, 3).copy()

    def object_transform(self, obj):
        """Current 4x4 world transform of an ABD object (updated on every retrieve)."""
        geo = obj.geo_slot_list[0].geometry()
        return geo.transforms().view().reshape(4, 4).copy()

    # ---- main query --------------------------------------------------------

    def query(self):
        """Returns ``{name: {'vidx': (K,), 'force': (K,3) N world, 'pos': (K,3) m world}}``."""
        if self.disabled:
            return {}
        if self._feature is None:
            self._build_feature()
        if self._ranges is None:
            self._build_ranges()

        self._feature.contact_gradient(self._grad_geo)
        inst = self._grad_geo.instances()
        n = inst.size()
        if n == 0:
            return {}
        # views are transient: copy immediately
        idx = inst.find("i").view().reshape(-1).copy().astype(np.int64)
        grad = inst.find("grad").view().reshape(-1, 3).copy()
        force = -grad / (self.dt * self.dt)

        out = {}
        for lo, hi, name, obj, kind in self._ranges:
            mask = (idx >= lo) & (idx < hi)
            if not mask.any():
                continue
            local = idx[mask] - lo
            f = force[mask]
            # multiple contact pairs can touch the same vertex: accumulate
            uniq, inv = np.unique(local, return_inverse=True)
            f_acc = np.zeros((uniq.shape[0], 3), dtype=np.float64)
            np.add.at(f_acc, inv, f)

            pos = self.object_positions(obj, kind)[uniq]
            if kind == 'abd':
                T = self.object_transform(obj)
                pos = pos @ T[:3, :3].T + T[:3, 3]
            out[name] = {
                'vidx': uniq.astype(np.int32),
                'force': f_acc.astype(np.float32),
                'pos': pos.astype(np.float32),
            }
        return out


class ForceRecorder:
    """Records per-step contact forces over a whole episode and saves an NPZ sidecar.

    Independent of the observation-saving gate: ``record()`` is called every sim
    step (including pre_move), so the gripper force history covers the full episode.
    """

    def __init__(self, task: 'BaseTask'):
        self.task = task
        self.extractor = ContactForceExtractor(task)
        self.map_size = tuple(task.cfg.force_map_size)  # (h, w)
        self._sensor_meta = None
        self._static_meta = None
        self._warned = False
        self.reset()

    # ---- static metadata ---------------------------------------------------

    def _build_sensor_meta(self):
        meta = {}
        for name, tact in self.task._tactile_manager.tactiles.items():
            gelpad = tact.gelpad
            W0 = _to_numpy(gelpad.init_world_transform).astype(np.float64).reshape(4, 4)
            attach_to_init = _to_numpy(tact.attach_to_init).astype(np.float64).reshape(4, 4)
            rest_world = _to_numpy(gelpad.init_vertex_pos).astype(np.float64).reshape(-1, 3)
            # gel local frame == the `origin_pts` frame of VisualTactileSensor.setup:
            # world -> attach body -> local via attach_to_init (no prim transform involved)
            attach_now = np.asarray(tact.get_attach_pose().to_transformation_matrix(), dtype=np.float64)
            L0 = attach_to_init @ np.linalg.inv(attach_now)
            rest_local = rest_world @ L0[:3, :3].T + L0[:3, 3]

            extents = rest_local.max(axis=0) - rest_local.min(axis=0)
            normal_axis = int(np.argmin(extents))
            tang = [a for a in range(3) if a != normal_axis]
            # image x = sensor width (larger tangential extent for GelSight-Mini)
            if extents[tang[0]] >= extents[tang[1]]:
                x_axis, y_axis = tang[0], tang[1]
            else:
                x_axis, y_axis = tang[1], tang[0]
            # the outer (contact) gel face is the one away from the attachment points
            attach_local = rest_local[tact.attachment.attachment_points_idx.cpu().numpy()] \
                if isinstance(tact.attachment.attachment_points_idx, torch.Tensor) \
                else rest_local[np.asarray(tact.attachment.attachment_points_idx)]
            center_n = 0.5 * (rest_local[:, normal_axis].max() + rest_local[:, normal_axis].min())
            outer_sign = -1.0 if attach_local[:, normal_axis].mean() > center_n else 1.0

            sensor_cfg = tact.cfg.sensor_cfg
            try:
                real_size = tuple(sensor_cfg.marker_motion_sim_cfg.real_size)
            except AttributeError:
                real_size = (0.0266, 0.0209)
            try:
                resolution = tuple(sensor_cfg.sensor_camera_cfg.resolution)
            except AttributeError:
                resolution = (320, 240)
            youngs_pa = float(tact.cfg.gelpad_cfg.constitution_cfg.youngs_modulus) * 1e6

            meta[name] = {
                'tact': tact,
                'W0': W0,
                'attach_to_init': attach_to_init,
                'rest_local': rest_local,
                'normal_axis': normal_axis,
                'outer_sign': outer_sign,
                'x_axis': x_axis,
                'y_axis': y_axis,
                'bounds': (rest_local.min(axis=0), rest_local.max(axis=0)),
                'real_size': real_size,
                'resolution': resolution,
                'youngs_pa': youngs_pa,
                'thickness_m': float(extents[normal_axis]),
            }
        self._sensor_meta = meta

    def _build_static_meta(self):
        objects = {}
        for lo, hi, name, obj, kind in self.extractor.ranges:
            rest = self.extractor.object_positions(obj, kind)
            entry = {'kind': kind, 'vertex_count': int(obj._vertex_count),
                     'mass_density': float(getattr(obj.cfg, 'mass_density', 0.0))}
            # tet volume (for the static-equilibrium force calibration check)
            try:
                geo = obj.geo_slot_list[0].geometry()
                tets = geo.tetrahedra().topo().view().reshape(-1, 4).copy()
                v = rest[tets]
                vol = np.abs(np.einsum('ij,ij->i',
                                       np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]),
                                       v[:, 3] - v[:, 0])) / 6.0
                entry['volume_m3'] = float(vol.sum())
            except Exception:
                entry['volume_m3'] = 0.0
            objects[name] = {'entry': entry, 'rest': rest.astype(np.float32)}
        self._static_meta = objects

    # ---- per-step recording --------------------------------------------------

    def record(self, step_count: int, atom_id: int, atom_tag: str, in_pre_move: bool):
        try:
            self._record(step_count, atom_id, atom_tag, in_pre_move)
        except Exception as e:
            if not self._warned:
                self._warned = True
                print(f'\n[ForceRecorder] disabled after error: {e!r}')
            self.extractor.disabled = True

    def _record(self, step_count, atom_id, atom_tag, in_pre_move):
        if self._sensor_meta is None:
            self._build_sensor_meta()
        if self._static_meta is None:
            self._build_static_meta()

        data = self.extractor.query()

        self.steps.append(step_count)
        self.atom_ids.append(atom_id)
        self.atom_tags.append(atom_tag or '')
        self.pre_move_flags.append(bool(in_pre_move))

        summary = {}
        for lo, hi, name, obj, kind in self.extractor.ranges:
            buf = self.buffers.setdefault(name, {'vidx': [], 'force': [], 'pos': [], 'ptr': [0]})
            block = data.get(name)
            if block is None:
                buf['ptr'].append(buf['ptr'][-1])
            else:
                buf['vidx'].append(block['vidx'])
                buf['force'].append(block['force'])
                buf['pos'].append(block['pos'])
                buf['ptr'].append(buf['ptr'][-1] + block['vidx'].shape[0])
            if kind == 'abd':
                self.transforms.setdefault(name, []).append(
                    self.extractor.object_transform(obj).astype(np.float32))

        for name, sm in self._sensor_meta.items():
            key = f'gel_{name}'
            block = data.get(key)
            pose = sm['tact'].get_attach_pose()
            pose_arr = np.asarray(pose.tolist(), dtype=np.float32)
            entry = self.summaries.setdefault(name, {
                'total_force_w': [], 'total_force_local': [], 'max_force': [],
                'contact_count': [], 'normal_map': [], 'pose': []})
            h, w = self.map_size
            if block is None:
                entry['total_force_w'].append(np.zeros(3, np.float32))
                entry['total_force_local'].append(np.zeros(3, np.float32))
                entry['max_force'].append(0.0)
                entry['contact_count'].append(0)
                entry['normal_map'].append(np.zeros((h, w), np.float32))
            else:
                f_w = block['force'].astype(np.float64)
                p_w = block['pos'].astype(np.float64)
                M = self._world_to_local(sm, pose)
                f_l = f_w @ M[:3, :3].T
                p_l = p_w @ M[:3, :3].T + M[:3, 3]
                entry['total_force_w'].append(f_w.sum(axis=0).astype(np.float32))
                entry['total_force_local'].append(f_l.sum(axis=0).astype(np.float32))
                mags = np.linalg.norm(f_w, axis=1)
                entry['max_force'].append(float(mags.max()))
                entry['contact_count'].append(int(len(mags)))
                entry['normal_map'].append(self._splat_normal_map(sm, f_l, p_l, (h, w)))
            entry['pose'].append(pose_arr)
            summary[name] = {k: entry[k][-1] for k in entry}

        for name in self._actor_names():
            block = data.get(name)
            entry = self.summaries.setdefault(name, {'total_force_w': [], 'max_force': []})
            if block is None:
                entry['total_force_w'].append(np.zeros(3, np.float32))
                entry['max_force'].append(0.0)
            else:
                entry['total_force_w'].append(block['force'].sum(axis=0).astype(np.float32))
                entry['max_force'].append(float(np.linalg.norm(block['force'], axis=1).max()))
            summary[name] = {k: entry[k][-1] for k in entry}

        self._latest_summary = summary

    def _actor_names(self):
        return [name for lo, hi, name, obj, kind in self.extractor.ranges
                if not name.startswith('gel_')]

    def _world_to_local(self, sm, pose):
        """world -> gelpad local frame at the current attach pose."""
        attach_mat = np.asarray(pose.to_transformation_matrix(), dtype=np.float64)
        return sm['attach_to_init'] @ np.linalg.inv(attach_mat)

    def _splat_normal_map(self, sm, f_local, p_local, shape):
        h, w = shape
        lo, hi = sm['bounds']
        xa, ya, na = sm['x_axis'], sm['y_axis'], sm['normal_axis']
        # force pressing INTO the gel (toward the attachment side) is positive
        f_n = np.maximum(-sm['outer_sign'] * f_local[:, na], 0.0)
        px = (p_local[:, xa] - lo[xa]) / max(hi[xa] - lo[xa], 1e-9) * (w - 1)
        py = (p_local[:, ya] - lo[ya]) / max(hi[ya] - lo[ya], 1e-9) * (h - 1)
        ix = np.clip(np.round(px).astype(np.int64), 0, w - 1)
        iy = np.clip(np.round(py).astype(np.int64), 0, h - 1)
        m = np.zeros((h, w), dtype=np.float64)
        np.add.at(m, (iy, ix), f_n)
        cell_area = (hi[xa] - lo[xa]) / w * (hi[ya] - lo[ya]) / h
        return (m / max(cell_area, 1e-12)).astype(np.float32)  # Pa

    # ---- output ---------------------------------------------------------------

    def latest_summary(self):
        out = {}
        for name, entry in (self._latest_summary or {}).items():
            out[name] = {k: np.asarray(v) for k, v in entry.items()}
        return out

    def reset(self):
        self.steps, self.atom_ids, self.atom_tags, self.pre_move_flags = [], [], [], []
        self.buffers = {}
        self.transforms = {}
        self.summaries = {}
        self._latest_summary = None

    def save(self, path):
        if not self.steps:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        arrays = {
            'steps': np.asarray(self.steps, dtype=np.int64),
            'atom_id': np.asarray(self.atom_ids, dtype=np.int64),
            'atom_tag': np.asarray(self.atom_tags),
            'in_pre_move': np.asarray(self.pre_move_flags, dtype=bool),
        }
        for name, buf in self.buffers.items():
            arrays[f'{name}/ptr'] = np.asarray(buf['ptr'], dtype=np.int64)
            arrays[f'{name}/vidx'] = (np.concatenate(buf['vidx'])
                                      if buf['vidx'] else np.zeros(0, np.int32))
            arrays[f'{name}/force'] = (np.concatenate(buf['force'])
                                       if buf['force'] else np.zeros((0, 3), np.float32))
            arrays[f'{name}/pos'] = (np.concatenate(buf['pos'])
                                     if buf['pos'] else np.zeros((0, 3), np.float32))
        for name, mats in self.transforms.items():
            arrays[f'{name}/transform'] = np.stack(mats)
        for name, entry in self.summaries.items():
            for k, vals in entry.items():
                arrays[f'summary/{name}/{k}'] = np.stack([np.asarray(v) for v in vals])

        meta = {
            'dt': self.extractor.dt,
            'created': time.strftime('%Y-%m-%d %H:%M:%S'),
            'd_hat': float(self.task.cfg.uipc_sim.contact.d_hat),
            'force_map_size': list(self.map_size),
            'objects': {}, 'sensors': {},
        }
        if self._static_meta is not None:
            for name, info in self._static_meta.items():
                meta['objects'][name] = info['entry']
                arrays[f'meta/{name}/rest_verts'] = info['rest']
        if self._sensor_meta is not None:
            for name, sm in self._sensor_meta.items():
                meta['sensors'][name] = {
                    'normal_axis': sm['normal_axis'],
                    'outer_sign': sm['outer_sign'],
                    'x_axis': sm['x_axis'],
                    'y_axis': sm['y_axis'],
                    'real_size': list(sm['real_size']),
                    'resolution': list(sm['resolution']),
                    'youngs_pa': sm['youngs_pa'],
                    'thickness_m': sm['thickness_m'],
                }
                arrays[f'meta/gel_{name}/attach_to_init'] = sm['attach_to_init'].astype(np.float64)
                arrays[f'meta/gel_{name}/init_world_transform'] = sm['W0'].astype(np.float64)
        arrays['meta'] = np.asarray(json.dumps(meta))

        np.savez_compressed(path, **arrays)
        print(f'\n[ForceRecorder] saved {len(self.steps)} steps -> {path}')
