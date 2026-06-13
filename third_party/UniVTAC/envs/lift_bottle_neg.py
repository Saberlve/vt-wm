"""Counterexample (negative) data collection for the lift_bottle task.

The positive lift_bottle script uses smooth cuRobo primitives.  VT-WM/CEM failures are
different: deploy executes raw absolute qpos commands, and bad plans tend to look like
high-frequency joint jitter, wrong-direction drift, occasional spikes, and gripper
mis-toggles.  This task keeps the scripted pre-grasp so the bottle is actually in play,
then replaces the expert manipulation with CEM-like bad qpos windows and records the
physical consequences.
"""
from ._base_task import *
from .lift_bottle import Task as LiftBottleTask, TaskCfg as LiftBottleCfg
import numpy as np


@configclass
class TaskCfg(LiftBottleCfg):
    # Overall multiplier on the CEM-like qpos perturbation profile.
    neg_noise_scale: float = 1.0
    # Backward-compatible field. Gripper drops are now produced by qpos toggles.
    neg_drop_prob: float = 0.25
    # CEM/deploy shape for lift_bottle: 4 world-model steps x 10 raw qpos commands.
    neg_horizon: int = 4
    neg_action_chunk: int = 10
    # Number of bad CEM windows to execute after grasping.
    neg_qpos_windows: int = 2
    # Per-command probability of a large single-joint qpos spike.
    neg_spike_prob: float = 0.18
    # Per-window probability of opening the gripper for a short bad segment.
    neg_gripper_toggle_prob: float = 0.25
    # Same scale as VT-WM qpos_sigma_ramp: std grows by this per raw command.
    neg_cem_sigma_step: float = 0.002
    # Minimum early std and one-shot offset so the first saved actions are already bad.
    neg_early_sigma: float = 0.035
    neg_early_kick: float = 0.12
    # Episode-level diversity: some trajectories are bad immediately, others become bad later.
    neg_late_bad_prob: float = 0.5
    neg_late_bad_steps: int = 20


FRANKA_ARM_LOWER = np.array(
    [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
    dtype=np.float32,
)
FRANKA_ARM_UPPER = np.array(
    [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
    dtype=np.float32,
)


class Task(LiftBottleTask):
    def pre_move(self):
        # Keep pre-grasp close to the expert. The negative signal should come from
        # bad actions after contact, not from trivially missing the bottle at reset.
        self.delay(10)

        bottle_pose = self.bottle.get_pose()
        target_pose = bottle_pose.add_bias([-0.13, 0, -0.015])
        target_mat = target_pose.to_transformation_matrix()
        self.grasp_noise = self.create_noise(euler=[0, [-np.pi / 12, 0.0], 0])
        target_pose = construct_grasp_pose(
            target_pose.p,
            target_mat[:3, 2],
            target_mat[:3, 0],
        ).add_offset(self.grasp_noise)
        grasp_idx = self.bottle.register_point(pose=target_pose, type='contact')
        self.move(self.atom.grasp_actor(
            self.bottle,
            contact_point_id=grasp_idx,
            is_close=False,
            pre_dis=0.5,
        ))
        self.target_pose = self.wall.get_pose().add_bias([-0.08, 0, 0])

    def _play_once(self):
        self.move(self.atom.close_gripper())

        params = {
            'profile': 'cem_qpos_bad',
            'temporal_profile': self._sample_temporal_profile(),
            'qpos_windows': [],
            'executed_steps': 0,
        }
        self.atom_id += 1
        self.atom_tag = 'neg_cem_qpos'

        for window_id in range(int(self.cfg.neg_qpos_windows)):
            qpos_seq, window_params = self._sample_bad_qpos_window(
                window_id, params['temporal_profile']
            )
            params['qpos_windows'].append(window_params)
            executed = self._execute_qpos_sequence(qpos_seq)
            params['executed_steps'] += executed
            if executed < len(qpos_seq) or not self.plan_success:
                break

        self.delay(30, is_save=False)
        self._record_neg(params)

    def _sample_temporal_profile(self):
        if self.rng.random() < float(self.cfg.neg_late_bad_prob):
            return 'late_bad'
        return 'always_bad'

    def _sample_bad_qpos_window(self, window_id: int, temporal_profile: str):
        """Sample one CEM-shaped absolute-qpos window around the current robot state."""
        rng = self.rng
        s = float(self.cfg.neg_noise_scale)
        horizon = int(self.cfg.neg_horizon)
        chunk = int(self.cfg.neg_action_chunk)
        steps = max(1, horizon * chunk)

        base = self._robot_manager.get_qpos()[0, :8].numpy().astype(np.float32)
        step_ids = np.arange(1, steps + 1, dtype=np.float32)
        ramp_sigma = float(self.cfg.neg_cem_sigma_step) * step_ids
        early_decay = np.exp(-(step_ids - 1.0) / 7.0).astype(np.float32)
        sigma = np.maximum(ramp_sigma, float(self.cfg.neg_early_sigma) * early_decay * s)

        # Per-dim multipliers make the later commands visibly CEM-like: mostly small
        # jitter, with larger pitch/wrist/gripper-sensitive dimensions.
        dim_scale = np.array([1.2, 1.5, 1.7, 1.6, 1.9, 1.8, 1.3, 0.35], dtype=np.float32)
        seq = base[None, :] + rng.normal(
            loc=0.0,
            scale=(sigma[:, None] * dim_scale[None, :] * s),
            size=(steps, 8),
        ).astype(np.float32)

        profile = rng.choice(['wrong_lift', 'jerk_rotate', 'push_side', 'drop_open'])
        drift = np.zeros((steps, 8), dtype=np.float32)
        ramp = np.linspace(0.35, 1.0, steps, dtype=np.float32)
        kick = np.zeros(8, dtype=np.float32)
        if profile == 'wrong_lift':
            kick[[2, 5]] = [-rng.uniform(0.04, 0.10), -rng.uniform(0.08, 0.18)]
            drift[:, 2] -= ramp * rng.uniform(0.06, 0.18) * s
            drift[:, 5] -= ramp * rng.uniform(0.12, 0.32) * s
        elif profile == 'jerk_rotate':
            kick[[3, 5]] = [
                rng.choice([-1.0, 1.0]) * rng.uniform(0.08, 0.18),
                rng.choice([-1.0, 1.0]) * rng.uniform(0.08, 0.20),
            ]
            drift[:, 3] += ramp * rng.choice([-1.0, 1.0]) * rng.uniform(0.10, 0.26) * s
            drift[:, 5] += ramp * rng.choice([-1.0, 1.0]) * rng.uniform(0.12, 0.34) * s
        elif profile == 'push_side':
            kick[[0, 1]] = [
                rng.choice([-1.0, 1.0]) * rng.uniform(0.04, 0.12),
                rng.choice([-1.0, 1.0]) * rng.uniform(0.06, 0.16),
            ]
            drift[:, 0] += ramp * rng.choice([-1.0, 1.0]) * rng.uniform(0.05, 0.16) * s
            drift[:, 1] += ramp * rng.choice([-1.0, 1.0]) * rng.uniform(0.08, 0.24) * s
        else:
            kick[[4, 6]] = [
                rng.choice([-1.0, 1.0]) * rng.uniform(0.06, 0.16),
                rng.choice([-1.0, 1.0]) * rng.uniform(0.04, 0.12),
            ]
            drift[:, 4] += ramp * rng.choice([-1.0, 1.0]) * rng.uniform(0.08, 0.24) * s
            drift[:, 6] += ramp * rng.choice([-1.0, 1.0]) * rng.uniform(0.06, 0.18) * s
        kick *= float(self.cfg.neg_early_kick) / 0.12 * s
        drift += kick[None, :] * early_decay[:, None]
        seq += drift

        early_steps = min(12, steps)
        early_spike_mask = rng.random((early_steps, 7)) < min(0.9, 0.45 * s)
        early_spike_mag = rng.uniform(0.04, 0.16, size=(early_steps, 7)).astype(np.float32)
        early_spike_sign = rng.choice([-1.0, 1.0], size=(early_steps, 7)).astype(np.float32)
        seq[:early_steps, :7] += (
            early_spike_mask.astype(np.float32) * early_spike_mag * early_spike_sign * s
        )

        spike_prob = min(0.95, float(self.cfg.neg_spike_prob) * s)
        spike_mask = rng.random((steps, 7)) < spike_prob
        spike_mag = rng.uniform(0.06, 0.30, size=(steps, 7)).astype(np.float32)
        spike_sign = rng.choice([-1.0, 1.0], size=(steps, 7)).astype(np.float32)
        seq[:, :7] += spike_mask.astype(np.float32) * spike_mag * spike_sign * s

        gripper_max = float(self._robot_manager.gripper_max_qpos)
        gripper_opened = False
        seq[:, 7] = base[7] + rng.normal(0.0, 0.002 * s, size=steps).astype(np.float32)
        if profile == 'drop_open' or rng.random() < min(1.0, float(self.cfg.neg_gripper_toggle_prob) * s):
            gripper_opened = True
            start = int(rng.integers(max(1, steps // 5), max(2, steps - 5)))
            length = int(rng.integers(3, min(12, steps - start) + 1))
            seq[start:start + length, 7] = gripper_max * rng.uniform(0.55, 1.0)
            if start + length < steps:
                seq[start + length:, 7] = rng.uniform(0.0, 0.35) * gripper_max

        late_bad_steps = 0
        if temporal_profile == 'late_bad' and window_id == 0:
            late_bad_steps = min(int(self.cfg.neg_late_bad_steps), steps - 1)
            seq[:late_bad_steps] = self._sample_normal_prefix(
                base, late_bad_steps, gripper_max
            )

        seq = self._clip_qpos(seq, gripper_max)
        stats = {
            'window_id': int(window_id),
            'profile': str(profile),
            'temporal_profile': str(temporal_profile),
            'late_bad_steps': int(late_bad_steps),
            'steps': int(steps),
            'noise_scale': float(s),
            'sigma_step': float(self.cfg.neg_cem_sigma_step),
            'spike_prob': float(spike_prob),
            'spike_count': int(spike_mask.sum() + early_spike_mask.sum()),
            'early_sigma': float(self.cfg.neg_early_sigma),
            'early_kick': kick.astype(float).tolist(),
            'gripper_opened': bool(gripper_opened),
            'qpos_min': seq.min(axis=0).astype(float).tolist(),
            'qpos_max': seq.max(axis=0).astype(float).tolist(),
            'qpos_std': seq.std(axis=0).astype(float).tolist(),
        }
        return seq, stats

    def _sample_normal_prefix(self, base: np.ndarray, steps: int, gripper_max: float):
        """Small smooth holding motion before a late-bad trajectory starts."""
        if steps <= 0:
            return np.empty((0, 8), dtype=np.float32)

        rng = self.rng
        prefix = np.repeat(base[None, :], steps, axis=0).astype(np.float32)
        t = np.linspace(0.0, 1.0, steps, dtype=np.float32)
        smooth = 0.5 - 0.5 * np.cos(np.pi * t)

        target = np.zeros(8, dtype=np.float32)
        target[[2, 5]] = [
            rng.uniform(-0.012, 0.018),
            rng.uniform(-0.020, 0.030),
        ]
        target[[0, 1, 3, 4, 6]] = rng.uniform(-0.008, 0.008, size=5)
        prefix += smooth[:, None] * target[None, :]
        prefix[:, :7] += rng.normal(0.0, 0.003, size=(steps, 7)).astype(np.float32)
        prefix[:, 7] = np.clip(
            base[7] + rng.normal(0.0, 0.0008, size=steps).astype(np.float32),
            0.0,
            gripper_max,
        )
        return prefix

    def _clip_qpos(self, seq: np.ndarray, gripper_max: float):
        seq = np.asarray(seq, dtype=np.float32).copy()
        seq[:, :7] = np.clip(seq[:, :7], FRANKA_ARM_LOWER, FRANKA_ARM_UPPER)
        seq[:, 7] = np.clip(seq[:, 7], 0.0, gripper_max)
        return seq

    def _execute_qpos_sequence(self, qpos_seq: np.ndarray):
        executed = 0
        for qpos in qpos_seq:
            if self.take_action_cnt >= self.cfg.step_lim or not self.plan_success:
                break
            # Negative data should not stop just because a transient pose happens to
            # satisfy lift_bottle's success heuristic.
            self.eval_success = False
            action = torch.as_tensor(qpos, dtype=torch.float32, device=self.device)
            self.take_action(action, action_type='qpos', force=True)
            self.eval_success = False
            executed += 1
        return executed

    def _record_neg(self, params):
        """Stash perturbation params + the true outcome label into the episode metadata."""
        params['grasp_noise'] = {
            'p': np.asarray(self.grasp_noise.p, dtype=float).tolist(),
            'q': np.asarray(self.grasp_noise.q, dtype=float).tolist(),
        }
        params['neg_noise_scale'] = float(self.cfg.neg_noise_scale)
        params['plan_success'] = bool(self.plan_success)
        try:
            params['success'] = bool(self.check_success())
            params['mid_success'] = bool(self.check_mid_success())
        except Exception:
            params['success'] = False
        self.metadata['neg_params'] = params

    def check_early_stop(self):
        # Do not truncate negative episodes on the expert success-spin heuristic;
        # we want the full perturbed window regardless of outcome.
        return False
