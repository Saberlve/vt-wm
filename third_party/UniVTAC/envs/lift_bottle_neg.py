"""Counterexample (negative) data collection for the lift_bottle task.

Reuses lift_bottle's scripted approach (`pre_move` grasps the bottle exactly as the
expert does), then from the grasp onward replays the expert manipulation with
*amplified* noise injected at the action-primitive target level. cuRobo still plans a
smooth trajectory to each (wrong) target, so the arm motion stays physically valid
despite `set_arm(force=True)` being a hard kinematic set -- only the *goal* is off.

The point is to give the VT-WM world model coverage of off-distribution actions and
their real consequences (missed grasp, topple, slip, mid-lift drop, push-off) so CEM
planning can learn to avoid them. Episodes are saved regardless of success via the
`save_all: true` flag in the task_config (see scripts/collect_data.py); the true
success/fail label and the perturbation parameters are stored under
`metadata.json[seed]['neg_params']`.
"""
from ._base_task import *
from .lift_bottle import Task as LiftBottleTask, TaskCfg as LiftBottleCfg
import numpy as np


@configclass
class TaskCfg(LiftBottleCfg):
    # Overall multiplier on every injected perturbation (1.0 = default profile below).
    neg_noise_scale: float = 1.0
    # Probability of an early mid-lift gripper release (bottle drops).
    neg_drop_prob: float = 0.25


class Task(LiftBottleTask):
    def pre_move(self):
        # Identical scripted approach to the expert, but with an *amplified* grasp
        # perturbation so the grasp itself is often off (too high / off-centre / tilted)
        # -> "perturb from pre_move onward, including the grasp".
        self.delay(10)

        s = self.cfg.neg_noise_scale
        bottle_pose = self.bottle.get_pose()
        target_pose = bottle_pose.add_bias([-0.13, 0, -0.015])
        target_mat = target_pose.to_transformation_matrix()
        # vec: lateral/height offset ranges (m); euler: in-plane tilt range (rad).
        self.grasp_noise = self.create_noise(
            vec=[
                [-0.015 * s, 0.015 * s],   # x: depth toward bottle
                [-0.025 * s, 0.025 * s],   # y: off-centre along the bottle
                [-0.020 * s, 0.030 * s],   # z: too-high grasp
            ],
            euler=[0, [-np.pi / 6 * s, np.pi / 12 * s], 0],
        )
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
        s = self.cfg.neg_noise_scale
        rng = self.rng
        params = {}

        # 1) Grasp (already perturbed via the amplified grasp_noise in pre_move).
        self.move(self.atom.close_gripper())

        # 2) Noisy lift-rotate: spread around the expert 70deg, occasionally reversed.
        base = 70 / 180 * np.pi
        theta = base + np.deg2rad(rng.uniform(-40, 40)) * s
        if rng.random() < 0.15 * s:
            theta = -theta * rng.uniform(0.3, 1.0)   # wrong-direction rotate
        params['rotate_theta_deg'] = float(np.rad2deg(theta))
        self.gripper_rotate(self.bottle, theta, steps=4)

        # 3) Early-drop branch: release mid-lift so the bottle falls.
        if rng.random() < self.cfg.neg_drop_prob:
            params['early_drop'] = True
            self.move(self.atom.open_gripper(float(rng.uniform(0.5, 1.0))))
            self.delay(40)
            self._record_neg(params)
            return

        # 4) Perturbed carry: wrong pitch then wrong xy displacement
        #    -> topple / push the bottle off / pull it the wrong way.
        pitch = rng.uniform(-np.pi / 4, np.pi / 2) * s
        dx = rng.uniform(-0.05, 0.16)
        dy = rng.uniform(-0.06, 0.06) * s
        params.update(
            pitch_deg=float(np.rad2deg(pitch)),
            dx=float(dx),
            dy=float(dy),
        )
        self.move(self.atom.move_by_displacement(
            rpy=[0, pitch, 0], rpy_coord='gripper'
        ), time_dilation_factor=0.5)
        self.move(self.atom.move_by_displacement(
            x=dx, y=dy
        ), time_dilation_factor=0.5)

        # 5) Release and settle.
        self.move(self.atom.open_gripper(0.5))
        self.delay(30, is_save=False)
        self._record_neg(params)

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
