"""Scripted expert for 'put the red cube in the bowl'.

A waypoint state machine over absolute end-effector targets (the env's IK controller
does the rest). Diversity comes from randomized speeds, approach / lift / release
heights, small grasp and placement offsets, timing, and optional noise on the
*executed* action (DART-style) while the *recorded* action stays clean.
"""
from __future__ import annotations

import numpy as np

from .env import BOWL_BASE_H, CUBE_HALF, PickPlaceEnv


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# miss : grasp target offset perpendicular to the finger closing direction -> fingers close beside the cube
# knock: offset along the closing direction -> a finger lands on / pushes the cube
# high : close the gripper above the cube
# drop : open the gripper for a few steps during the transfer
PERTURBATIONS = ("miss", "knock", "high", "drop")
PERTURB_PROBS = (0.35, 0.3, 0.15, 0.2)
MAX_REGRASP = 3


class ScriptedExpert:
    """recovery=False reproduces the v1 expert exactly. recovery=True (v2) injects one deliberate
    mistake on the first attempt (grasp offset -> miss / knock, wrong yaw, closing too high, or
    dropping the cube during transfer), checks the grasp from the gripper state, and re-grasps.
    Actions that *are* the deliberate mistake are flagged (self.bad) and excluded from training."""

    def __init__(self, env: PickPlaceEnv, rng: np.random.Generator, recovery: bool = False):
        self.env = env
        self.rng = rng
        r = rng
        self.speed = r.uniform(0.18, 0.32)  # m/s
        self.yaw_speed = r.uniform(1.5, 2.5)  # rad/s
        self.h_approach = r.uniform(0.07, 0.14)
        self.h_lift = r.uniform(0.12, 0.20)
        self.h_release = r.uniform(0.07, 0.10)
        self.grasp_dz = r.uniform(-0.006, 0.004)
        self.grasp_off = r.normal(0, 0.003, 2)
        self.place_off = r.uniform(-0.02, 0.02, 2)
        self.close_steps = int(r.integers(5, 8))
        self.release_steps = int(r.integers(3, 6))
        self.via = r.normal(0, 0.03, 2)  # lateral detour on the transfer
        self.phase = "approach"
        self.counter = 0
        self.recovery = recovery
        self.bad = False
        self.n_regrasp = 0
        self.n_close = 0
        self.perturb = None
        if recovery:  # sampled from a separate generator so the base randomization above is unchanged
            pr = np.random.default_rng(int(r.integers(1 << 31)))
            self.perturb = PERTURBATIONS[pr.choice(len(PERTURBATIONS), p=PERTURB_PROBS)]
            self.p_along = pr.choice([-1, 1]) * pr.uniform(0.03, 0.045)  # magnitude of the offset
            self.p_side = pr.normal(0, 0.004)
            self.p_high = pr.uniform(0.035, 0.05)
            self.p_drop_frac = pr.uniform(0.1, 0.45)
            self.p_drop_steps = int(pr.integers(2, 4))
            self.transfer_start = None
        self.perturb_active = self.perturb is not None
        p = env.tcp_pos()
        self.cmd = np.array([p[0], p[1], p[2], env.tcp_yaw(), 0.0])

    def _move_towards(self, goal_xyz, goal_yaw, grip, dt=0.1):
        cmd = self.cmd.copy()
        d = goal_xyz - cmd[:3]
        n = np.linalg.norm(d)
        step = self.speed * dt
        cmd[:3] = goal_xyz if n <= step else cmd[:3] + d / n * step
        dy = _wrap(goal_yaw - cmd[3])
        ys = self.yaw_speed * dt
        cmd[3] = goal_yaw if abs(dy) <= ys else cmd[3] + np.sign(dy) * ys
        cmd[4] = grip
        self.cmd = cmd
        return np.linalg.norm(goal_xyz - self.env.tcp_pos()), abs(_wrap(goal_yaw - self.env.tcp_yaw()))

    def grasped(self) -> bool:
        env = self.env
        return env.finger_width() > 0.025 and np.linalg.norm(env.cube_pos("red") - env.tcp_pos()) < 0.04

    def _start_regrasp(self):
        self.phase, self.counter = "reopen", 0
        self.n_regrasp += 1
        self.perturb_active = False
        self.bad = False

    def act(self) -> np.ndarray:
        env = self.env
        cube = env.cube_pos("red")
        cyaw = env.cube_yaw("red")
        bowl = env.bowl_pos()
        ph = self.phase
        self.bad = False
        if self.recovery and ph in ("lift", "transfer", "lower") and not self.grasped():
            self._start_regrasp()  # dropped / slipped: re-grasp from wherever the cube is now
            ph = self.phase
        if ph == "reopen":
            self.cmd[4] = 0.0
            self.counter += 1
            if self.counter >= 3:
                self.phase = "rise"
        elif ph == "rise":
            goal = np.array([self.cmd[0], self.cmd[1], max(self.cmd[2], cube[2] + self.h_approach)])
            err, _ = self._move_towards(goal, self.cmd[3], 0.0)
            if err < 0.015 and np.allclose(self.cmd[:3], goal):
                self.phase = "approach"
        elif ph == "approach":
            goal = np.array([cube[0] + self.grasp_off[0], cube[1] + self.grasp_off[1], cube[2] + self.h_approach])
            # keep track of the cube's yaw; the cmd yaw is continuous so this is fine
            err, yerr = self._move_towards(goal, cyaw, 0.0)
            if err < 0.012 and yerr < 0.08 and np.allclose(self.cmd[:3], goal):
                self.phase = "descend"
        elif ph == "descend":
            goal = np.array([cube[0] + self.grasp_off[0], cube[1] + self.grasp_off[1],
                             max((cube[2] if self.recovery else CUBE_HALF) + self.grasp_dz, 0.012)])
            gyaw = cyaw
            if self.perturb_active and self.perturb in ("miss", "knock", "high"):
                self.bad = True
                if self.perturb in ("miss", "knock"):
                    if not hasattr(self, "_p_goal"):  # freeze the wrong target (the cube may get pushed)
                        c, s_ = np.cos(self.cmd[3]), np.sin(self.cmd[3])
                        hx, hy = np.array([c, s_]), np.array([s_, -c])  # hand x / hand y (closing dir)
                        main, side = (hx, hy) if self.perturb == "miss" else (hy, hx)
                        self._p_goal = goal[:2] + self.p_along * main + self.p_side * side
                    goal[:2] = self._p_goal
                else:
                    goal[2] = cube[2] + self.p_high
            err, _ = self._move_towards(goal, gyaw, 0.0)
            tol = 0.02 if self.bad else 0.008  # a perturbed descent may be blocked by the cube
            if (err < tol and np.allclose(self.cmd[:3], goal)) or (self.bad and self.counter > 25):
                self.phase, self.counter = "close", 0
            elif self.bad:
                self.counter += 1
        elif ph == "close":
            self.cmd[4] = 1.0
            self.bad = self.perturb_active and self.perturb in ("miss", "knock", "high")
            self.counter += 1
            if self.counter >= self.close_steps:
                self.n_close += 1
                if self.recovery and not self.grasped():
                    self._start_regrasp()
                else:
                    self.perturb_active = self.perturb_active and self.perturb == "drop"
                    self.phase = "lift"
        elif ph == "lift":
            goal = np.array([self.cmd[0], self.cmd[1], self.h_lift])
            err, _ = self._move_towards(goal, self.cmd[3], 1.0)
            if err < 0.02 and np.allclose(self.cmd[:3], goal):
                self.phase = "transfer"
        elif ph == "transfer":
            goal = np.array([bowl[0] + self.place_off[0], bowl[1] + self.place_off[1], self.h_lift])
            mid = 0.5 * (self.cmd[:2] + goal[:2])
            if np.linalg.norm(self.cmd[:2] - goal[:2]) > 0.08:
                goal_eff = goal.copy()
                goal_eff[:2] = goal[:2] + self.via * min(1.0, np.linalg.norm(mid - goal[:2]) / 0.1)
            else:
                goal_eff = goal
            err, _ = self._move_towards(goal_eff, self.cmd[3], 1.0)
            if self.perturb_active and self.perturb == "drop":
                if self.transfer_start is None:
                    self.transfer_start = np.linalg.norm(self.cmd[:2] - goal[:2])
                prog = 1 - np.linalg.norm(self.cmd[:2] - goal[:2]) / max(self.transfer_start, 1e-6)
                if prog >= self.p_drop_frac:
                    self.phase, self.counter = "drop", 0
            if np.linalg.norm(self.env.tcp_pos()[:2] - goal[:2]) < 0.015 and np.allclose(self.cmd[:3], goal):
                self.phase = "lower"
        elif ph == "drop":  # deliberate mistake: open the gripper mid-transfer
            self.bad = True
            self.cmd[4] = 0.0
            self.counter += 1
            if self.counter >= self.p_drop_steps:
                self.perturb_active = False
                self._start_regrasp()
        elif ph == "lower":
            goal = np.array([bowl[0] + self.place_off[0], bowl[1] + self.place_off[1],
                             BOWL_BASE_H + self.h_release])
            err, _ = self._move_towards(goal, self.cmd[3], 1.0)
            if err < 0.015 and np.allclose(self.cmd[:3], goal):
                self.phase, self.counter = "release", 0
        elif ph == "release":
            self.cmd[4] = 0.0
            self.counter += 1
            if self.counter >= self.release_steps:
                self.phase = "retreat"
        elif ph == "retreat":
            goal = np.array([self.cmd[0], self.cmd[1], self.h_lift + 0.05])
            self._move_towards(goal, self.cmd[3], 0.0)
        return self.cmd.copy()


def rollout_expert(env: PickPlaceEnv, seed: int, noise_std: float = 0.0, extra_steps: int = 3,
                   max_steps: int = 160, recovery: bool = False, return_info: bool = False):
    """Run the expert once. Returns (episode dict, success[, info])."""
    rng = np.random.default_rng(seed + 10_000_019)
    obs = env.reset(seed=seed)
    expert = ScriptedExpert(env, rng, recovery=recovery)
    ep = dict(front_rgb=[], wrist_rgb=[], proprio=[], action=[], expert_mask=[])
    success_at = None
    for t in range(max_steps):
        a = expert.act()
        ep["front_rgb"].append(obs["front_rgb"])
        ep["wrist_rgb"].append(obs["wrist_rgb"])
        ep["proprio"].append(obs["proprio"])
        ep["action"].append(a.astype(np.float32))
        ep["expert_mask"].append(not expert.bad)
        a_exec = a.copy()
        if noise_std > 0 and expert.phase in ("approach", "transfer"):
            a_exec[:3] += rng.normal(0, noise_std, 3)
        obs, _, _, info = env.step(a_exec)
        if info["success"] and success_at is None:
            success_at = t
        if success_at is not None and t >= success_at + extra_steps:
            break
        if expert.n_regrasp > MAX_REGRASP:  # expert gives up -> failed episode (discarded)
            break
    ep = {k: np.stack(v) for k, v in ep.items()}
    ep["expert_mask"] = ep["expert_mask"].astype(np.uint8)
    if return_info:
        return ep, success_at is not None, dict(perturb=expert.perturb, n_regrasp=expert.n_regrasp,
                                                n_close=expert.n_close)
    return ep, success_at is not None
