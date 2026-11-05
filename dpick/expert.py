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


class ScriptedExpert:
    def __init__(self, env: PickPlaceEnv, rng: np.random.Generator):
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

    def act(self) -> np.ndarray:
        env = self.env
        cube = env.cube_pos("red")
        cyaw = env.cube_yaw("red")
        bowl = env.bowl_pos()
        ph = self.phase
        if ph == "approach":
            goal = np.array([cube[0] + self.grasp_off[0], cube[1] + self.grasp_off[1], cube[2] + self.h_approach])
            # keep track of the cube's yaw; the cmd yaw is continuous so this is fine
            err, yerr = self._move_towards(goal, cyaw, 0.0)
            if err < 0.012 and yerr < 0.08 and np.allclose(self.cmd[:3], goal):
                self.phase = "descend"
        elif ph == "descend":
            goal = np.array([cube[0] + self.grasp_off[0], cube[1] + self.grasp_off[1],
                             max(CUBE_HALF + self.grasp_dz, 0.012)])
            err, _ = self._move_towards(goal, cyaw, 0.0)
            if err < 0.008 and np.allclose(self.cmd[:3], goal):
                self.phase, self.counter = "close", 0
        elif ph == "close":
            self.cmd[4] = 1.0
            self.counter += 1
            if self.counter >= self.close_steps:
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
            if np.linalg.norm(self.env.tcp_pos()[:2] - goal[:2]) < 0.015 and np.allclose(self.cmd[:3], goal):
                self.phase = "lower"
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
                   max_steps: int = 160):
    """Run the expert once. Returns (episode dict, success)."""
    rng = np.random.default_rng(seed + 10_000_019)
    obs = env.reset(seed=seed)
    expert = ScriptedExpert(env, rng)
    ep = dict(front_rgb=[], wrist_rgb=[], proprio=[], action=[])
    success_at = None
    for t in range(max_steps):
        a = expert.act()
        ep["front_rgb"].append(obs["front_rgb"])
        ep["wrist_rgb"].append(obs["wrist_rgb"])
        ep["proprio"].append(obs["proprio"])
        ep["action"].append(a.astype(np.float32))
        a_exec = a.copy()
        if noise_std > 0 and expert.phase in ("approach", "transfer"):
            a_exec[:3] += rng.normal(0, noise_std, 3)
        obs, _, _, info = env.step(a_exec)
        if info["success"] and success_at is None:
            success_at = t
        if success_at is not None and t >= success_at + extra_steps:
            break
    ep = {k: np.stack(v) for k, v in ep.items()}
    return ep, success_at is not None
