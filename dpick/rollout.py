"""Closed-loop policy execution in the env (receding-horizon for DP, temporal ensembling for ACT)."""
from __future__ import annotations

from collections import deque

import numpy as np
import torch

from .dataset import IMG_KEYS
from .policy_io import load_checkpoint


class PolicyRunner:
    def __init__(self, ckpt_path, device="cpu", act_temporal_ensemble=True, ensemble_k=0.01,
                 num_inference_steps=None):
        self.model, self.cfg, self.anorm, self.pnorm, _ = load_checkpoint(ckpt_path, device)
        self.device = torch.device(device)
        self.kind = self.cfg["policy"]
        m = self.cfg["model"]
        if self.kind == "diffusion":
            self.To, self.Tp, self.Ta = m["obs_horizon"], m["pred_horizon"], m["action_horizon"]
            if num_inference_steps:
                self.model.sched.set_inference_steps(num_inference_steps)
        else:
            self.To, self.Tp, self.Ta = 1, m["chunk_size"], m["chunk_size"] // 2
        self.ensemble = self.kind == "act" and act_temporal_ensemble
        self.k = ensemble_k
        self.reset()

    def reset(self):
        self.hist = deque(maxlen=self.To)
        self.queue = deque()
        self.t = 0
        self.ens = {}  # time -> list of predicted actions (oldest first)

    def _batch(self):
        h = list(self.hist)
        while len(h) < self.To:
            h.insert(0, h[0])
        b = {}
        for k in IMG_KEYS:
            x = np.stack([o[k] for o in h])  # To,H,W,3
            b[k] = torch.from_numpy(x).permute(0, 3, 1, 2)[None].to(self.device)
        p = np.stack([o["proprio"] for o in h]).astype(np.float32)
        b["proprio"] = torch.from_numpy(self.pnorm.normalize(p))[None].to(self.device)
        return b

    def _predict(self):
        chunk = self.model.sample(self._batch())[0].cpu().numpy()
        return self.anorm.unnormalize(chunk)

    def act(self, obs):
        self.hist.append(obs)
        if self.ensemble:
            chunk = self._predict()
            for i, a in enumerate(chunk):
                self.ens.setdefault(self.t + i, []).append(a)
            preds = np.stack(self.ens.pop(self.t))
            w = np.exp(-self.k * np.arange(len(preds)))
            a = (preds * w[:, None]).sum(0) / w.sum()
        else:
            if not self.queue:
                chunk = self._predict()
                start = self.To - 1 if self.kind == "diffusion" else 0
                self.queue.extend(chunk[start:start + self.Ta])
            a = self.queue.popleft()
        self.t += 1
        return a


def run_episode(env, runner: PolicyRunner, seed: int, max_steps: int, record=False):
    obs = env.reset(seed=seed)
    runner.reset()
    frames = []
    success, t = False, 0
    grasp_attempts, closed = 0, False  # number of open->close gripper transitions (>1 = re-grasp)
    for t in range(max_steps):
        if record:
            frames.append(make_frame(obs))
        a = runner.act(obs)
        if a[4] >= 0.5 and not closed:
            grasp_attempts += 1
        closed = a[4] >= 0.5
        obs, _, done, info = env.step(a)
        if info["success"]:
            success = True
            if record:
                for _ in range(4):
                    frames.append(make_frame(obs, ok=True))
            break
    return dict(success=success, steps=t + 1, grasp_attempts=grasp_attempts), frames


def make_frame(obs, ok=False):
    """[viz | front over wrist] panel for GIFs."""
    viz = obs.get("viz_rgb")
    f, w = obs["front_rgb"], obs["wrist_rgb"]
    H = viz.shape[0] if viz is not None else 256
    s = H // 2
    rep = max(1, s // f.shape[0])
    f2 = np.kron(f, np.ones((rep, rep, 1), np.uint8))[:s, :s]
    w2 = np.kron(w, np.ones((rep, rep, 1), np.uint8))[:s, :s]
    side = np.concatenate([f2, w2], 0)
    if side.shape[0] < H:
        side = np.pad(side, ((0, H - side.shape[0]), (0, 0), (0, 0)))
    panel = np.concatenate([viz, side], 1) if viz is not None else side
    if ok:  # green border on success
        panel = panel.copy()
        panel[:4], panel[-4:], panel[:, :4], panel[:, -4:] = (40, 200, 40), (40, 200, 40), (40, 200, 40), (40, 200, 40)
    return panel
