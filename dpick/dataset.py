"""Zarr replay buffer + sequence dataset (Diffusion-Policy style layout).

zarr layout:
    data/front_rgb  (N, H, W, 3) uint8
    data/wrist_rgb  (N, H, W, 3) uint8
    data/proprio    (N, P) float32
    data/action     (N, A) float32
    meta/episode_ends (E,) int64   (exclusive end index of each episode)
"""
from __future__ import annotations

import numpy as np
import torch
import zarr

IMG_KEYS = ("front_rgb", "wrist_rgb")


def write_zarr(path, episodes, attrs=None):
    root = zarr.open_group(str(path), mode="w")
    data = root.create_group("data")
    meta = root.create_group("meta")
    keys = list(episodes[0].keys())
    ends = np.cumsum([len(ep["action"]) for ep in episodes]).astype(np.int64)
    for k in keys:
        arr = np.concatenate([ep[k] for ep in episodes], 0)
        chunks = (256,) + arr.shape[1:] if arr.ndim > 2 else (4096,) + arr.shape[1:]
        data.create_array(k, data=arr, chunks=chunks)
    meta.create_array("episode_ends", data=ends)
    if attrs:
        root.attrs.update(attrs)
    return root


def load_zarr(path):
    root = zarr.open_group(str(path), mode="r")
    data = {k: root["data"][k][:] for k in root["data"].array_keys()}
    ends = root["meta"]["episode_ends"][:]
    return data, ends, dict(root.attrs)


class Normalizer:
    """Min-max normalizer to [-1, 1] (per dim)."""

    def __init__(self, lo, hi):
        self.lo = np.asarray(lo, np.float32)
        self.hi = np.asarray(hi, np.float32)
        span = self.hi - self.lo
        span[span < 1e-4] = 1.0
        self.span = span

    @classmethod
    def fit(cls, x, margin=0.0):
        lo, hi = x.min(0), x.max(0)
        pad = (hi - lo) * margin
        return cls(lo - pad, hi + pad)

    def normalize(self, x):
        lo = torch.as_tensor(self.lo, device=x.device) if torch.is_tensor(x) else self.lo
        sp = torch.as_tensor(self.span, device=x.device) if torch.is_tensor(x) else self.span
        return (x - lo) / sp * 2.0 - 1.0

    def unnormalize(self, x):
        lo = torch.as_tensor(self.lo, device=x.device) if torch.is_tensor(x) else self.lo
        sp = torch.as_tensor(self.span, device=x.device) if torch.is_tensor(x) else self.span
        return (x + 1.0) / 2.0 * sp + lo

    def state_dict(self):
        return dict(lo=self.lo.tolist(), hi=self.hi.tolist())

    @classmethod
    def from_state_dict(cls, d):
        return cls(d["lo"], d["hi"])


class SequenceDataset(torch.utils.data.Dataset):
    """Samples (obs window of length To, action chunk of length Ta) pairs.

    The obs window ends at t; the action chunk starts at t - To + 1 (DP convention) or at
    t (ACT convention, `action_from_current=True`). Out-of-episode indices are padded by
    repeating the first / last frame.
    """

    def __init__(self, path, obs_horizon=2, pred_horizon=16, episode_idx=None, action_from_current=False):
        data, ends, self.attrs = load_zarr(path)
        starts = np.concatenate([[0], ends[:-1]])
        if episode_idx is None:
            episode_idx = np.arange(len(ends))
        self.To, self.Tp = obs_horizon, pred_horizon
        self.action_from_current = action_from_current
        self.data = data
        # images stored as uint8 NHWC; converted lazily to float CHW in __getitem__
        self.index = []
        for e in episode_idx:
            s, t = starts[e], ends[e]
            for i in range(s, t):
                self.index.append((i, s, t))
        self.index = np.array(self.index, dtype=np.int64)
        sel = np.concatenate([np.arange(starts[e], ends[e]) for e in episode_idx])
        self.action_norm = Normalizer.fit(data["action"][sel])
        self.proprio_norm = Normalizer.fit(data["proprio"][sel])

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        i, s, t = self.index[k]
        obs_idx = np.clip(np.arange(i - self.To + 1, i + 1), s, t - 1)
        a0 = i if self.action_from_current else i - self.To + 1
        act_idx = np.arange(a0, a0 + self.Tp)
        pad_mask = (act_idx < t)  # True = valid
        act_idx = np.clip(act_idx, s, t - 1)
        out = {}
        for key in IMG_KEYS:
            out[key] = torch.from_numpy(self.data[key][obs_idx]).permute(0, 3, 1, 2).contiguous()  # To,3,H,W uint8
        out["proprio"] = torch.from_numpy(self.proprio_norm.normalize(self.data["proprio"][obs_idx]))
        out["action"] = torch.from_numpy(self.action_norm.normalize(self.data["action"][act_idx]))
        out["action_valid"] = torch.from_numpy(pad_mask)
        return out
