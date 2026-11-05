"""Build / save / load policies together with their normalizers."""
from __future__ import annotations

import torch

from .dataset import Normalizer
from .models.act import ACTPolicy
from .models.diffusion import DiffusionPolicy


def build_policy(cfg: dict):
    kind = cfg["policy"]
    m = dict(cfg["model"])
    if kind == "diffusion":
        m["down_dims"] = tuple(m.get("down_dims", (256, 512, 1024)))
        return DiffusionPolicy(**m)
    if kind == "act":
        return ACTPolicy(**m)
    raise ValueError(kind)


def save_checkpoint(path, cfg, model, action_norm, proprio_norm, step, extra=None):
    torch.save(dict(cfg=cfg, state_dict=model.state_dict(), action_norm=action_norm.state_dict(),
                    proprio_norm=proprio_norm.state_dict(), step=step, extra=extra or {}), path)


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_policy(ck["cfg"])
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    return model, ck["cfg"], Normalizer.from_state_dict(ck["action_norm"]), Normalizer.from_state_dict(ck["proprio_norm"]), ck
