"""CNN-based Diffusion Policy (Chi et al. 2023): ResNet18 obs encoder + 1D conditional UNet,
DDPM training (epsilon prediction, squared-cosine schedule) and DDIM sampling."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ImageEncoder, SinusoidalPosEmb, augment, normalize_img

CAMS = ("front_rgb", "wrist_rgb")


# ----------------------------------------------------------------------------- noise schedule
def cosine_alphas_cumprod(T: int, s: float = 0.008) -> np.ndarray:
    def ab(t):
        return math.cos((t / T + s) / (1 + s) * math.pi / 2) ** 2
    betas = np.array([min(1 - ab(i + 1) / ab(i), 0.999) for i in range(T)], dtype=np.float64)
    return np.cumprod(1.0 - betas)


class DDIMScheduler(nn.Module):
    def __init__(self, num_train_steps=100, num_inference_steps=10):
        super().__init__()
        self.T = num_train_steps
        self.register_buffer("alphas_cumprod", torch.tensor(cosine_alphas_cumprod(num_train_steps), dtype=torch.float32))
        self.set_inference_steps(num_inference_steps)

    def set_inference_steps(self, n):
        self.S = n
        ratio = self.T // n
        self.timesteps = [int(t) for t in (np.arange(0, n) * ratio).round()[::-1]]
        self.ratio = ratio

    def add_noise(self, x0, noise, t):
        ab = self.alphas_cumprod[t].view(-1, 1, 1)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    def step(self, eps, t: int, x):
        """Deterministic DDIM (eta=0) update from t to t - ratio."""
        ab_t = self.alphas_cumprod[t]
        prev = t - self.ratio
        ab_prev = self.alphas_cumprod[prev] if prev >= 0 else torch.ones((), device=x.device)
        x0 = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
        x0 = x0.clamp(-1, 1)
        eps = (x - ab_t.sqrt() * x0) / (1 - ab_t).sqrt()
        return ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps


# ----------------------------------------------------------------------------- 1D UNet
class Conv1dBlock(nn.Module):
    def __init__(self, cin, cout, k, groups=8):
        super().__init__()
        self.block = nn.Sequential(nn.Conv1d(cin, cout, k, padding=k // 2), nn.GroupNorm(groups, cout), nn.Mish())

    def forward(self, x):
        return self.block(x)


class CondResBlock1D(nn.Module):
    def __init__(self, cin, cout, cond_dim, k=5, groups=8):
        super().__init__()
        self.b1 = Conv1dBlock(cin, cout, k, groups)
        self.b2 = Conv1dBlock(cout, cout, k, groups)
        self.film = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, 2 * cout))
        self.res = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()
        self.cout = cout

    def forward(self, x, cond):
        out = self.b1(x)
        scale, bias = self.film(cond).unsqueeze(-1).chunk(2, dim=1)
        out = scale * out + bias
        out = self.b2(out)
        return out + self.res(x)


class ConditionalUnet1D(nn.Module):
    def __init__(self, input_dim, global_cond_dim, step_embed_dim=128, down_dims=(256, 512, 1024), k=5, groups=8):
        super().__init__()
        self.step_enc = nn.Sequential(SinusoidalPosEmb(step_embed_dim), nn.Linear(step_embed_dim, 4 * step_embed_dim),
                                      nn.Mish(), nn.Linear(4 * step_embed_dim, step_embed_dim))
        cond_dim = step_embed_dim + global_cond_dim
        dims = [input_dim] + list(down_dims)
        in_out = list(zip(dims[:-1], dims[1:]))
        self.down = nn.ModuleList()
        for i, (a, b) in enumerate(in_out):
            last = i == len(in_out) - 1
            self.down.append(nn.ModuleList([CondResBlock1D(a, b, cond_dim, k, groups), CondResBlock1D(b, b, cond_dim, k, groups),
                                            nn.Conv1d(b, b, 3, 2, 1) if not last else nn.Identity()]))
        mid = dims[-1]
        self.mid = nn.ModuleList([CondResBlock1D(mid, mid, cond_dim, k, groups), CondResBlock1D(mid, mid, cond_dim, k, groups)])
        self.up = nn.ModuleList()
        for a, b in reversed(in_out[1:]):  # every up level upsamples (mirrors the down path)
            self.up.append(nn.ModuleList([CondResBlock1D(2 * b, a, cond_dim, k, groups), CondResBlock1D(a, a, cond_dim, k, groups),
                                          nn.ConvTranspose1d(a, a, 4, 2, 1)]))
        self.final = nn.Sequential(Conv1dBlock(down_dims[0], down_dims[0], k, groups), nn.Conv1d(down_dims[0], input_dim, 1))

    def forward(self, x, t, global_cond):
        """x: (B, T, A); t: (B,) ; global_cond: (B, C) -> (B, T, A)"""
        x = x.transpose(1, 2)
        cond = torch.cat([self.step_enc(t), global_cond], dim=-1)
        hs = []
        for r1, r2, ds in self.down:
            x = r2(r1(x, cond), cond)
            hs.append(x)
            x = ds(x)
        for m in self.mid:
            x = m(x, cond)
        for r1, r2, us in self.up:
            x = torch.cat([x, hs.pop()], dim=1)
            x = r2(r1(x, cond), cond)
            x = us(x)
        return self.final(x).transpose(1, 2)


# ----------------------------------------------------------------------------- policy
class DiffusionPolicy(nn.Module):
    def __init__(self, action_dim=5, proprio_dim=12, obs_horizon=2, pred_horizon=16, action_horizon=8,
                 enc_width=64, enc_stem="standard", num_kp=32, img_feat_dim=64,
                 down_dims=(256, 512, 1024), step_embed_dim=128, num_train_steps=100, num_inference_steps=10,
                 aug_pad=4, aug_jitter=0.2):
        super().__init__()
        self.hparams = dict(locals())
        self.hparams.pop("self")
        self.hparams.pop("__class__", None)
        self.To, self.Tp, self.Ta, self.A = obs_horizon, pred_horizon, action_horizon, action_dim
        self.aug_pad, self.aug_jitter = aug_pad, aug_jitter
        self.encoders = nn.ModuleDict({c: ImageEncoder(enc_width, enc_stem, num_kp, img_feat_dim) for c in CAMS})
        obs_dim = len(CAMS) * img_feat_dim + proprio_dim
        self.unet = ConditionalUnet1D(action_dim, obs_dim * obs_horizon, step_embed_dim, down_dims)
        self.sched = DDIMScheduler(num_train_steps, num_inference_steps)

    def encode_obs(self, batch, train=False):
        feats = []
        for c in CAMS:
            x = batch[c]  # B, To, 3, H, W  (uint8 or float 0..255)
            if train and (self.aug_pad or self.aug_jitter):
                x = augment(x, self.aug_pad, self.aug_jitter)
            b, t = x.shape[:2]
            f = self.encoders[c](normalize_img(x.reshape(b * t, *x.shape[2:])))
            feats.append(f.reshape(b, t, -1))
        feats.append(batch["proprio"])
        return torch.cat(feats, dim=-1).reshape(feats[0].shape[0], -1)

    def compute_loss(self, batch):
        cond = self.encode_obs(batch, train=True)
        a = batch["action"]
        noise = torch.randn_like(a)
        t = torch.randint(0, self.sched.T, (a.shape[0],), device=a.device)
        pred = self.unet(self.sched.add_noise(a, noise, t), t, cond)
        # like DP, padded (repeated last) actions past the episode end are trained on as well
        return F.mse_loss(pred, noise), {}

    @torch.no_grad()
    def sample(self, batch, noise=None):
        """Returns normalized action chunk (B, Tp, A)."""
        cond = self.encode_obs(batch, train=False)
        b = cond.shape[0]
        x = torch.randn(b, self.Tp, self.A, device=cond.device) if noise is None else noise
        for t in self.sched.timesteps:
            tt = torch.full((b,), t, device=cond.device, dtype=torch.long)
            eps = self.unet(x, tt, cond)
            x = self.sched.step(eps, t, x)
        return x
