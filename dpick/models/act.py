"""ACT baseline (Zhao et al. 2023): CVAE + transformer encoder/decoder predicting action chunks.

Image tokens come from the ResNet18 layer-4 feature map of each camera (+2D sinusoidal
position embedding and a learned camera embedding). The CVAE posterior encoder sees the
ground-truth action chunk at train time; at test time z = 0. Attention is written with
plain matmuls so the model exports cleanly to ONNX.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ResNet18, augment, normalize_img

CAMS = ("front_rgb", "wrist_rgb")


class MHA(nn.Module):
    def __init__(self, d, h, dropout=0.0):
        super().__init__()
        self.h, self.dk = h, d // h
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.drop = nn.Dropout(dropout)

    def forward(self, q, k, v, key_mask=None):
        b, nq, d = q.shape
        nk = k.shape[1]
        q = self.q(q).view(b, nq, self.h, self.dk).transpose(1, 2)
        k = self.k(k).view(b, nk, self.h, self.dk).transpose(1, 2)
        v = self.v(v).view(b, nk, self.h, self.dk).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
        if key_mask is not None:  # True = ignore
            att = att.masked_fill(key_mask[:, None, None, :], float("-inf"))
        att = self.drop(torch.softmax(att, dim=-1))
        return self.o((att @ v).transpose(1, 2).reshape(b, nq, d))


class EncLayer(nn.Module):
    def __init__(self, d, h, ff, dropout):
        super().__init__()
        self.att = MHA(d, h, dropout)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ff, d))
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pos, key_mask=None):
        qk = x + pos
        x = self.n1(x + self.drop(self.att(qk, qk, x, key_mask)))
        return self.n2(x + self.drop(self.ff(x)))


class DecLayer(nn.Module):
    def __init__(self, d, h, ff, dropout):
        super().__init__()
        self.sa, self.ca = MHA(d, h, dropout), MHA(d, h, dropout)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ff, d))
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, qpos, mem, mem_pos):
        q = x + qpos
        x = self.n1(x + self.drop(self.sa(q, q, x)))
        x = self.n2(x + self.drop(self.ca(x + qpos, mem + mem_pos, mem)))
        return self.n3(x + self.drop(self.ff(x)))


def sine_pos_2d(h, w, d, device):
    """DETR-style 2D sinusoidal embedding -> (h*w, d)."""
    npf = d // 2
    y = torch.arange(h, device=device, dtype=torch.float32).unsqueeze(1).expand(h, w) + 1
    x = torch.arange(w, device=device, dtype=torch.float32).unsqueeze(0).expand(h, w) + 1
    y = y / h * 2 * math.pi
    x = x / w * 2 * math.pi
    dim_t = 10000 ** (2 * (torch.arange(npf, device=device) // 2) / npf)
    px, py = x[..., None] / dim_t, y[..., None] / dim_t
    px = torch.stack([px[..., 0::2].sin(), px[..., 1::2].cos()], -1).flatten(2)
    py = torch.stack([py[..., 0::2].sin(), py[..., 1::2].cos()], -1).flatten(2)
    return torch.cat([py, px], -1).reshape(h * w, d)


def sine_pos_1d(n, d):
    pos = torch.arange(n, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class ACTPolicy(nn.Module):
    def __init__(self, action_dim=5, proprio_dim=12, chunk_size=16, enc_width=64, enc_stem="standard",
                 d_model=256, nheads=8, ff_dim=2048, enc_layers=4, dec_layers=4, vae_layers=4,
                 latent_dim=32, dropout=0.1, kl_weight=10.0, aug_pad=4, aug_jitter=0.2, img_size=64):
        super().__init__()
        self.hparams = dict(locals())
        self.hparams.pop("self")
        self.hparams.pop("__class__", None)
        self.To, self.Tp, self.A = 1, chunk_size, action_dim
        self.kl_weight, self.latent_dim = kl_weight, latent_dim
        self.aug_pad, self.aug_jitter = aug_pad, aug_jitter
        self.backbones = nn.ModuleDict({c: ResNet18(enc_width, enc_stem) for c in CAMS})
        cb = self.backbones[CAMS[0]].out_channels
        self.input_proj = nn.Conv2d(cb, d_model, 1)
        self.cam_embed = nn.Parameter(torch.zeros(len(CAMS), d_model))
        self.proprio_proj = nn.Linear(proprio_dim, d_model)
        self.latent_proj = nn.Linear(latent_dim, d_model)
        self.extra_pos = nn.Parameter(torch.zeros(2, d_model))  # latent, proprio tokens
        self.encoder = nn.ModuleList([EncLayer(d_model, nheads, ff_dim, dropout) for _ in range(enc_layers)])
        self.decoder = nn.ModuleList([DecLayer(d_model, nheads, ff_dim, dropout) for _ in range(dec_layers)])
        self.query_embed = nn.Parameter(torch.randn(chunk_size, d_model) * 0.02)
        self.action_head = nn.Linear(d_model, action_dim)
        # CVAE posterior encoder
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.vae_action_proj = nn.Linear(action_dim, d_model)
        self.vae_proprio_proj = nn.Linear(proprio_dim, d_model)
        self.vae_encoder = nn.ModuleList([EncLayer(d_model, nheads, ff_dim, dropout) for _ in range(vae_layers)])
        self.latent_out = nn.Linear(d_model, 2 * latent_dim)
        self.register_buffer("vae_pos", sine_pos_1d(chunk_size + 2, d_model), persistent=False)
        nn.init.normal_(self.cam_embed, std=0.02)
        nn.init.normal_(self.extra_pos, std=0.02)

    def _encode_posterior(self, proprio, actions, valid):
        b = actions.shape[0]
        x = torch.cat([self.cls.expand(b, -1, -1), self.vae_proprio_proj(proprio)[:, None],
                       self.vae_action_proj(actions)], 1)
        mask = torch.cat([torch.zeros(b, 2, dtype=torch.bool, device=x.device), ~valid], 1)
        pos = self.vae_pos[None]
        for layer in self.vae_encoder:
            x = layer(x, pos, mask)
        mu, logvar = self.latent_out(x[:, 0]).chunk(2, -1)
        return mu, logvar

    def _forward(self, batch, z, train):
        b = batch["proprio"].shape[0]
        toks, poss = [], []
        for i, c in enumerate(CAMS):
            x = batch[c][:, -1:]  # B,1,3,H,W (ACT uses the current frame only)
            if train and (self.aug_pad or self.aug_jitter):
                x = augment(x, self.aug_pad, self.aug_jitter)
            f = self.input_proj(self.backbones[c](normalize_img(x[:, 0])))  # B,d,h,w
            _, d, h, w = f.shape
            toks.append(f.flatten(2).transpose(1, 2))
            poss.append(sine_pos_2d(h, w, d, f.device) + self.cam_embed[i])
        proprio = batch["proprio"][:, -1]
        extra = torch.stack([self.latent_proj(z), self.proprio_proj(proprio)], 1)
        mem = torch.cat([extra] + toks, 1)
        pos = torch.cat([self.extra_pos] + poss, 0)[None]
        for layer in self.encoder:
            mem = layer(mem, pos)
        q = torch.zeros(b, self.Tp, mem.shape[-1], device=mem.device)
        qpos = self.query_embed[None]
        for layer in self.decoder:
            q = layer(q, qpos, mem, pos)
        return self.action_head(q)

    def compute_loss(self, batch):
        proprio = batch["proprio"][:, -1]
        act, valid = batch["action"], batch["action_valid"]
        mu, logvar = self._encode_posterior(proprio, act, valid)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        pred = self._forward(batch, z, train=True)
        l1 = (F.l1_loss(pred, act, reduction="none").mean(-1) * valid).sum() / valid.sum().clamp(min=1)
        kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(-1).mean()
        return l1 + self.kl_weight * kl, dict(l1=l1.item(), kl=kl.item())

    @torch.no_grad()
    def sample(self, batch, noise=None):
        b = batch["proprio"].shape[0]
        z = torch.zeros(b, self.latent_dim, device=batch["proprio"].device)
        return self._forward(batch, z, train=False)
