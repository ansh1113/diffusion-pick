"""Shared building blocks: device selection, ResNet18 encoder (GroupNorm), augmentation."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_device(pref: str = "auto") -> torch.device:
    """cuda > mps > cpu unless explicitly requested."""
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _gn(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(max(1, min(32, c // 4)), c)


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.n1 = _gn(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.n2 = _gn(cout)
        self.down = None
        if stride != 1 or cin != cout:
            self.down = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), _gn(cout))

    def forward(self, x):
        idt = x if self.down is None else self.down(x)
        x = F.relu(self.n1(self.conv1(x)))
        x = self.n2(self.conv2(x))
        return F.relu(x + idt)


class ResNet18(nn.Module):
    """ResNet-18 ([2,2,2,2] BasicBlocks) with GroupNorm and a configurable width.

    width=64 + stem='standard' is the torchvision ResNet18 topology (BatchNorm -> GroupNorm,
    as in Diffusion Policy). stem='small' (3x3/s2 conv, no max-pool) suits 64x64 inputs.
    Output: feature map (B, 8*width, H/16 or H/32...).
    """

    def __init__(self, width: int = 64, stem: str = "standard"):
        super().__init__()
        w = width
        if stem == "standard":
            self.stem = nn.Sequential(nn.Conv2d(3, w, 7, 2, 3, bias=False), _gn(w), nn.ReLU(),
                                      nn.MaxPool2d(3, 2, 1))
        else:
            self.stem = nn.Sequential(nn.Conv2d(3, w, 3, 2, 1, bias=False), _gn(w), nn.ReLU())
        layers, cin = [], w
        for i, c in enumerate([w, 2 * w, 4 * w, 8 * w]):
            stride = 1 if i == 0 else 2
            layers += [BasicBlock(cin, c, stride), BasicBlock(c, c, 1)]
            cin = c
        self.layers = nn.Sequential(*layers)
        self.out_channels = cin

    def forward(self, x):
        return self.layers(self.stem(x))


class SpatialSoftmax(nn.Module):
    """Keypoint extraction: per-channel softmax over space -> expected (x, y)."""

    def __init__(self, in_ch: int, num_kp: int = 32):
        super().__init__()
        self.kp = nn.Conv2d(in_ch, num_kp, 1)
        self.num_kp = num_kp

    def forward(self, x):
        x = self.kp(x)
        b, k, h, w = x.shape
        ys = torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype)
        xs = torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        att = torch.softmax(x.reshape(b, k, h * w), dim=-1)
        ex = (att * gx.reshape(1, 1, -1)).sum(-1)
        ey = (att * gy.reshape(1, 1, -1)).sum(-1)
        return torch.cat([ex, ey], dim=-1)  # b, 2k


IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD = (0.229, 0.224, 0.225)


def normalize_img(x_uint8: torch.Tensor) -> torch.Tensor:
    """uint8/float [0,255] (..., 3, H, W) -> ImageNet-normalized float."""
    x = x_uint8.float() / 255.0
    mean = torch.tensor(IMNET_MEAN, device=x.device).view(3, 1, 1)
    std = torch.tensor(IMNET_STD, device=x.device).view(3, 1, 1)
    return (x - mean) / std


class ImageEncoder(nn.Module):
    """ResNet18 -> spatial softmax -> linear. One per camera."""

    def __init__(self, width=64, stem="standard", num_kp=32, out_dim=64):
        super().__init__()
        self.backbone = ResNet18(width, stem)
        self.pool = SpatialSoftmax(self.backbone.out_channels, num_kp)
        self.proj = nn.Sequential(nn.Linear(2 * num_kp, out_dim), nn.ReLU())
        self.out_dim = out_dim

    def forward(self, x):  # x: (B, 3, H, W) normalized
        return self.proj(self.pool(self.backbone(x)))


def augment(imgs: torch.Tensor, pad: int = 4, jitter: float = 0.2) -> torch.Tensor:
    """Train-time augmentation on uint8 image batches (B, T, 3, H, W) -> same dtype float.

    Random shift (replicate pad + crop, same shift across the T frames of a sample) and
    per-sample brightness / contrast / saturation jitter.
    """
    b, t, c, h, w = imgs.shape
    x = imgs.float().reshape(b, t * c, h, w)
    if pad > 0:
        x = F.pad(x, (pad, pad, pad, pad), mode="replicate")
        # integer shifts via a single grid_sample with nearest sampling
        eps = 1.0 / (h + 2 * pad)
        ar = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * pad, device=x.device)[:h]
        base = torch.stack(torch.meshgrid(ar, ar, indexing="ij")[::-1], -1)  # h,w,2 (x,y)
        shift = torch.randint(0, 2 * pad + 1, (b, 1, 1, 2), device=x.device).float() * (2.0 / (h + 2 * pad))
        grid = base.unsqueeze(0) + shift
        x = F.grid_sample(x, grid, mode="nearest", padding_mode="border", align_corners=False)
    x = x.reshape(b, t, c, h, w) / 255.0
    if jitter > 0:
        br = 1 + (torch.rand(b, 1, 1, 1, 1, device=x.device) * 2 - 1) * jitter
        ct = 1 + (torch.rand(b, 1, 1, 1, 1, device=x.device) * 2 - 1) * jitter
        sat = 1 + (torch.rand(b, 1, 1, 1, 1, device=x.device) * 2 - 1) * jitter
        x = x * br
        m = x.mean(dim=(2, 3, 4), keepdim=True)
        x = (x - m) * ct + m
        g = x.mean(dim=2, keepdim=True)
        x = (x - g) * sat + g
    return (x.clamp(0, 1) * 255.0)


def augment_strong(imgs: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """v2 photometric augmentation (train time only), (B, T, 3, H, W) uint8 -> float 0..255.

    Random shift, then per sample (shared across the T frames): brightness x[0.55, 1.45] (log-uniform),
    contrast x[0.65, 1.35], saturation x[0.7, 1.4], per-channel gain x[0.8, 1.2] (color tint),
    a random linear illumination gradient (+-35% across the image, random direction) and pixel noise.
    No hue rotation, so red stays red.
    """
    x = augment(imgs, pad, jitter=0.0) / 255.0
    b, t, c, h, w = x.shape
    dev = x.device

    def u(lo, hi, *shape):
        return lo + (hi - lo) * torch.rand(b, *shape, device=dev)

    bright = torch.exp(u(math.log(0.55), math.log(1.45), 1, 1, 1, 1))
    gain = u(0.8, 1.2, 1, 3, 1, 1)
    x = x * bright * gain
    m = x.mean(dim=(2, 3, 4), keepdim=True)
    x = (x - m) * u(0.65, 1.35, 1, 1, 1, 1) + m
    g = x.mean(dim=2, keepdim=True)
    x = (x - g) * u(0.7, 1.4, 1, 1, 1, 1) + g
    ang = u(0, 2 * math.pi, 1, 1, 1, 1)
    ys = torch.linspace(-1, 1, h, device=dev).view(1, 1, 1, h, 1)
    xs = torch.linspace(-1, 1, w, device=dev).view(1, 1, 1, 1, w)
    x = x * (1 + u(0, 0.35, 1, 1, 1, 1) * (torch.cos(ang) * xs + torch.sin(ang) * ys))
    x = x + torch.randn_like(x) * u(0, 0.02, 1, 1, 1, 1)
    return x.clamp(0, 1) * 255.0


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=x.device, dtype=torch.float32) * -emb)
        emb = x.float()[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class EMA:
    """EMA of model parameters with DP-style warmup (decay = 1 - (1+step/inv_gamma)^-power)."""

    def __init__(self, model: nn.Module, power=0.75, max_decay=0.9999):
        import copy
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.power, self.max_decay, self.step = power, max_decay, 0

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.step += 1
        decay = 1 - (1 + self.step) ** -self.power
        decay = min(max(decay, 0.0), self.max_decay)
        for pe, p in zip(self.model.parameters(), model.parameters()):
            pe.mul_(decay).add_(p.detach(), alpha=1 - decay)
        for be, b in zip(self.model.buffers(), model.buffers()):
            be.copy_(b)
