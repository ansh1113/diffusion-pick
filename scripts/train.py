"""Train a Diffusion Policy or ACT policy from a zarr demo file.

    python scripts/train.py --config configs/dp_reduced.yaml
    python scripts/train.py --config configs/dp_full.yaml          # A10 / any CUDA GPU
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml

from dpick.dataset import IMG_KEYS, SequenceDataset
from dpick.models.common import EMA, get_device
from dpick.policy_io import build_policy, save_checkpoint


def batch_iter(ds: SequenceDataset, batch_size, device, rng):
    """Vectorized batch sampler (fast on CPU, no worker processes needed)."""
    idx_all = ds.index
    To, Tp = ds.To, ds.Tp
    d = ds.data
    pin = device.type == "cuda"
    while True:
        perm = rng.permutation(len(idx_all))
        for s in range(0, len(perm) - batch_size + 1, batch_size):
            sel = idx_all[perm[s:s + batch_size]]
            i, st, en = sel[:, 0:1], sel[:, 1:2], sel[:, 2:3]
            obs_idx = np.clip(i + np.arange(-To + 1, 1)[None], st, en - 1)
            a0 = i if ds.action_from_current else i - To + 1
            act_idx = a0 + np.arange(Tp)[None]
            valid = act_idx < en
            act_idx = np.clip(act_idx, st, en - 1)
            b = {}
            for k in IMG_KEYS:
                x = torch.from_numpy(d[k][obs_idx]).permute(0, 1, 4, 2, 3)
                b[k] = (x.pin_memory() if pin else x).to(device, non_blocking=True)
            b["proprio"] = torch.from_numpy(ds.proprio_norm.normalize(d["proprio"][obs_idx])).to(device)
            b["action"] = torch.from_numpy(ds.action_norm.normalize(d["action"][act_idx])).to(device)
            b["action_valid"] = torch.from_numpy(valid).to(device)
            yield b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    tc = cfg["train"]
    if args.steps:
        tc["steps"] = args.steps
    out = Path(args.out or f"runs/{Path(args.config).stem}")
    out.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(tc.get("seed", 0))
    rng = np.random.default_rng(tc.get("seed", 0))
    print(f"device={device}  threads={torch.get_num_threads()}  out={out}", flush=True)

    m = cfg["model"]
    is_act = cfg["policy"] == "act"
    To = 1 if is_act else m["obs_horizon"]
    Tp = m["chunk_size"] if is_act else m["pred_horizon"]
    ds = SequenceDataset(cfg["data"], obs_horizon=To, pred_horizon=Tp, action_from_current=is_act)
    assert ds.attrs.get("img_size", cfg["img_size"]) == cfg["img_size"], "dataset / config image size mismatch"
    print(f"dataset: {len(ds)} samples from {cfg['data']}", flush=True)

    model = build_policy(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{cfg['policy']}: {n_params / 1e6:.2f}M params", flush=True)
    if is_act:
        bb = [p for n, p in model.named_parameters() if n.startswith("backbones")]
        rest = [p for n, p in model.named_parameters() if not n.startswith("backbones")]
        groups = [dict(params=rest), dict(params=bb, lr=tc.get("lr_backbone", tc["lr"]))]
    else:
        groups = [dict(params=list(model.parameters()))]
    opt = torch.optim.AdamW(groups, lr=tc["lr"], weight_decay=tc.get("weight_decay", 1e-6), betas=(0.95, 0.999))
    base_lrs = [g["lr"] for g in opt.param_groups]
    ema = EMA(model) if tc.get("ema", False) else None
    step0 = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["extra"]["raw_state_dict"])
        opt.load_state_dict(ck["extra"]["opt"])
        if ema is not None:
            ema.model.load_state_dict(ck["state_dict"])
            ema.step = ck["step"]
        step0 = ck["step"]

    use_amp = device.type == "cuda" and tc.get("amp", True)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    it = batch_iter(ds, tc["batch_size"], device, rng)
    log = open(out / "train_log.jsonl", "a")
    t0, losses = time.time(), []
    warm, total = tc.get("warmup", 500), tc["steps"]
    for step in range(step0, total):
        lr_scale = min(1.0, (step + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, step / total)))
        for g, bl in zip(opt.param_groups, base_lrs):
            g["lr"] = bl * lr_scale
        batch = next(it)
        model.train()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            loss, info = model.compute_loss(batch)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc.get("grad_clip", 10.0))
        scaler.step(opt)
        scaler.update()
        if ema is not None:
            ema.update(model)
        losses.append(loss.item())
        if (step + 1) % tc.get("log_every", 100) == 0:
            rec = dict(step=step + 1, loss=float(np.mean(losses)), lr=opt.param_groups[0]["lr"],
                       sec=round(time.time() - t0, 1), **info)
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            losses = []
        if (step + 1) % tc.get("ckpt_every", 5000) == 0 or step + 1 == total:
            pol = ema.model if ema is not None else model
            extra = dict(raw_state_dict=model.state_dict(), opt=opt.state_dict())
            save_checkpoint(out / "last.pt", cfg, pol, ds.action_norm, ds.proprio_norm, step + 1, extra)
            if tc.get("keep_ckpts", True):
                save_checkpoint(out / f"step_{step + 1:06d}.pt", cfg, pol, ds.action_norm, ds.proprio_norm, step + 1)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
