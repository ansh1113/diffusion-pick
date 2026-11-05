"""Fine-tune a trained Diffusion Policy on a mix of datasets (e.g. MuJoCo-rendered + three.js-rendered
copies of the same demos), keeping the checkpoint's normalizers so action/proprio scaling is unchanged.

Each batch is drawn equally from every dataset (50/50 for two). On top of the model's own
augmentation (random shift + brightness/contrast/saturation jitter, raised to --jitter), a per-channel
colour gain (--tint) is applied, so the encoder becomes less sensitive to renderer colour/lighting.

    python scripts/finetune.py --ckpt runs/dp_reduced/best.pt \
        --data data/demos_64.zarr data/demos_64_threejs.zarr --out runs/dp_browser --minutes 80
"""
import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from dpick.dataset import IMG_KEYS, SequenceDataset
from dpick.models.common import EMA
from dpick.policy_io import load_checkpoint, save_checkpoint
from scripts.train import batch_iter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=12000, help="upper bound; --minutes usually stops first")
    ap.add_argument("--minutes", type=float, default=80)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--jitter", type=float, default=0.3, help="brightness/contrast/saturation jitter (was 0.2)")
    ap.add_argument("--tint", type=float, default=0.12, help="per-channel colour gain range +/-")
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model, cfg, anorm, pnorm, ck = load_checkpoint(args.ckpt, "cpu")
    model.aug_jitter = args.jitter
    cfg = copy.deepcopy(cfg)
    cfg["model"]["aug_jitter"] = args.jitter
    cfg["finetune"] = dict(vars(args), base_ckpt=args.ckpt, base_step=ck["step"])
    To, Tp = model.To, model.Tp
    datasets = []
    for p in args.data:
        ds = SequenceDataset(p, obs_horizon=To, pred_horizon=Tp)
        ds.action_norm, ds.proprio_norm = anorm, pnorm  # keep the original scaling
        datasets.append(ds)
        print(f"{p}: {len(ds)} samples", flush=True)
    per = args.batch // len(datasets)
    iters = [batch_iter(ds, per, device, rng) for ds in datasets]

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6, betas=(0.95, 0.999))
    ema = EMA(model)
    log = open(out / "train_log.jsonl", "w")
    t0, losses = time.time(), []
    step = 0
    for step in range(args.steps):
        frac = min(1.0, (time.time() - t0) / (60 * args.minutes))  # cosine over the time budget
        lr = args.lr * min(1.0, (step + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * frac))
        for g in opt.param_groups:
            g["lr"] = lr
        parts = [next(it) for it in iters]
        batch = {k: torch.cat([p[k] for p in parts], 0) for k in parts[0]}
        if args.tint > 0:
            for k in IMG_KEYS:  # uint8 (B, T, 3, H, W) -> float with a per-sample, per-channel gain
                g = 1 + (torch.rand(batch[k].shape[0], 1, 3, 1, 1) * 2 - 1) * args.tint
                batch[k] = (batch[k].float() * g).clamp(0, 255)
        model.train()
        loss, info = model.compute_loss(batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        ema.update(model)
        losses.append(loss.item())
        done = frac >= 1.0
        if (step + 1) % 100 == 0 or done:
            rec = dict(step=step + 1, loss=float(np.mean(losses)), lr=lr, min=round((time.time() - t0) / 60, 1), **info)
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            losses = []
        if (step + 1) % args.ckpt_every == 0 or done:
            save_checkpoint(out / "last.pt", cfg, ema.model, anorm, pnorm, ck["step"] + step + 1)
            if (step + 1) % args.ckpt_every == 0:
                save_checkpoint(out / f"step_{step + 1:06d}.pt", cfg, ema.model, anorm, pnorm, ck["step"] + step + 1)
        if done:
            break
    print(f"done: {step + 1} steps in {(time.time() - t0) / 60:.1f} min -> {out / 'last.pt'}", flush=True)


if __name__ == "__main__":
    main()
