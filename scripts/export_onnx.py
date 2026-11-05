"""Export a trained policy to ONNX (end-to-end: raw obs -> unnormalized action chunk) and verify
it against PyTorch with onnxruntime.

Diffusion Policy graph: the DDIM loop (num_inference_steps) is unrolled; the initial noise is
an explicit input so the result is deterministic and reproducible.

    python scripts/export_onnx.py --ckpt runs/dp_reduced/best.pt --out exports/dp_reduced.onnx
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn

from dpick.policy_io import load_checkpoint


class ExportWrapper(nn.Module):
    def __init__(self, model, anorm, pnorm):
        super().__init__()
        self.model = model
        self.register_buffer("a_lo", torch.tensor(anorm.lo))
        self.register_buffer("a_sp", torch.tensor(anorm.span))
        self.register_buffer("p_lo", torch.tensor(pnorm.lo))
        self.register_buffer("p_sp", torch.tensor(pnorm.span))

    def forward(self, front_rgb, wrist_rgb, proprio, noise=None):
        batch = dict(front_rgb=front_rgb, wrist_rgb=wrist_rgb,
                     proprio=(proprio - self.p_lo) / self.p_sp * 2 - 1)
        a = self.model.sample(batch, noise=noise)
        return (a + 1) / 2 * self.a_sp + self.a_lo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()
    torch.set_num_threads(2)
    model, cfg, anorm, pnorm, ck = load_checkpoint(args.ckpt, "cpu")
    To, Tp, A = model.To, model.Tp, model.A
    H = cfg["img_size"]
    wrap = ExportWrapper(model, anorm, pnorm).eval()
    g = torch.Generator().manual_seed(0)
    front = torch.randint(0, 256, (1, To, 3, H, H), dtype=torch.uint8, generator=g)
    wrist = torch.randint(0, 256, (1, To, 3, H, H), dtype=torch.uint8, generator=g)
    proprio = torch.tensor(pnorm.lo + (pnorm.hi - pnorm.lo) * 0.5, dtype=torch.float32).expand(1, To, -1).contiguous()
    noise = torch.randn(1, Tp, A, generator=g)
    if cfg["policy"] == "diffusion":
        inputs = (front, wrist, proprio, noise)
        names = ["front_rgb", "wrist_rgb", "proprio", "noise"]
    else:  # ACT is deterministic at test time (z = 0)
        inputs = (front, wrist, proprio)
        names = ["front_rgb", "wrist_rgb", "proprio"]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        ref = wrap(*inputs).numpy()
        torch.onnx.export(wrap, inputs, args.out, input_names=names, output_names=["actions"],
                          opset_version=args.opset, dynamo=False,
                          dynamic_axes={n: {0: "batch"} for n in names + ["actions"]})
    import onnx
    import onnxruntime as ort
    onnx.checker.check_model(onnx.load(args.out))
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    feed = {n: x.numpy() for n, x in zip(names, inputs)}
    out = sess.run(None, feed)[0]
    err = float(np.abs(out - ref).max())
    t = time.time()
    for _ in range(20):
        sess.run(None, feed)
    ms = (time.time() - t) / 20 * 1000
    meta = dict(ckpt=args.ckpt, policy=cfg["policy"], img_size=H, obs_horizon=To, pred_horizon=Tp, action_dim=A,
                inputs={n: list(x.shape) for n, x in zip(names, inputs)}, input_dtypes={n: str(x.dtype) for n, x in zip(names, inputs)},
                output="actions [batch, pred_horizon, 5] = (x, y, z, yaw, grip)", max_abs_err_vs_torch=err,
                ort_cpu_latency_ms=round(ms, 2), size_mb=round(Path(args.out).stat().st_size / 1e6, 2),
                note=("DDIM loop unrolled; feed noise ~ N(0, I) of shape [batch, pred_horizon, 5]."
                      if cfg["policy"] == "diffusion" else "ACT with CVAE latent z = 0."))
    Path(args.out).with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))
    assert err < 1e-3, f"ONNX mismatch {err}"


if __name__ == "__main__":
    main()
