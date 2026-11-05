"""Closed-loop rollouts driven purely by an exported ONNX policy (onnxruntime, no PyTorch model).

    python scripts/eval_onnx.py --onnx exports/dp_reduced.onnx --n 20
"""
import argparse
import json
import os
import sys
from collections import deque
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import onnxruntime as ort

from dpick.env import EnvConfig, PickPlaceEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=1_000_000)
    ap.add_argument("--max-steps", type=int, default=150)
    args = ap.parse_args()
    meta = json.loads(Path(args.onnx).with_suffix(".json").read_text())
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    sess = ort.InferenceSession(args.onnx, so, providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]
    To, Tp, H = meta["obs_horizon"], meta["pred_horizon"], meta["img_size"]
    is_dp = meta["policy"] == "diffusion"
    Ta = 8
    env = PickPlaceEnv(EnvConfig(img_size=H, max_steps=args.max_steps))
    rng = np.random.default_rng(0)
    succ = 0
    for i in range(args.n):
        obs = env.reset(seed=args.seed0 + i)
        hist, queue, ok = deque(maxlen=To), deque(), False
        for t in range(args.max_steps):
            hist.append(obs)
            if not queue:
                h = list(hist)
                h = [h[0]] * (To - len(h)) + h
                feed = {"front_rgb": np.stack([o["front_rgb"] for o in h]).transpose(0, 3, 1, 2)[None],
                        "wrist_rgb": np.stack([o["wrist_rgb"] for o in h]).transpose(0, 3, 1, 2)[None],
                        "proprio": np.stack([o["proprio"] for o in h])[None].astype(np.float32)}
                if "noise" in names:
                    feed["noise"] = rng.standard_normal((1, Tp, 5)).astype(np.float32)
                chunk = sess.run(None, feed)[0][0]
                start = To - 1 if is_dp else 0
                queue.extend(chunk[start:start + Ta])
            obs, _, done, info = env.step(queue.popleft())
            if info["success"]:
                ok = True
                break
        succ += ok
        print(f"seed {args.seed0 + i}: {'success' if ok else 'fail'} ({t + 1} steps)", flush=True)
    env.close()
    print(json.dumps(dict(onnx=args.onnx, n=args.n, success_rate=succ / args.n)))


if __name__ == "__main__":
    main()
