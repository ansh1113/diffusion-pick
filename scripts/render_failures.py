"""Re-run seeds that failed in an evaluation and save GIFs of failures (for qualitative analysis).

Rollouts are stochastic (DDIM noise), so a failed seed may succeed on a re-run; we try the failed
seeds in order and keep the first `--k` episodes that fail again.

    python scripts/render_failures.py --results results/dp_reduced/results.json --k 1
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import imageio
import torch

from dpick.env import EnvConfig, PickPlaceEnv
from dpick.rollout import PolicyRunner, run_episode
from evaluate import CONDITIONS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--max-tries", type=int, default=6)
    args = ap.parse_args()
    torch.set_num_threads(2)
    res = json.loads(Path(args.results).read_text())
    gif_dir = Path(args.results).parent / "gifs"
    runner = PolicyRunner(res["ckpt"], "cpu")
    img = runner.cfg["img_size"]
    for cond, r in res["results"].items():
        failed = [e["seed"] for e in r["episodes"] if not e["success"]]
        env = PickPlaceEnv(EnvConfig(img_size=img, max_steps=150, viz_size=256, **CONDITIONS[cond]))
        saved = 0
        for seed in failed[: args.max_tries]:
            out, frames = run_episode(env, runner, seed, 150, record=True)
            if not out["success"]:
                imageio.mimsave(gif_dir / f"{cond}_seed{seed}_fail.gif", frames[::2], duration=0.2, loop=0)
                print(cond, seed, "fail gif saved", flush=True)
                saved += 1
                if saved >= args.k:
                    break
        env.close()


if __name__ == "__main__":
    main()
