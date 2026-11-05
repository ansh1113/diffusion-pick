"""Render GIFs for specific evaluation seeds (e.g. episodes where the policy re-grasped).

    python scripts/render_seeds.py --ckpt runs/dp_v2/best.pt --condition nominal --seeds 1000012 --out results/dp_v2/gifs --tag recovery
"""
import argparse
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
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--condition", default="nominal")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--want-regrasp", action="store_true", help="only keep successful episodes with >1 grasp attempt")
    ap.add_argument("--tries", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(2)
    runner = PolicyRunner(args.ckpt, "cpu")
    env = PickPlaceEnv(EnvConfig(img_size=runner.cfg["img_size"], max_steps=150, viz_size=256,
                                 **CONDITIONS[args.condition]))
    Path(args.out).mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        for _ in range(args.tries):
            res, frames = run_episode(env, runner, seed, 150, record=True)
            if args.want_regrasp and not (res["success"] and res["grasp_attempts"] > 1):
                continue
            tag = ("success" if res["success"] else "fail") + (f"_{args.tag}" if args.tag else "")
            path = Path(args.out) / f"{args.condition}_seed{seed}_{tag}.gif"
            imageio.mimsave(path, frames[::2], duration=0.2, loop=0)
            print(path, res, flush=True)
            break
    env.close()


if __name__ == "__main__":
    main()
