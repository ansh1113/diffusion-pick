"""Evaluate a trained policy: randomized rollouts under nominal + robustness conditions.

    python scripts/evaluate.py --ckpt runs/dp_reduced/last.pt --n 100 --out results/dp_reduced
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CONDITIONS = {
    "nominal": dict(),
    "distractors": dict(n_distractors=3),
    "lighting": dict(light_perturb=True),
    "camera_shift": dict(camera_shift=True),
}
TEST_SEED0 = 1_000_000  # demos use seeds < 1e5; test layouts are disjoint


def _worker(job):
    ckpt, cond, tasks, img_size, max_steps, gif_dir, extra = job
    os.environ.setdefault("LP_NUM_THREADS", "1")
    import imageio
    import torch
    torch.set_num_threads(1)
    from dpick.env import EnvConfig, PickPlaceEnv
    from dpick.rollout import PolicyRunner, run_episode
    runner = PolicyRunner(ckpt, "cpu", **extra)
    out = []
    envs = {}
    for seed, gif in tasks:
        key = bool(gif)
        if key not in envs:
            envs[key] = PickPlaceEnv(EnvConfig(img_size=img_size, max_steps=max_steps, viz_size=256 if gif else 0,
                                               **CONDITIONS[cond]))
        res, frames = run_episode(envs[key], runner, seed, max_steps, record=bool(gif))
        if gif:
            tag = "success" if res["success"] else "fail"
            imageio.mimsave(Path(gif_dir) / f"{cond}_seed{seed}_{tag}.gif", frames[::1], duration=0.1, loop=0)
        res.update(seed=seed, condition=cond)
        out.append(res)
    for e in envs.values():
        e.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--max-steps", type=int, default=150)
    ap.add_argument("--gifs", type=int, default=2, help="GIFs saved per condition")
    ap.add_argument("--seed0", type=int, default=TEST_SEED0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-ensemble", action="store_true", help="ACT: execute chunks instead of temporal ensembling")
    args = ap.parse_args()

    import torch
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    img_size = ck["cfg"]["img_size"]
    out = Path(args.out)
    gif_dir = out / "gifs"
    gif_dir.mkdir(parents=True, exist_ok=True)
    extra = dict(act_temporal_ensemble=not args.no_ensemble) if ck["cfg"]["policy"] == "act" else {}
    summary = dict(ckpt=args.ckpt, policy=ck["cfg"]["policy"], train_step=ck["step"], n=args.n, results={})
    ctx = mp.get_context("spawn")
    t0 = time.time()
    with ctx.Pool(args.workers) as pool:
        for cond in args.conditions.split(","):
            seeds = [args.seed0 + i for i in range(args.n)]
            tasks = [(s, i < args.gifs) for i, s in enumerate(seeds)]
            jobs = [(args.ckpt, cond, tasks[w::args.workers], img_size, args.max_steps, str(gif_dir), extra)
                    for w in range(args.workers)]
            res = [r for chunk in pool.map(_worker, jobs) for r in chunk]
            res.sort(key=lambda r: r["seed"])
            sr = sum(r["success"] for r in res) / len(res)
            succ_steps = [r["steps"] for r in res if r["success"]]
            summary["results"][cond] = dict(
                success_rate=sr, n=len(res), n_success=int(sum(r["success"] for r in res)),
                mean_steps_success=float(sum(succ_steps) / len(succ_steps)) if succ_steps else None,
                episodes=res)
            print(f"[{time.time() - t0:.0f}s] {cond}: {sr * 100:.1f}% ({len(res)} eps)", flush=True)
    with open(out / "results.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: v["success_rate"] for k, v in summary["results"].items()}))


if __name__ == "__main__":
    main()
