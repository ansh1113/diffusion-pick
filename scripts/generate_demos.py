"""Generate scripted-expert demonstrations and save them to zarr.

    python scripts/generate_demos.py --n 200 --img-size 64 --out data/demos_64.zarr
"""
import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _worker(args):
    seeds, img_size, noise = args
    os.environ.setdefault("LP_NUM_THREADS", "1")
    from dpick.env import EnvConfig, PickPlaceEnv
    from dpick.expert import rollout_expert
    env = PickPlaceEnv(EnvConfig(img_size=img_size))
    out = []
    for s in seeds:
        ep, ok = rollout_expert(env, int(s), noise_std=noise)
        out.append((int(s), ok, ep if ok else None))
    env.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--img-size", type=int, default=64)
    ap.add_argument("--out", type=str, default="data/demos_64.zarr")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--noise", type=float, default=0.004, help="DART noise on executed actions (m)")
    ap.add_argument("--seed-offset", type=int, default=0, help="demo seeds start here (eval uses 1_000_000+)")
    args = ap.parse_args()

    from dpick.dataset import write_zarr
    t0 = time.time()
    episodes, seeds_used, n_fail = [], [], 0
    next_seed = args.seed_offset
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        while len(episodes) < args.n:
            need = args.n - len(episodes)
            batch = int(need * 1.1) + args.workers
            seeds = list(range(next_seed, next_seed + batch))
            next_seed += batch
            chunks = [(seeds[i::args.workers], args.img_size, args.noise) for i in range(args.workers)]
            for res in pool.imap_unordered(_worker, chunks):
                for s, ok, ep in res:
                    if ok:
                        episodes.append((s, ep))
                    else:
                        n_fail += 1
            print(f"{len(episodes)} ok / {n_fail} failed  ({time.time() - t0:.0f}s)", flush=True)
    episodes.sort(key=lambda x: x[0])
    episodes = episodes[: args.n]
    seeds_used = [s for s, _ in episodes]
    eps = [ep for _, ep in episodes]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_zarr(args.out, eps, attrs=dict(img_size=args.img_size, seeds=seeds_used, n_failed=n_fail,
                                         noise=args.noise, control_hz=10))
    lens = [len(e["action"]) for e in eps]
    print(f"saved {len(eps)} demos, {sum(lens)} steps (len {min(lens)}-{max(lens)}, mean {sum(lens)/len(lens):.1f}),"
          f" expert success {len(eps)}/{len(eps)+n_fail} -> {args.out}  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
