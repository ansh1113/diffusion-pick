"""Recover the full simulator state for every frame of a demo zarr.

The zarr stores images, proprio and actions but not qpos. The expert is deterministic given its seed,
so re-running it (without rendering) reproduces each episode; we record qpos + the bowl position at
every observation and check the re-run proprio/actions against the zarr.

    python scripts/export_states.py --zarr data/demos_64.zarr --out data/demos_64_states.npz
"""
import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")  # no rendering happens; avoids probing EGL
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from dpick.dataset import load_zarr
from dpick.env import EnvConfig, PickPlaceEnv
from dpick.expert import rollout_expert


class StateEnv(PickPlaceEnv):
    """PickPlaceEnv whose observations are the raw state instead of rendered images."""

    def obs(self):
        z = np.zeros((1, 1, 3), np.uint8)
        return dict(front_rgb=z, wrist_rgb=z, proprio=self.proprio(), qpos=self.data.qpos.copy())


def run(args):
    seeds, noise = args
    env = StateEnv(EnvConfig(img_size=64))
    out = []
    for s in seeds:
        # rollout_expert stores obs["front_rgb"]; stash qpos through it by wrapping obs()
        qs = []
        orig = env.obs

        def obs_hook():
            o = orig()
            qs.append(o["qpos"])
            return o
        env.obs = obs_hook
        ep, ok = rollout_expert(env, int(s), noise_std=noise)
        env.obs = orig
        T = len(ep["action"])
        out.append(dict(seed=int(s), ok=ok, qpos=np.stack(qs[:T]), bowl=env.bowl_pos().copy(),
                        proprio=ep["proprio"], action=ep["action"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", default="data/demos_64.zarr")
    ap.add_argument("--out", default="data/demos_64_states.npz")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    data, ends, attrs = load_zarr(args.zarr)
    seeds, noise = attrs["seeds"], attrs.get("noise", 0.004)
    starts = np.concatenate([[0], ends[:-1]])
    chunks = [(seeds[i::args.workers], noise) for i in range(args.workers)]
    with Pool(args.workers) as pool:
        eps = {e["seed"]: e for res in pool.map(run, chunks) for e in res}
    qpos, bowl, max_dp, max_da = [], [], 0.0, 0.0
    for k, s in enumerate(seeds):
        e = eps[s]
        st, en = starts[k], ends[k]
        assert e["ok"], f"seed {s} did not succeed on re-run"
        assert len(e["action"]) == en - st, f"seed {s}: length {len(e['action'])} != {en - st}"
        max_dp = max(max_dp, float(np.abs(e["proprio"] - data["proprio"][st:en]).max()))
        max_da = max(max_da, float(np.abs(e["action"] - data["action"][st:en]).max()))
        qpos.append(e["qpos"])
        bowl.append(e["bowl"])
    qpos = np.concatenate(qpos).astype(np.float64)
    np.savez_compressed(args.out, qpos=qpos, bowl=np.stack(bowl), episode_ends=ends, seeds=np.array(seeds))
    print(f"{len(seeds)} episodes, {len(qpos)} frames -> {args.out}")
    print(f"max |proprio - zarr| = {max_dp:.2e}, max |action - zarr| = {max_da:.2e}")
    assert max_dp < 1e-4 and max_da < 1e-4, "re-run does not reproduce the zarr"


if __name__ == "__main__":
    main()
