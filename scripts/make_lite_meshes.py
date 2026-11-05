"""Decimate the Menagerie Panda visual meshes for fast software (llvmpipe/OSMesa) rendering.

Writes assets/franka_emika_panda/assets_lite/*.obj (same file names, ~12% of the faces).
The collision meshes and kinematics are untouched; only the rendered look is simplified.
"""
import argparse
from pathlib import Path

import fast_simplification
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parent.parent / "assets" / "franka_emika_panda"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reduction", type=float, default=0.88)
    args = ap.parse_args()
    out = ROOT / "assets_lite"
    out.mkdir(exist_ok=True)
    tot_in = tot_out = 0
    for f in sorted((ROOT / "assets").glob("*.obj")):
        m = trimesh.load(f, force="mesh")
        # OBJ verts are duplicated per normal/uv; merge by position so the decimator sees a connected mesh
        m = trimesh.Trimesh(m.vertices, m.faces, process=True)
        m.merge_vertices()
        v, fc = np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)
        if len(fc) > 200:
            v2, f2 = fast_simplification.simplify(v, fc, target_reduction=args.reduction)
        else:
            v2, f2 = v, fc
        tot_in += len(fc)
        tot_out += len(f2)
        with open(out / f.name, "w") as fh:
            for p in v2:
                fh.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
            for t in f2:
                fh.write(f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}\n")
    print(f"faces {tot_in} -> {tot_out}")


if __name__ == "__main__":
    main()
