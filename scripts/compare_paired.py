"""Paired comparison of two evaluations on the same seeds (exact McNemar test).

    python scripts/compare_paired.py results/dp_reduced results/dp_v2
"""
import json
import sys
from math import comb
from pathlib import Path


def mcnemar_exact(b, c):
    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    a_dir, b_dir = Path(sys.argv[1]), Path(sys.argv[2])
    A = json.loads((a_dir / "results.json").read_text())["results"]
    B = json.loads((b_dir / "results.json").read_text())["results"]
    rows = [f"| condition | `{a_dir.name}` | `{b_dir.name}` | Δ | fixed (A fail → B success) | broken (A success → B fail) | McNemar p |",
            "|---|---|---|---|---|---|---|"]
    for c in A:
        if c not in B:
            continue
        a = {e["seed"]: e["success"] for e in A[c]["episodes"]}
        b = {e["seed"]: e["success"] for e in B[c]["episodes"]}
        assert a.keys() == b.keys(), "evaluations must use the same seeds"
        fixed = sum(1 for s in a if not a[s] and b[s])
        broke = sum(1 for s in a if a[s] and not b[s])
        na, nb = sum(a.values()), sum(b.values())
        p = mcnemar_exact(fixed, broke)
        rows.append(f"| {c.replace('_', ' ')} | {na}% | **{nb}%** | {nb - na:+d} | {fixed} | {broke} | {p:.2g} |")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
