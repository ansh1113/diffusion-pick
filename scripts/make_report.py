"""Collect results/*/results.json into a markdown table + bar chart.

    python scripts/make_report.py results/dp_reduced results/act_reduced
"""
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COND_ORDER = ["nominal", "distractors", "lighting", "camera_shift"]
NAMES = {"diffusion": "Diffusion Policy (CNN, DDIM-10)", "act": "ACT"}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def main():
    dirs = [Path(p) for p in sys.argv[1:]] or sorted(Path("results").glob("*_reduced"))
    rows, data = [], []
    for d in dirs:
        r = json.loads((d / "results.json").read_text())
        data.append((d.name, r))
    conds = [c for c in COND_ORDER if any(c in r["results"] for _, r in data)]
    hdr = "| Policy | train steps | " + " | ".join(conds) + " |"
    sep = "|---|---|" + "---|" * len(conds)
    rows = [hdr, sep]
    for name, r in data:
        cells = []
        for c in conds:
            if c in r["results"]:
                x = r["results"][c]
                lo, hi = wilson(x["n_success"], x["n"])
                cells.append(f"**{100 * x['success_rate']:.0f}%** ({x['n_success']}/{x['n']}, 95% CI {100 * lo:.0f}-{100 * hi:.0f})")
            else:
                cells.append("-")
        rows.append(f"| {NAMES.get(r['policy'], r['policy'])} [`{name}`] | {r['train_step']} | " + " | ".join(cells) + " |")
    table = "\n".join(rows)
    out = Path("results")
    out.mkdir(exist_ok=True)
    (out / "results.md").write_text(table + "\n")
    print(table)

    fig, ax = plt.subplots(figsize=(7.5, 3.4), dpi=150)
    w = 0.8 / max(1, len(data))
    colors = ["#2a6fdb", "#e07b39", "#3aa676", "#8e5bd0"]
    for i, (name, r) in enumerate(data):
        xs, ys, errs = [], [], [[], []]
        for j, c in enumerate(conds):
            if c not in r["results"]:
                continue
            x = r["results"][c]
            lo, hi = wilson(x["n_success"], x["n"])
            xs.append(j + (i - (len(data) - 1) / 2) * w)
            ys.append(100 * x["success_rate"])
            errs[0].append(100 * (x["success_rate"] - lo))
            errs[1].append(100 * (hi - x["success_rate"]))
        ax.bar(xs, ys, w * 0.95, yerr=errs, capsize=3, label=NAMES.get(r["policy"], name), color=colors[i % 4])
        for x_, y_ in zip(xs, ys):
            ax.text(x_, 3, f"{y_:.0f}%", ha="center", va="bottom", fontsize=8, color="white", fontweight="bold")
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([c.replace("_", " ") for c in conds])
    ax.set_ylabel("success rate (%)")
    ax.set_ylim(0, 105)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title("Put the red cube in the bowl - 100 randomized rollouts per condition", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "success_rates.png")
    print("wrote results/results.md, results/success_rates.png")


if __name__ == "__main__":
    main()
