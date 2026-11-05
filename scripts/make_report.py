"""Collect results/*/results.json into a markdown table + bar chart.

    python scripts/make_report.py results/dp_reduced results/act_reduced
    python scripts/make_report.py --out results/v1_vs_v2 "DP v1=results/dp_reduced" "DP v2=results/dp_v2"
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
    args = sys.argv[1:]
    out_stem = Path("results/results")
    if args and args[0] == "--out":
        out_stem, args = Path(args[1]), args[2:]
    title = "Put the red cube in the bowl - 100 randomized rollouts per condition"
    if args and args[0] == "--title":
        title, args = args[1], args[2:]
    specs = []
    for a in args or [str(p) for p in sorted(Path("results").glob("*_reduced"))]:
        label, path = a.split("=", 1) if "=" in a else (None, a)
        specs.append((label, Path(path)))
    rows, data, labels = [], [], []
    for label, d in specs:
        r = json.loads((d / "results.json").read_text())
        data.append((d.name, r))
        labels.append(label or NAMES.get(r["policy"], r["policy"]))
    conds = [c for c in COND_ORDER if any(c in r["results"] for _, r in data)]
    hdr = "| Policy | train steps | " + " | ".join(conds) + " |"
    sep = "|---|---|" + "---|" * len(conds)
    rows = [hdr, sep]
    for (name, r), label in zip(data, labels):
        cells = []
        for c in conds:
            if c in r["results"]:
                x = r["results"][c]
                lo, hi = wilson(x["n_success"], x["n"])
                cells.append(f"**{100 * x['success_rate']:.0f}%** ({x['n_success']}/{x['n']}, 95% CI {100 * lo:.0f}-{100 * hi:.0f})")
            else:
                cells.append("-")
        rows.append(f"| {label} [`{name}`] | {r['train_step']} | " + " | ".join(cells) + " |")
    table = "\n".join(rows)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    out_stem.with_suffix(".md").write_text(table + "\n")
    print(table)

    fig, ax = plt.subplots(figsize=(7.5 + 0.6 * max(0, len(data) - 2), 3.4), dpi=150)
    w = 0.8 / max(1, len(data))
    colors = ["#2a6fdb", "#e07b39", "#3aa676", "#8e5bd0", "#c94f6d", "#7a7a7a"]
    for i, ((name, r), label) in enumerate(zip(data, labels)):
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
        ax.bar(xs, ys, w * 0.95, yerr=errs, capsize=3, label=label, color=colors[i % len(colors)])
        for x_, y_ in zip(xs, ys):
            ax.text(x_, 3, f"{y_:.0f}%", ha="center", va="bottom", fontsize=8, color="white", fontweight="bold")
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([c.replace("_", " ") for c in conds])
    ax.set_ylabel("success rate (%)")
    ax.set_ylim(0, 105)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title, fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=min(3, len(data)))
    fig.tight_layout()
    png = out_stem.parent / ("success_rates.png" if out_stem.name == "results" else out_stem.name + ".png")
    fig.savefig(png)
    print(f"wrote {out_stem.with_suffix('.md')}, {png}")


if __name__ == "__main__":
    main()
