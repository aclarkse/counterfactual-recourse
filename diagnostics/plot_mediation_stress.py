"""
plot_mediation_stress.py — Closure against the true NIE on German-Synth.

One panel per classifier. Each point is a German-Synth version (mediation
scale kappa); x is the oracle classifier-scale NIE and y is recipient-level
closure, both as the mean over seeds, with ±1 SD across seeds as error bars.

Usage
-----
  python diagnostics/plot_mediation_stress.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from paper_style import (
    PALETTE, LW, MS,
    set_paper_style, save_figure,
)

set_paper_style()

VERSIONS = [  # (dataset, kappa)
    ("german_synth_m0", 0.0),
    ("german_synth_m05", 0.5),
    ("german_synth", 1.0),
    ("german_synth_m2", 2.0),
]
MODELS = [
    ("Logistic Reg.", "Logistic regression"),
    ("MLP (64--32)", "MLP"),
    ("Random Forest", "Random forest"),
]
METHODS = [  # (scenario, label, color, marker, linestyle)
    ("ordinary_actionable_recourse", "Ordinary recourse", PALETTE[0], "o", "--"),
    ("mediation_aware_recourse", "Ours (parity-constrained)", PALETTE[1], "s", "-"),
]


def oracle_nie(root: Path, dataset: str, model: str) -> list[float]:
    values = []
    for path in sorted(root.glob(f"{dataset}/seed_*/gaps/{dataset}_gender_gap_oracle.json")):
        with path.open(encoding="utf-8") as f:
            values.append(json.load(f)["models"][model]["oracle"]["nie"])
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/multiseed"))
    parser.add_argument("--save-dir", default="figures/german_synth")
    args = parser.parse_args()

    with (args.root / "summary.json").open(encoding="utf-8") as f:
        seed_rows = json.load(f)["seed_level"]

    fig, axes = plt.subplots(1, len(MODELS), figsize=(6.5, 2.1), sharey=True)
    for ax, (model, title) in zip(axes, MODELS):
        for scenario, label, color, marker, style in METHODS:
            xs, xerr, ys, yerr = [], [], [], []
            for dataset, _ in VERSIONS:
                nie = oracle_nie(args.root, dataset, model)
                closure = [100 * row["recipient_level_closure"]
                           for row in seed_rows
                           if row["dataset"] == dataset and row["model"] == model
                           and row["scenario"] == scenario]
                xs.append(np.mean(nie)); xerr.append(np.std(nie, ddof=1))
                ys.append(np.mean(closure)); yerr.append(np.std(closure, ddof=1))
            ax.errorbar(xs, ys, xerr=xerr, yerr=yerr, color=color, marker=marker,
                        linestyle=style, lw=LW, ms=MS + 1, capsize=2,
                        elinewidth=0.8, label=label)
        ax.set_title(title)
        ax.set_xlabel(r"True $\mathrm{NIE}$")
        ax.set_xlim(-0.03, 0.34)
    axes[0].set_ylabel(r"Closure (\%)")
    axes[0].set_ylim(50, 100)
    for ax in axes:
        ax.grid(axis="y", color="0.9", lw=0.6)
        ax.set_axisbelow(True)

    handles = [Line2D([], [], color=c, marker=m, linestyle=s, lw=LW, ms=MS + 1)
               for _, _, c, m, s in METHODS]
    # A horizontal legend above the panels keeps the full width for the data.
    fig.legend(handles=handles, labels=[m[1] for m in METHODS],
               loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = save_figure(fig, "closure_vs_true_nie", save_dir=args.save_dir,
                           ext=ext)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
