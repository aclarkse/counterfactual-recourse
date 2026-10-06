"""
plot_gamma_sensitivity.py — Closure and abstention causes against gamma (ACS).

Left: recipient-level closure of both methods as the validity level gamma
grows. Thin lines are individual classifiers (mean over seeds); the thick
line averages them. Right: our method's abstention rate, split into causes
(a) no gamma-valid plan, (b) epsilon < epsilon_gamma, (c) unreachable, as
the mean over classifiers and seeds. The dotted line marks where the median
compatibility threshold epsilon_gamma reaches epsilon.

Usage
-----
  python diagnostics/plot_gamma_sensitivity.py
  python diagnostics/plot_gamma_sensitivity.py --dataset german_synth
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from paper_style import PALETTE, LW, MS, set_paper_style, save_figure

set_paper_style()

METHODS = [  # (scenario, label, color, marker, linestyle)
    ("ordinary_actionable_recourse", "Ordinary recourse", PALETTE[0], "o", "--"),
    ("mediation_aware_recourse", "Ours (parity-constrained)", PALETTE[1], "s", "-"),
]
CAUSES = [  # (metric, label, color)
    ("no_recourse_rate", r"(a) no valid plan", "0.75"),
    ("target_incompatible_rate", r"(b) $\varepsilon<\hat\varepsilon_\gamma$", PALETTE[2]),
    ("unreachable_rate", r"(c) unreachable", PALETTE[4]),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/multiseed"))
    parser.add_argument("--dataset", default="acs")
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--save-dir", default=None)
    args = parser.parse_args()
    save_dir = args.save_dir or f"figures/{args.dataset}"

    with (args.root / "summary.json").open(encoding="utf-8") as f:
        rows = [row for row in json.load(f)["gamma_seed_level"]
                if row["dataset"] == args.dataset]
    gammas = sorted({row["gamma"] for row in rows})
    models = sorted({row["model"] for row in rows})

    def mean_over_seeds(model, scenario, metric):
        return np.array([
            np.mean([row[metric] for row in rows
                     if row["model"] == model and row["scenario"] == scenario
                     and row["gamma"] == gamma])
            for gamma in gammas
        ])

    fig, (ax_closure, ax_abstain) = plt.subplots(1, 2, figsize=(6.5, 2.3))

    # Left: closure.
    ax_closure.axhline(0.0, color="0.6", lw=0.8, zorder=0)
    for scenario, _, color, marker, style in METHODS:
        per_model = np.array([100 * mean_over_seeds(m, scenario,
                                                    "recipient_level_closure")
                              for m in models])
        for curve in per_model:
            ax_closure.plot(gammas, curve, color=color, linestyle=style,
                            lw=0.8, alpha=0.35)
        ax_closure.plot(gammas, per_model.mean(axis=0), color=color,
                        linestyle=style, marker=marker, lw=LW, ms=MS + 1)
    ax_closure.set_xlabel(r"Validity level $\gamma$")
    ax_closure.set_ylabel(r"Closure (\%)")
    ax_closure.set_xticks(gammas)

    # Right: abstention causes for our method.
    ours = "mediation_aware_recourse"
    stacks = [100 * np.mean([mean_over_seeds(m, ours, metric) for m in models],
                            axis=0)
              for metric, _, _ in CAUSES]
    ax_abstain.stackplot(gammas, *stacks, colors=[c for _, _, c in CAUSES],
                         edgecolor="white", linewidth=1.0)
    eps_gamma = np.mean([mean_over_seeds(m, ours,
                                         "compatibility_threshold_median")
                         for m in models], axis=0)
    if eps_gamma.min() < args.epsilon < eps_gamma.max():
        crossing = float(np.interp(args.epsilon, eps_gamma, gammas))
        ax_abstain.axvline(crossing, color="0.2", lw=0.9, linestyle=":")
        ax_abstain.text(crossing - 0.01, 4,
                        r"median $\hat\varepsilon_\gamma=\varepsilon$",
                        ha="right", va="bottom", fontsize=7, color="0.2")
    ax_abstain.set_xlim(gammas[0], gammas[-1])
    ax_abstain.set_ylim(0, 100)
    ax_abstain.set_xticks(gammas)
    ax_abstain.set_xlabel(r"Validity level $\gamma$")
    ax_abstain.set_ylabel(r"Our abstentions (\%)")

    for ax in (ax_closure, ax_abstain):
        ax.grid(axis="y", color="0.9", lw=0.6)
        ax.set_axisbelow(True)

    method_handles = [Line2D([], [], color=c, marker=m, linestyle=s, lw=LW,
                             ms=MS + 1) for _, _, c, m, s in METHODS]
    cause_handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in CAUSES]
    fig.legend(handles=method_handles + cause_handles,
               labels=[m[1] for m in METHODS] + [c[1] for c in CAUSES],
               loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = save_figure(fig, "gamma_sensitivity", save_dir=save_dir, ext=ext)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
