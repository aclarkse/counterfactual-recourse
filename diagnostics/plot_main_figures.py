"""
plot_main_figures.py — The two main-text figures at two-column width.

(a) closure_vs_true_nie.pdf: German-Synth mediation stress test, closure
    against the true NIE, one panel per classifier (3.9 x 1.65 in).
(b) gamma_sensitivity.pdf: ACS validity-level sweep, closure and the causes
    of our abstentions against gamma (2.75 x 1.65 in).

Side by side they fill a 6.65 in text width. The method legend appears once,
above (a); (b) carries only the abstention-cause legend.

Usage
-----
  python diagnostics/plot_main_figures.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

mpl.rcParams.update({
    "text.usetex": True, "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "lines.markersize": 3.5, "lines.linewidth": 1.1,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "legend.frameon": False, "legend.handlelength": 1.6,
    "legend.columnspacing": 1.0, "legend.handletextpad": 0.4,
    "savefig.dpi": 300,
})

PALETTE = sns.color_palette("colorblind")
METHODS = [  # (scenario, label, color, marker, linestyle)
    ("ordinary_actionable_recourse", "Ordinary recourse", PALETTE[0], "o", "--"),
    ("mediation_aware_recourse", "Ours (parity-constrained)", PALETTE[1], "s", "-"),
]
MODELS = [("Logistic Reg.", "Logistic regression"),
          ("MLP (64--32)", "MLP"),
          ("Random Forest", "Random forest")]
VERSIONS = ["german_synth_m0", "german_synth_m05", "german_synth",
            "german_synth_m2"]
CAUSES = [  # (metric, label, color); letters avoided to not clash with (a)/(b)
    ("no_recourse_rate", "No valid plan", "0.75"),
    ("target_incompatible_rate", "Incompat.", PALETTE[2]),
    ("unreachable_rate", "Unreach.", PALETTE[4]),
]


def grid(ax):
    ax.grid(axis="y", color="0.9", lw=0.5)
    ax.set_axisbelow(True)


def save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.01)
        print(f"Saved {path}")


def figure_stress(root: Path, seed_rows: list, out_dir: Path) -> None:
    def oracle_nie(dataset, model):
        values = []
        for path in sorted(root.glob(
                f"{dataset}/seed_*/gaps/{dataset}_gender_gap_oracle.json")):
            with path.open(encoding="utf-8") as f:
                values.append(json.load(f)["models"][model]["oracle"]["nie"])
        return values

    fig, axes = plt.subplots(1, 3, figsize=(3.9, 1.65), sharey=True)
    for ax, (model, title) in zip(axes, MODELS):
        for scenario, _, color, marker, style in METHODS:
            xs, xerr, ys, yerr = [], [], [], []
            for dataset in VERSIONS:
                nie = oracle_nie(dataset, model)
                closure = [100 * row["recipient_level_closure"]
                           for row in seed_rows
                           if row["dataset"] == dataset
                           and row["model"] == model
                           and row["scenario"] == scenario]
                xs.append(np.mean(nie)); xerr.append(np.std(nie, ddof=1))
                ys.append(np.mean(closure)); yerr.append(np.std(closure, ddof=1))
            ax.errorbar(xs, ys, xerr=xerr, yerr=yerr, color=color,
                        marker=marker, linestyle=style, capsize=1.2,
                        elinewidth=0.6, capthick=0.6)
        ax.set_title(title, pad=2)
        ax.set_xlim(-0.03, 0.34)
        ax.set_xticks([0.0, 0.1, 0.2, 0.3])
        ax.set_xticklabels(["0", ".1", ".2", ".3"])
        grid(ax)
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    axes[0].set_ylim(50, 100)
    axes[0].set_ylabel(r"Closure (\%)")
    axes[1].set_xlabel(r"True $\mathrm{NIE}$")

    handles = [Line2D([], [], color=c, marker=m, linestyle=s)
               for _, _, c, m, s in METHODS]
    fig.legend(handles=handles, labels=[m[1] for m in METHODS],
               loc="lower center", bbox_to_anchor=(0.55, 0.98), ncol=2)
    fig.tight_layout(pad=0.2, w_pad=0.6)
    save(fig, out_dir, "closure_vs_true_nie")


def figure_gamma(gamma_rows: list, dataset: str, epsilon: float,
                 out_dir: Path) -> None:
    rows = [row for row in gamma_rows if row["dataset"] == dataset]
    gammas = sorted({row["gamma"] for row in rows})
    models = sorted({row["model"] for row in rows})

    def mean_over_seeds(model, scenario, metric):
        return np.array([
            np.mean([row[metric] for row in rows
                     if row["model"] == model and row["scenario"] == scenario
                     and row["gamma"] == gamma])
            for gamma in gammas
        ])

    fig, (ax_c, ax_a) = plt.subplots(1, 2, figsize=(2.75, 1.65))

    ax_c.axhline(0.0, color="0.6", lw=0.6, zorder=0)
    for scenario, _, color, marker, style in METHODS:
        per_model = np.array([
            100 * mean_over_seeds(m, scenario, "recipient_level_closure")
            for m in models])
        for curve in per_model:
            ax_c.plot(gammas, curve, color=color, linestyle=style, lw=0.6,
                      alpha=0.35)
        ax_c.plot(gammas, per_model.mean(axis=0), color=color,
                  linestyle=style, marker=marker)
    # Titles rather than y-labels so (b) aligns with the titled panels of (a).
    ax_c.set_title(r"Closure (\%)", pad=2)
    grid(ax_c)

    ours = "mediation_aware_recourse"
    stacks = [100 * np.mean([mean_over_seeds(m, ours, metric) for m in models],
                            axis=0) for metric, _, _ in CAUSES]
    ax_a.stackplot(gammas, *stacks, colors=[c for _, _, c in CAUSES],
                   edgecolor="white", linewidth=0.6)
    eps_gamma = np.mean([mean_over_seeds(m, ours,
                                         "compatibility_threshold_median")
                         for m in models], axis=0)
    if eps_gamma.min() < epsilon < eps_gamma.max():
        ax_a.axvline(float(np.interp(epsilon, eps_gamma, gammas)),
                     color="0.15", lw=0.7, linestyle=":")
    ax_a.set_xlim(gammas[0], gammas[-1])
    ax_a.set_ylim(0, 100)
    ax_a.set_title(r"Our abstentions (\%)", pad=2)
    # One legend row, level with the method legend above (a).
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c)
                        for _, _, c in CAUSES],
               labels=[label for _, label, _ in CAUSES],
               loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=3,
               handlelength=0.9, columnspacing=0.6)

    for ax in (ax_c, ax_a):
        ax.set_xticks([0.5, 0.7, 0.9])
        ax.set_xlabel(r"$\gamma$", labelpad=1)
    fig.tight_layout(pad=0.2, w_pad=0.5)
    save(fig, out_dir, "gamma_sensitivity")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/multiseed"))
    parser.add_argument("--gamma-dataset", default="acs")
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--out-dir", type=Path, default=Path("figures/main"))
    args = parser.parse_args()
    with (args.root / "summary.json").open(encoding="utf-8") as f:
        summary = json.load(f)
    figure_stress(args.root, summary["seed_level"], args.out_dir)
    figure_gamma(summary["gamma_seed_level"], args.gamma_dataset,
                 args.epsilon, args.out_dir)


if __name__ == "__main__":
    main()
