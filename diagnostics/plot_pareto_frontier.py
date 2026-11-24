"""
plot_pareto_frontier.py — Closure and cost against abstention across epsilon.

For each dataset, epsilon sweeps [0.025, 0.5] at the published gamma
(classifier- and seed-averaged). Every epsilon is plotted, except that
epsilon values sharing the same abstention rate are collapsed to the one with
the highest aggregate closure (looser epsilon at the same abstention serves
the same recipients with less closure and more cost). Panels:

  (a) closure among served recipients,
  (b) aggregate closure over all recipients (abstainers keep their factual gap),
  (c) mean action cost c(a*) among served recipients (log scale).

The ringed point is the main-results epsilon (0.1). Adult is not plotted by default:
it abstains on (almost) everyone for small epsilon and has strongly negative
closure otherwise; the caption reports its ranges (see --datasets).

Usage
-----
  python diagnostics/plot_pareto_frontier.py
  python diagnostics/plot_pareto_frontier.py --datasets german_synth bar acs adult
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.join(os.getcwd(), "diagnostics"))
from paper_style import PALETTE, LW, MS, set_paper_style, save_figure  # noqa: E402

set_paper_style()

# Fixed order; red (PALETTE[3]) is skipped so no pair is orange/red.
DATASETS = [("german_synth", "German-Synth", PALETTE[0], "o"),
            ("bar", "LSAC Bar Passage", PALETTE[1], "s"),
            ("acs", "ACS Income", PALETTE[2], "D"),
            ("adult", "Adult", PALETTE[4], "^")]


def dedupe_abstention(points, ndigits=6):
    """Collapse epsilon values with the same abstention rate to the one with
    the highest aggregate closure; return points ordered by epsilon."""
    best = {}
    for p in points:
        a = round(p["abstention"], ndigits)
        if a not in best or p["closure_all"] > best[a]["closure_all"]:
            best[a] = p
    return sorted(best.values(), key=lambda p: p["epsilon"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/pareto"))
    parser.add_argument("--model", default="Average")
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--published-eps", type=float, default=0.1)
    parser.add_argument("--eps-max", type=float, default=0.5)
    parser.add_argument("--datasets", nargs="+",
                        default=["german_synth", "bar", "acs"])
    parser.add_argument("--save-dir", default="figures/main")
    parser.add_argument("--name", default="pareto_closure_abstention")
    args = parser.parse_args()

    summary = json.loads((args.root / "summary.json").read_text())
    fig, (ax_s, ax_a, ax_k) = plt.subplots(1, 3, figsize=(6.9, 2.4))
    present = []

    for key, label, color, marker in DATASETS:
        if key not in summary or key not in args.datasets:
            continue
        present.append((label, color, marker))
        grid = [p for p in summary[key][args.model]
                if p["gamma"] == args.gamma and p["epsilon"] is not None
                and p["epsilon"] <= args.eps_max
                and p["closure_all"] is not None]
        front = dedupe_abstention(grid)
        eps = np.array([p["epsilon"] for p in front])
        abst = np.array([100 * p["abstention"] for p in front])
        series = {
            ax_s: np.array([np.nan if p["closure_served"] is None
                            else 100 * p["closure_served"] for p in front]),
            ax_a: np.array([100 * p["closure_all"] for p in front]),
            ax_k: np.array([np.nan if p["cost_served"] is None
                            else p["cost_served"] for p in front]),
        }
        pub = np.isclose(eps, args.published_eps)
        for ax, y in series.items():
            ok = ~np.isnan(y)
            if not ok.any():
                continue
            ax.plot(abst[ok], y[ok], color=color, lw=LW, marker=marker,
                    ms=MS, zorder=3)
            if (pub & ok).any():
                ax.plot(abst[pub & ok], y[pub & ok], ls="", marker="o",
                        ms=MS + 4.5, mfc="none", mec="0.15", mew=0.9,
                        zorder=5)

    for ax, title in ((ax_s, r"(a) Closure $\mid$ served"),
                      (ax_a, r"(b) Aggregate closure")):
        ax.set_ylim(-3, 103)
        ax.axhline(0, color="0.6", lw=0.6, zorder=0)
        ax.set_ylabel(r"Closure (\%)")
        ax.set_title(title, loc="left")
    ax_k.set_yscale("log")
    ax_k.set_yticks([0.5, 1, 2, 5, 10])
    ax_k.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax_k.yaxis.set_minor_formatter(plt.NullFormatter())
    ax_k.set_ylim(0.4, 12)
    ax_k.set_ylabel(r"Mean cost $c(a^\star)$, served")
    ax_k.set_title(r"(c) Action cost", loc="left")
    for ax in (ax_s, ax_a, ax_k):
        ax.set_xlim(-3, 103)
        ax.set_xlabel(r"Abstention rate (\%)")
        ax.grid(axis="y", color="0.9", lw=0.6)
        ax.set_axisbelow(True)

    handles = [Line2D([], [], color=c, lw=LW, marker=m, ms=MS)
               for _, c, m in present]
    labels = [lab for lab, _, _ in present]
    handles.append(Line2D([], [], ls="", marker="o", ms=MS + 4.5, mfc="none",
                          mec="0.15"))
    labels.append(rf"Main results ($\varepsilon = {args.published_eps:g}$)")
    fig.legend(handles=handles, labels=labels, loc="lower center",
               bbox_to_anchor=(0.5, 1.0), ncol=len(labels))
    fig.tight_layout()
    for ext in ("pdf", "png"):
        print("Saved", save_figure(fig, args.name, save_dir=args.save_dir,
                                   ext=ext))


if __name__ == "__main__":
    main()
