"""
make_baseline_table.py — Additional recourse baselines for the appendix.

Every baseline selects from the same candidate plans, intervention draws, and
reference draws as ordinary recourse and our method; only the selection rule
differs. Rows:

  Ordinary            cheapest gamma-valid plan (gamma = 0.5)
  Robust (gamma=0.9)  ordinary recourse at gamma = 0.9 (probabilistic robustness)
  Tuned gamma*        ordinary recourse at the gamma in {.5,...,.9} with the best
                      mean closure, chosen post hoc on the evaluation recipients
  Soft anchor         cost + eta*shortfall + rho*mediator transport
                      (eta = rho = 10); validity is penalized, not imposed
  Mean parity         gamma-valid and |E[P_a] - E[R_z]| <= 0.1
  Mediator matching   gamma-valid plan closest to the reference in mediator space
  Plausibility        cheapest gamma-valid plan at least as likely under the
                      fitted mediator model as the recipient's factual mediators
  Ours                gamma-valid and W1(P_a, R_z) <= 0.1

New-rule rows are read from outputs/baselines/ (recourse stage rerun with the
seed's existing models; ordinary and ours reproduce outputs/multiseed exactly).
The other rows are read from outputs/multiseed/.

Writes drafts/appendix_baselines_table.tex and outputs/baselines/summary.md.

Usage
-----
  python diagnostics/make_baseline_table.py
"""

import argparse
import json
from pathlib import Path

import numpy as np

DATASETS = [("german_synth", "German-Synth"), ("bar", "Law School"),
            ("acs", "ACS Income"), ("adult", "Adult")]
MODELS = [("logistic_reg", "LR"), ("mlp_6432", "MLP"),
          ("random_forest", "RF")]
SEEDS = [42, 43, 44, 45, 46]
GAMMAS = [0.5, 0.6, 0.7, 0.8, 0.9]
METHODS = [
    ("ordinary", "Ordinary"),
    ("robust", r"Robust ($\gamma=0.9$)"),
    ("tuned", r"Tuned $\gamma^\star$"),
    ("soft_anchor", "Soft anchor"),
    ("mean_parity_recourse", "Mean parity"),
    ("mediator_matching_recourse", "Mediator matching"),
    ("plausibility_constrained_recourse", "Plausibility"),
    ("ours", "Ours"),
]
NEW_RULES = {"mean_parity_recourse", "mediator_matching_recourse",
             "plausibility_constrained_recourse"}


def _load(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _metrics(row, soft=False):
    served = row["feasible_rate"][0] if soft else row["constraint_feasible_rate"][0]
    return {
        "closure": row["recipient_level_closure"][0],
        "overshoot": row["overshoot_rate"][0],
        "served": served,
        "cost": row["cost"][0],
        "gain": (None if soft else
                 row["paired_absolute_gap_gain_vs_actionable_baseline"][0]),
    }


def _gamma_row(ablation, gamma, method):
    entry = next(e for e in ablation["gamma_sensitivity"]
                 if np.isclose(e["min_success_probability"], gamma))
    return next(r for r in entry["rows"] if r["method"] == method)


def collect(root):
    """Return {(dataset, model, method): [metrics per seed]}."""
    out = {}
    for ds, _ in DATASETS:
        for slug, _ in MODELS:
            per_gamma = {g: [] for g in GAMMAS}
            for seed in SEEDS:
                name = f"ablation_{ds}_{slug}.json"
                main = _load(root / "multiseed" / ds / f"seed_{seed}"
                             / "recourse" / name)
                new = _load(root / "baselines" / ds / f"seed_{seed}"
                            / "recourse" / name)
                rows = {r["method"]: r for r in new["rows"]}
                old = {r["method"]: r for r in main["rows"]}
                for method in ("ordinary_actionable_recourse",
                               "distribution_constrained_recourse"):
                    a = old[method]["recipient_level_closure"][0]
                    b = rows[method]["recipient_level_closure"][0]
                    if not np.isclose(a, b):
                        raise RuntimeError(
                            f"{ds} {slug} seed {seed}: {method} did not "
                            f"reproduce ({a} vs {b})")
                add = lambda key, m: out.setdefault((ds, slug, key), []).append(m)
                add("ordinary", _metrics(rows["ordinary_actionable_recourse"]))
                add("ours", _metrics(rows["distribution_constrained_recourse"]))
                for method in NEW_RULES:
                    add(method, _metrics(rows[method]))
                robust = _metrics(_gamma_row(
                    main, 0.9, "ordinary_actionable_recourse"))
                robust["gain"] = None  # paired against gamma=0.9 ordinary
                add("robust", robust)
                anchor = next(
                    r for r in main["rows"]
                    if r["method"] == "transport_anchor_only"
                    and np.isclose(r["eta"], 10.0)
                    and np.isclose(r["rho_anchor"], 10.0))
                add("soft_anchor", _metrics(anchor, soft=True))
                for g in GAMMAS:
                    m = _metrics(_gamma_row(
                        main, g, "ordinary_actionable_recourse"))
                    m["gain"] = None
                    per_gamma[g].append(m)
            best = max(GAMMAS, key=lambda g: np.mean(
                [m["closure"] for m in per_gamma[g]]))
            for m in per_gamma[best]:
                out.setdefault((ds, slug, "tuned"), []).append(
                    dict(m, gamma=best))
    return out


def _avg_over_models(data, ds, method, key):
    """Per-seed value averaged over classifiers; returns array over seeds."""
    values = []
    for slug, _ in MODELS:
        seq = data[(ds, slug, method)]
        values.append([m[key] if m[key] is not None else np.nan for m in seq])
    return np.nanmean(np.asarray(values, dtype=float), axis=0)


def _cell(values, scale=100.0, digits=1, lead=True):
    a = np.asarray(values, dtype=float) * scale
    if np.all(np.isnan(a)):
        return "--"
    s = f"{np.nanmean(a):.{digits}f}"
    if not lead:
        s = s.replace("0.", ".", 1)
    return s.replace("-", "$-$")


def latex(data):
    lines = [r"""% Generated by diagnostics/make_baseline_table.py; do not edit by hand.
\begin{table*}[t]
\centering
\footnotesize
\caption{Additional baselines at $\gamma=0.5$ ($\varepsilon=0.1$ where it
applies). All methods select from the same candidate plans, intervention
draws, and reference draws, and differ only in the selection rule.
\emph{Robust}: ordinary recourse at $\gamma=0.9$, i.e.\ probabilistically
robust recourse. \emph{Tuned $\gamma^\star$}: ordinary recourse at the
$\gamma\in\{0.5,\ldots,0.9\}$ with the highest closure, chosen post hoc on the
evaluation recipients. \emph{Soft anchor}: minimizes cost plus penalties on
the threshold shortfall and on the mediator-space distance to the advantaged
reference ($\eta=\rho=10$); validity is penalized rather than imposed.
\emph{Mean parity}: cheapest $\gamma$-valid plan with
$|\mathbb E[P_a]-\hat\mu_{\mathrm{ref}}(z)|\le\varepsilon$, the first-moment
relaxation of our constraint. On ACS it serves more recipients and closes more
of the disparity than our method, but overshoots the reference for 43--49\% of
recipients against 16--17\% for ours. \emph{Mediator matching}: the $\gamma$-valid plan
whose post-intervention mediators are closest to the advantaged reference
mediators (sum of per-mediator normalized $W_1$), a counterfactual-twin
target. \emph{Plausibility}: cheapest $\gamma$-valid plan whose mediators are
at least as likely under the fitted mediator model as the recipient's factual
mediators. Closure is recipient-level closure~\eqref{eq:closure}; Served is
the share of recipients who receive a recommendation (for the soft anchor, the
share whose expected score clears the soft threshold); cost is averaged over
all recipients. Gain is the paired reduction in $|D_{\mathrm{post}}|$ relative
to ordinary recourse at $\gamma=0.5$. Means over three classifiers and five
seeds; per-classifier results are in \texttt{outputs/baselines/summary.md}.}
\label{tab:app_baselines}
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{l""" + "rrrr" * len(DATASETS) + r"""}
\hline"""]
    head1 = " & ".join(rf"\multicolumn{{4}}{{c}}{{{name}}}"
                       for _, name in DATASETS)
    lines.append(rf"& {head1} \\")
    rules = " ".join(rf"\cline{{{2 + 4 * k}-{5 + 4 * k}}}"
                     for k in range(len(DATASETS)))
    lines.append(rules)
    lines.append("Method & " + " & ".join(
        ["Clos.", "Serv.", "Cost", "Gain"] * len(DATASETS)) + r" \\")
    lines.append(r"\hline")
    for method, label in METHODS:
        cells = []
        for ds, _ in DATASETS:
            cells += [
                _cell(_avg_over_models(data, ds, method, "closure")),
                _cell(_avg_over_models(data, ds, method, "served"), digits=0),
                _cell(_avg_over_models(data, ds, method, "cost"), 1.0, 2),
                _cell(_avg_over_models(data, ds, method, "gain"), 1.0, 3,
                      lead=False),
            ]
        if method == "ours":
            lines.append(r"\hline")
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines.append(r"\hline" "\n" r"\end{tabular}" "\n" r"\end{table*}")
    return "\n".join(lines) + "\n"


def markdown(data):
    out = ["# Additional baselines (gamma = 0.5)", "",
           "Mean ± SD over seeds 42-46. Closure, overshoot, and served in %. "
           "Gain is the paired |D_post| reduction vs ordinary recourse.", ""]
    for ds, name in DATASETS:
        for slug, short in MODELS:
            out += [f"## {name} — {short}", "",
                    "| Method | Closure | Overshoot | Served | Cost | Gain |",
                    "|---|---:|---:|---:|---:|---:|"]
            for method, label in METHODS:
                seq = data[(ds, slug, method)]
                def ms(key, scale=100.0, d=1):
                    v = [m[key] for m in seq if m[key] is not None]
                    if not v:
                        return "--"
                    a = np.asarray(v) * scale
                    return f"{a.mean():.{d}f} ± {a.std(ddof=1):.{d}f}"
                label = label.replace("$", "").replace(r"\gamma", "gamma") \
                    .replace(r"^\star", "*")
                if method == "tuned":
                    label += f" (gamma={seq[0]['gamma']})"
                out.append(
                    f"| {label} | {ms('closure')} | {ms('overshoot')} | "
                    f"{ms('served', d=0)} | {ms('cost', 1.0, 2)} | "
                    f"{ms('gain', 1.0, 3)} |")
            out.append("")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs"))
    parser.add_argument("--tex", type=Path,
                        default=Path("drafts/appendix_baselines_table.tex"))
    args = parser.parse_args()
    data = collect(args.root)
    args.tex.write_text(latex(data), encoding="utf-8")
    (args.root / "baselines" / "summary.md").write_text(
        markdown(data), encoding="utf-8")
    print(f"Wrote {args.tex} and {args.root / 'baselines' / 'summary.md'}")


if __name__ == "__main__":
    main()
