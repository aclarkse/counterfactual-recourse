"""
make_appendix_tables.py — LaTeX tables for Appendix "Additional Results".

Reads the multi-seed outputs (summary.json plus each seed's gap and recourse
artifacts) and writes drafts/appendix_results_tables.tex, which defines:

  tab:app_effects    classifier quality and classifier-scale effects
  tab:app_recourse   full recourse metrics for both methods at gamma = 0.5
  tab:app_compat     compatibility diagnostics and abstention causes
  tab:app_coverage   joint coverage as a function of epsilon
  tab:app_stress     German-Synth mediation stress test, per classifier
  tab:app_seed_a/b   per-seed results with within-seed bootstrap intervals

Usage
-----
  python diagnostics/make_appendix_tables.py
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

DATASETS = [("german_synth", "German-Synth"), ("bar", "Law School"),
            ("acs", "ACS Income"), ("adult", "Adult")]
VERSIONS = [("german_synth_m0", "0.0"), ("german_synth_m05", "0.5"),
            ("german_synth", "1.0"), ("german_synth_m2", "2.0")]
MODELS = [("Logistic Reg.", "LR", "logistic_reg"),
          ("MLP (64--32)", "MLP", "mlp_6432"),
          ("Random Forest", "RF", "random_forest")]
ORD, OURS = "ordinary_actionable_recourse", "mediation_aware_recourse"
EPSILONS = [0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5]
GAMMA = 0.5


# ---------------------------------------------------------------------------
# Formatting

def _num(x, digits, lead=True, sign=False):
    s = f"{x:+.{digits}f}" if sign else f"{x:.{digits}f}"
    if not lead:
        s = s.replace("0.", ".", 1)
    return s.replace("-", "$-$")


def ms(values, digits=3, scale=1.0, lead=False, sign=False):
    """Mean $\\pm$ SD across seeds."""
    a = np.asarray(values, dtype=float) * scale
    sd = a.std(ddof=1) if len(a) > 1 else 0.0
    return (f"{_num(a.mean(), digits, lead, sign)} $\\pm$ "
            f"{_num(sd, digits, lead)}")


def ci(triple, digits=3, scale=1.0, lead=False):
    """Point estimate with [lo, hi] percentile-bootstrap interval."""
    p, lo, hi = (scale * v for v in triple)
    return (f"{_num(p, digits, lead)} "
            f"\\,[{_num(lo, digits, lead)},\\,{_num(hi, digits, lead)}]")


def mean(values, digits=1, scale=100.0):
    return _num(float(np.mean(values)) * scale, digits)


# ---------------------------------------------------------------------------
# Loading

class Results:
    def __init__(self, root: Path):
        self.root = root
        with (root / "summary.json").open(encoding="utf-8") as f:
            summary = json.load(f)
        self.seeds = summary["metadata"]["seeds"]
        self.seed_level = summary["seed_level"]
        self.gamma_level = summary["gamma_seed_level"]
        self.diag_level = summary["diagnostic_seed_level"]

    def rows(self, dataset, model, scenario):
        out = [r for r in self.seed_level if r["dataset"] == dataset
               and r["model"] == model and r["scenario"] == scenario]
        assert len(out) == len(self.seeds), (dataset, model, scenario)
        return sorted(out, key=lambda r: r["seed"])

    def gamma_rows(self, dataset, model, scenario, gamma=GAMMA):
        out = [r for r in self.gamma_level if r["dataset"] == dataset
               and r["model"] == model and r["scenario"] == scenario
               and np.isclose(r["gamma"], gamma)]
        assert len(out) == len(self.seeds), (dataset, model, scenario, gamma)
        return sorted(out, key=lambda r: r["seed"])

    def diag_rows(self, dataset, model):
        out = [r for r in self.diag_level
               if r["dataset"] == dataset and r["model"] == model]
        assert len(out) == len(self.seeds), (dataset, model)
        return sorted(out, key=lambda r: r["seed"])

    def col(self, dataset, model, scenario, metric):
        return [r[metric] for r in self.rows(dataset, model, scenario)]

    def ablation(self, dataset, seed, slug):
        path = (self.root / dataset / f"seed_{seed}" / "recourse"
                / f"ablation_{dataset}_{slug}.json")
        with path.open(encoding="utf-8") as f:
            return json.load(f)

    def recourse_row(self, dataset, seed, slug, scenario):
        rows = self.ablation(dataset, seed, slug)["rows"]
        if scenario == ORD:
            match = [r for r in rows if r.get("is_ordinary_actionable_baseline")]
        else:
            match = [r for r in rows
                     if r.get("method") == "distribution_constrained_recourse"]
        assert len(match) == 1
        return match[0]

    def effect_ci(self, dataset, seed, model, effect="NIE"):
        """Parse the within-seed bootstrap interval from the gap report."""
        label = "gender" if dataset != "oulad" else "disability"
        text = (self.root / dataset / f"seed_{seed}" / "gaps"
                / f"{dataset}_{label}_gap.txt").read_text(encoding="utf-8")
        block = text.split(f"{model} (n=")[1].split("\n\n")[0]
        m = re.search(rf"pure {effect}: ([+-][\d.]+) \[([+-][\d.]+), "
                      rf"([+-][\d.]+)\]", block)
        return tuple(float(v) for v in m.groups())

    def oracle(self, dataset, seed):
        path = (self.root / dataset / f"seed_{seed}" / "gaps"
                / f"{dataset}_gender_gap_oracle.json")
        with path.open(encoding="utf-8") as f:
            return json.load(f)


# ---------------------------------------------------------------------------
# Tables

def table_effects(R: Results) -> str:
    lines = [r"""\begin{table*}[t]
\centering
\small
\caption{Classifier quality and classifier-scale effects of the sensitive
attribute. AUC and accuracy are on the validation split at $\tau=0.5$. Gap is
the raw difference in mean predicted probability between groups; $\NDE$,
$\NIE$ and TE are the pure natural direct, pure natural indirect and total
effects on the fixed classifier's prediction scale, estimated with the fitted
mediator model ($n=500$ instances, $K=500$ draws per arm). Addr.\ is
$|\NIE|/|\mathrm{TE}|$, the share of the effect that mediator-based recourse
can address. Mean $\pm$ SD over five complete refits.}
\label{tab:app_effects}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llcccccc c}
\hline
Dataset & Clf. & AUC & Acc. & Gap & $\NDE$ & $\NIE$ & TE & Addr. \\
\hline"""]
    for ds, name in DATASETS:
        for i, (model, short, _) in enumerate(MODELS):
            c = lambda m: R.col(ds, model, ORD, m)
            gaps = [_raw_gap(R, ds, s, model) for s in R.seeds]
            lines.append(" & ".join([
                name if i == 0 else "", short,
                ms(c("validation_auc")), ms(c("validation_accuracy")),
                ms(gaps), ms(c("pure_nde")), ms(c("pure_nie")),
                ms(c("total_effect")), ms(c("addressability"), 2),
            ]) + r" \\")
        lines.append(r"\hline")
    lines.append(r"\end{tabular}" "\n" r"\end{table*}")
    return "\n".join(lines)


def _raw_gap(R, dataset, seed, model):
    path = (R.root / dataset / f"seed_{seed}" / "gaps"
            / f"{dataset}_gender_gap.json")
    with path.open(encoding="utf-8") as f:
        return json.load(f)[model]["raw_prediction_gap"]


def table_recourse(R: Results) -> str:
    lines = [r"""\begin{table*}[t]
\centering
\footnotesize
\caption{Full recourse metrics at $\gamma=0.5$, $\varepsilon=0.1$, extending
Table~\ref{tab:main_results}. $D_{\mathrm{post}}$ is the mean post-recourse
mediated disparity over recipients (negative means recipients end above
their reference on average). Closure is recipient-level
closure~\eqref{eq:closure}. Overshoot is the share of recipients whose disparity changes sign, and Size
the mean overshoot magnitude. Abst.\ is the share of recipients without a
recommendation; cost is averaged over all recipients, with abstentions
contributing zero. Gain is the paired reduction in $|D_{\mathrm{post}}|$
relative to ordinary recourse. Percentages; mean $\pm$ SD over five complete
refits.}
\label{tab:app_recourse}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lll ccccccc}
\hline
Dataset & Clf. & Method & $D_{\mathrm{post}}$ & Closure &
Overshoot & Size & Abst. & Cost & Gain \\
\hline"""]
    for ds, name in DATASETS:
        for i, (model, short, _) in enumerate(MODELS):
            for j, (scen, label) in enumerate([(ORD, "Ord."), (OURS, "Ours")]):
                c = lambda m: R.col(ds, model, scen, m)
                abst = [1 - v for v in c("constraint_feasible_rate")]
                gain = (ms(c("paired_absolute_gap_gain_vs_actionable_baseline"))
                        if scen == OURS else "--")
                lines.append(" & ".join([
                    name if i == 0 and j == 0 else "",
                    short if j == 0 else "", label,
                    ms(c("post_recourse_mediated_prediction_disparity"),
                       sign=True),
                    ms(c("recipient_level_closure"), 1, 100, True),
                    ms(c("overshoot_rate"), 1, 100, True),
                    ms(c("overshoot_magnitude")),
                    ms(abst, 1, 100, True),
                    ms(c("cost"), 2, lead=True),
                    gain,
                ]) + r" \\")
            if i < len(MODELS) - 1:
                lines.append(r"\cline{2-10}")
        lines.append(r"\hline")
    lines.append(r"\end{tabular}" "\n" r"\end{table*}")
    return "\n".join(lines)


def table_compat(R: Results) -> str:
    lines = [r"""\begin{table*}[t]
\centering
\footnotesize
\caption{Compatibility diagnostics of Section~\ref{sec:evaluation} at
$\gamma=0.5$, $\varepsilon=0.1$. Ref.\ valid: share of recipients whose
advantaged reference is $\gamma$-valid. $\bar p_{\mathrm{ref}}$: mean share
of reference scores at or above $\tau$. $\hat\varepsilon_\gamma$ and
$\hat\varepsilon_{\mathcal A}$: medians over recipients of the compatibility
threshold and of the smallest attainable $W_1$ among valid plans. Served:
share for which our method returns a plan. Causes~(a)--(c) partition the
abstentions: (a)~no $\gamma$-valid plan, (b)~$\varepsilon<
\hat\varepsilon_\gamma$, (c)~unreachable by the action grid; the share of
recipients with at least one $\gamma$-valid plan is $100-$(a). For
every recipient, the smallest attainable $W_1$ was at least
$\hat\varepsilon_\gamma$, as Theorem~\ref{thm:val-par} requires. Percentages;
mean $\pm$ SD over five complete refits.}
\label{tab:app_compat}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{ll cccc c ccc}
\hline
& & & & & & & \multicolumn{3}{c}{Abstention cause} \\
Dataset & Clf. & Ref.\ valid & $\bar p_{\mathrm{ref}}$ &
$\hat\varepsilon_\gamma$ & $\hat\varepsilon_{\mathcal A}$ & Served &
(a) & (b) & (c) \\
\hline"""]
    for ds, name in DATASETS:
        for i, (model, short, _) in enumerate(MODELS):
            d = R.diag_rows(ds, model)
            g = R.gamma_rows(ds, model, OURS)
            dc = lambda m: [r[m] for r in d]
            gc = lambda m: [r[m] for r in g]
            if sum(gc("bound_violations")):
                raise RuntimeError(f"Lower-bound violations for {ds} {model}; "
                                   "the caption of tab:app_compat is wrong")
            lines.append(" & ".join([
                name if i == 0 else "", short,
                ms(dc("reference_validity_rate"), 1, 100, True),
                ms(dc("reference_success_probability_mean"), 1, 100, True),
                ms(gc("compatibility_threshold_median")),
                ms(dc("minimum_attainable_w1_median")),
                ms(gc("constraint_feasible_rate"), 1, 100, True),
                ms(gc("no_recourse_rate"), 1, 100, True),
                ms(gc("target_incompatible_rate"), 1, 100, True),
                ms(gc("unreachable_rate"), 1, 100, True),
            ]) + r" \\")
        lines.append(r"\hline")
    lines.append(r"\end{tabular}" "\n" r"\end{table*}")
    return "\n".join(lines)


def table_coverage(R: Results) -> str:
    eps_head = " & ".join(f"{e:g}".lstrip("0") for e in EPSILONS)
    lines = [r"""\begin{table*}[t]
\centering
\small
\caption{Joint coverage as a function of the parity tolerance $\varepsilon$
at $\gamma=0.5$: the share of recipients (\%) with at least one plan that is
$\gamma$-valid and lies within $\varepsilon$ of the reference in $W_1$, i.e.\
the share our method would serve at that $\varepsilon$. The main results use
$\varepsilon=0.1$ (the column between vertical rules). Mean over five complete refits; the largest
SD over seeds in each row is given in the last column.}
\label{tab:app_coverage}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{ll """ + "r" * 3 + "|r|" + "r" * (len(EPSILONS) - 4) + r""" r}
\hline
& & \multicolumn{""" + str(len(EPSILONS)) + r"""}{c}{$\varepsilon$} & \\
Dataset & Clf. & """ + eps_head + r""" & Max SD \\
\hline"""]
    for ds, name in DATASETS:
        for i, (model, short, _) in enumerate(MODELS):
            d = R.diag_rows(ds, model)
            cells, sds = [], []
            for eps in EPSILONS:
                vals = [next(item["coverage"] for item in r["coverage_vs_epsilon"]
                             if np.isclose(item["epsilon"], eps)) for r in d]
                cells.append(mean(vals))
                sds.append(100 * np.std(vals, ddof=1))
            lines.append(" & ".join([name if i == 0 else "", short, *cells,
                                     f"{max(sds):.1f}"]) + r" \\")
        lines.append(r"\hline")
    lines.append(r"\end{tabular}" "\n" r"\end{table*}")
    return "\n".join(lines)


def table_stress(R: Results) -> str:
    lines = [r"""\begin{table*}[t]
\centering
\footnotesize
\caption{German-Synth mediation stress test per classifier ($\gamma=0.5$,
$\varepsilon=0.1$), expanding Table~\ref{tab:stress}. $\kappa$ scales every
sex$\to$mediator coefficient; the $\NDE$ on $P(Y=1)$ is $0.042$ in every
version. $\NIE$ is the oracle classifier-scale effect computed from the true
SCM mediator law, and Err.\ the fitted-minus-oracle $\NIE$. Columns otherwise
as in Table~\ref{tab:app_recourse}. Percentages; mean $\pm$ SD over five
complete refits.}
\label{tab:app_stress}
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{rl cc c cc cc c c}
\hline
& & & & & \multicolumn{2}{c}{Closure} & \multicolumn{2}{c}{Overshoot} & & \\
$\kappa$ & Clf. & $\NIE$ & Err. & Ref.\ valid & Ord. & Ours & Ord. & Ours &
Abst. & Gain \\
\hline"""]
    for ds, kappa in VERSIONS:
        for i, (model, short, _) in enumerate(MODELS):
            orc = [R.oracle(ds, s)["models"][model] for s in R.seeds]
            o = lambda m: R.col(ds, model, ORD, m)
            u = lambda m: R.col(ds, model, OURS, m)
            ref = [r["reference_validity_rate"] for r in R.diag_rows(ds, model)]
            lines.append(" & ".join([
                kappa if i == 0 else "", short,
                ms([x["oracle"]["nie"] for x in orc], sign=True),
                ms([x["flow_minus_oracle"]["nie"] for x in orc], 4, sign=True),
                ms(ref, 1, 100, True),
                ms(o("recipient_level_closure"), 1, 100, True),
                ms(u("recipient_level_closure"), 1, 100, True),
                ms(o("overshoot_rate"), 1, 100, True),
                ms(u("overshoot_rate"), 1, 100, True),
                ms([1 - v for v in u("constraint_feasible_rate")], 1, 100, True),
                ms(u("paired_absolute_gap_gain_vs_actionable_baseline"),
                   sign=True),
            ]) + r" \\")
        lines.append(r"\hline")
    lines.append(r"\end{tabular}" "\n" r"\end{table*}")
    return "\n".join(lines)


def table_per_seed(R: Results, datasets, label, part) -> str:
    names = " and ".join(n for _, n in datasets)
    lines = [r"""\begin{table*}[t]
\centering
\footnotesize
\caption{Per-seed results for """ + names + r""" ($\gamma=0.5$,
$\varepsilon=0.1$""" + part + r"""). Brackets are 95\% percentile-bootstrap
intervals within one seed, holding the fitted mediator model and classifier
fixed: $B=2000$ resamples of the 500 effect instances for $\NIE$ and $B=1000$
resamples of the $n$ recipients for the recourse metrics. They therefore
reflect recipient and instance sampling only; the spread across seeds
additionally reflects refitting. Closure in \%.}
\label{""" + label + r"""}
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{llrr cccc}
\hline
Dataset & Clf. & Seed & $n$ & $\NIE$ & Closure (Ord.) & Closure (Ours) &
Gain \\
\hline"""]
    for ds, name in datasets:
        for i, (model, short, slug) in enumerate(MODELS):
            for k, seed in enumerate(R.seeds):
                o = R.recourse_row(ds, seed, slug, ORD)
                u = R.recourse_row(ds, seed, slug, OURS)
                lines.append(" & ".join([
                    name if i == 0 and k == 0 else "",
                    short if k == 0 else "", str(seed), str(u["n"]),
                    ci(R.effect_ci(ds, seed, model)),
                    ci(o["recipient_level_closure"], 1, 100, True),
                    ci(u["recipient_level_closure"], 1, 100, True),
                    ci(u["paired_absolute_gap_gain_vs_actionable_baseline"]),
                ]) + r" \\")
            if i < len(MODELS) - 1:
                lines.append(r"\cline{2-8}")
        lines.append(r"\hline")
    lines.append(r"\end{tabular}" "\n" r"\end{table*}")
    return "\n".join(lines)


HEADER = r"""% Generated by diagnostics/make_appendix_tables.py from outputs/multiseed.
% Do not edit by hand; rerun the script after regenerating results.
% Uses the paper macros \NDE and \NIE.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--root", type=Path, default=Path("outputs/multiseed"))
    parser.add_argument("--out", type=Path,
                        default=Path("drafts/appendix_results_tables.tex"))
    args = parser.parse_args()

    R = Results(args.root)
    tables = [
        table_effects(R),
        table_recourse(R),
        table_compat(R),
        table_coverage(R),
        table_stress(R),
        table_per_seed(R, DATASETS[:2], "tab:app_seed_a", ""),
        table_per_seed(R, DATASETS[2:], "tab:app_seed_b",
                       "; 250 recipients per seed"),
    ]
    args.out.write_text(HEADER + "\n\n".join(tables) + "\n", encoding="utf-8")
    print(f"Wrote {len(tables)} tables to {args.out}")


if __name__ == "__main__":
    main()
