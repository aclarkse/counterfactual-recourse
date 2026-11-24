"""Closure vs. abstention across the validity level gamma and tolerance epsilon.

For each dataset, seed, and classifier, the recourse stage is rerun on the
main-result artifacts (outputs/multiseed: same split, mediator model,
classifiers, recipients, candidate draws, and references). The candidate pool
is captured once and the distribution-constrained selection rule is re-applied
on a gamma x epsilon grid, so every grid point uses identical draws. epsilon =
inf is the validity-only rule (ordinary actionable recourse up to cost ties).

Per grid point we record abstention, the validity-only abstention floor at
that gamma (recipients with no gamma-valid plan, which no rule can serve),
recipient-level closure over all recipients (abstainers keep their factual
gap) and over served recipients, cost among served, and overshoot among
served. The Pareto frontier (minimize abstention, maximize all-recipient
closure) is computed per classifier on the seed-averaged grid.

Usage
-----
  python -m evaluation.pareto_frontier --stage run --datasets german_synth bar acs adult
  python -m evaluation.pareto_frontier --stage summary
  # internal: --stage recourse --dataset D --seed S
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

# PARETO_ROOT redirects outputs (e.g. when the repo volume is full).
ROOT = Path(os.environ.get("PARETO_ROOT", "outputs/pareto"))
MULTISEED = Path("outputs/multiseed")
SEEDS = [42, 43, 44, 45, 46]
MODELS = ["Logistic Reg.", "MLP (64--32)", "Random Forest"]
GAMMAS = [0.5, 0.6, 0.7, 0.8, 0.9]
EPSILONS = [0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.3,
            0.4, 0.5, float("inf")]
DISPLAY = {"german_synth": "German-Synth", "bar": "Law School",
           "acs": "ACS Income", "adult": "Adult"}


def _closure(pre, post):
    denom = np.abs(pre).sum()
    return float(1.0 - np.abs(post).sum() / denom) if len(pre) and denom > 0 \
        else None


def evaluate_grid(candidates, references, factual_predictions,
                  disadvantaged_value):
    """Apply the constrained selection rule on the gamma x epsilon grid."""
    from evaluation.recourse_metrics import (
        attach_mediated_disparities, select_distribution_constrained)
    out = []
    for gamma in GAMMAS:
        floor = np.mean([
            not any(c.get("success_probability_xi", 0.0) >= gamma for c in cs)
            for cs in candidates])
        for eps in EPSILONS:
            recs = []
            for i, cs in enumerate(candidates):
                r = dict(select_distribution_constrained(cs, gamma, eps))
                attach_mediated_disparities(
                    r, references["reference"][i],
                    references["natural_disadvantaged"][i],
                    factual_predictions[i], disadvantaged_value)
                recs.append(r)
            served = np.array([r["constraint_feasible"] for r in recs])
            pre = np.array([r["pre_recourse_factual_mediated_prediction_"
                              "disparity"] for r in recs])
            post = np.array([r["post_recourse_mediated_prediction_disparity"]
                             for r in recs])
            cost = np.array([r["cost"] for r in recs])
            out.append({
                "gamma": gamma, "epsilon": eps, "n": len(recs),
                "served": float(served.mean()),
                "abstention": float(1.0 - served.mean()),
                "no_valid_plan_floor": float(floor),
                "closure_all": _closure(pre, post),
                "closure_served": _closure(pre[served], post[served]),
                "cost_served": (float(cost[served].mean())
                                if served.any() else None),
                "overshoot_served": (
                    float(((pre[served] * post[served]) < 0).mean())
                    if served.any() else None),
            })
    return out


def recourse_one(dataset, seed):
    from hydra import compose, initialize_config_dir

    import evaluation.ablate_recourse as ar

    src = MULTISEED / dataset / f"seed_{seed}"
    base = ROOT / dataset / f"seed_{seed}"
    store, pending = {"dataset": dataset, "seed": seed, "models": {}}, {}

    orig_run_grid = ar.run_grid

    def run_grid(*args, **kwargs):
        if args[3]:  # main grid call, not a gamma-sensitivity rerun
            pending["grid"] = evaluate_grid(args[0], args[1], args[2],
                                            int(args[6]))
        return orig_run_grid(*args, **kwargs)
    ar.run_grid = run_grid

    orig_dump = ar.json.dump

    def dump(obj, f, *a, **k):
        if isinstance(obj, dict) and "model" in obj and "rows" in obj:
            check = {r["method"]: r for r in obj["rows"]}
            main = check.get("distribution_constrained_recourse")
            store["models"][obj["model"]] = {
                "grid": pending.pop("grid"),
                "published_point": {
                    "gamma": obj["distribution_constrained_recourse"][
                        "min_success_probability"],
                    "epsilon": obj["distribution_constrained_recourse"][
                        "max_outcome_wasserstein"],
                    "served": main["constraint_feasible_rate"][0],
                    "closure_all": main["recipient_level_closure"][0]},
            }
        return orig_dump(obj, f, *a, **k)
    ar.json.dump = dump

    with initialize_config_dir(config_dir=str(Path("conf").resolve()),
                               version_base="1.1"):
        cfg = compose(config_name="config", overrides=[
            f"dataset={dataset}", f"seed={seed}",
            f"dataset.paths.tensors={src / 'data' / 'tensors.pt'}",
            f"dataset.paths.flows={src / 'flows' / 'flow_models.pt'}",
            f"dataset.paths.outcome_dir={src / 'outcome'}",
            f"dataset.paths.gaps_dir={src / 'gaps'}",
            f"dataset.paths.recourse_dir={base / 'recourse'}",
            f"dataset.paths.figures_dir={base / 'figures'}",
        ])
    ar.run(cfg)
    ar.json.dump = orig_dump
    (base / "grid.json").write_text(json.dumps(store, indent=1))


def run(datasets):
    for dataset in datasets:
        for seed in SEEDS:
            base = ROOT / dataset / f"seed_{seed}"
            if (base / "grid.json").exists():
                continue
            (base / "logs").mkdir(parents=True, exist_ok=True)
            with (base / "logs" / "recourse.log").open("w") as log:
                subprocess.run(
                    [sys.executable, os.path.abspath(__file__),
                     "--stage", "recourse", "--dataset", dataset,
                     "--seed", str(seed)],
                    stdout=log, stderr=subprocess.STDOUT, check=True)
            print(f"done {dataset} seed {seed}", flush=True)


def pareto(points):
    """Indices of points not dominated in (abstention down, closure up)."""
    pts = [p for p in points if p["closure_all"] is not None]
    keep = []
    for i, p in enumerate(points):
        if p["closure_all"] is None:
            continue
        dominated = any(
            q["abstention"] <= p["abstention"]
            and q["closure_all"] >= p["closure_all"]
            and (q["abstention"] < p["abstention"]
                 or q["closure_all"] > p["closure_all"])
            for q in pts)
        if not dominated:
            keep.append(i)
    return sorted(keep, key=lambda i: points[i]["abstention"])


def _mean_sd(values):
    v = np.array([x for x in values if x is not None], float)
    if not len(v):
        return None, None
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def summarize(datasets):
    keys = ["served", "abstention", "no_valid_plan_floor", "closure_all",
            "closure_served", "cost_served", "overshoot_served"]
    result, md = {}, ["# Closure vs. abstention across gamma and epsilon", "",
                      "Mean over seeds 42-46. Closure is recipient-level L1 "
                      "closure; abstainers keep their factual gap in 'all'. "
                      "Floor = recipients with no gamma-valid plan. "
                      "epsilon = inf is the validity-only rule. Rows marked "
                      "* are on the Pareto frontier (min abstention, max "
                      "all-recipient closure). (0.5, 0.1) is the published "
                      "operating point.", ""]
    for dataset in datasets:
        files = sorted((ROOT / dataset).glob("seed_*/grid.json"))
        if not files:
            continue
        seeds = [json.loads(f.read_text()) for f in files]
        for s in seeds:
            for m, v in s["models"].items():
                pt = next(g for g in v["grid"]
                          if g["gamma"] == v["published_point"]["gamma"]
                          and g["epsilon"] == v["published_point"]["epsilon"])
                assert np.isclose(pt["served"], v["published_point"]["served"])
                assert np.isclose(pt["closure_all"],
                                  v["published_point"]["closure_all"])
        result[dataset] = {}
        md += [f"## {DISPLAY.get(dataset, dataset)} ({len(seeds)} seeds)", ""]
        for model in MODELS + ["Average"]:
            grid = []
            for gi, base_pt in enumerate(seeds[0]["models"][MODELS[0]]["grid"]):
                pt = {"gamma": base_pt["gamma"], "epsilon": base_pt["epsilon"]}
                for k in keys:
                    if model == "Average":
                        per_seed = [
                            float(np.mean([s["models"][m]["grid"][gi][k]
                                           for m in MODELS]))
                            if all(s["models"][m]["grid"][gi][k] is not None
                                   for m in MODELS) else None
                            for s in seeds]
                    else:
                        per_seed = [s["models"][model]["grid"][gi][k]
                                    for s in seeds]
                    pt[k], pt[k + "_sd"] = _mean_sd(per_seed)
                grid.append(pt)
            front = pareto(grid)
            for i, pt in enumerate(grid):
                pt["pareto"] = i in front
            result[dataset][model] = grid
            md += [f"### {model}", "",
                   "| gamma | epsilon | Abstain % | Floor % | Closure all % | "
                   "Closure served % | Cost served | Overshoot served % |",
                   "|---:|---:|---:|---:|---:|---:|---:|---:|"]

            def f(x, scale=100, d=1):
                return "--" if x is None else f"{scale * x:.{d}f}"
            for pt in grid:
                eps = "inf" if np.isinf(pt["epsilon"]) else f"{pt['epsilon']:g}"
                star = "*" if pt["pareto"] else ""
                md.append(
                    f"| {pt['gamma']:g} | {eps}{star} | {f(pt['abstention'])} | "
                    f"{f(pt['no_valid_plan_floor'])} | "
                    f"{f(pt['closure_all'])} ± {f(pt['closure_all_sd'])} | "
                    f"{f(pt['closure_served'])} | "
                    f"{f(pt['cost_served'], 1, 2)} | "
                    f"{f(pt['overshoot_served'])} |")
            md.append("")
    (ROOT / "summary.json").write_text(json.dumps(
        result, indent=1, default=lambda x: None))
    (ROOT / "summary.md").write_text("\n".join(md))
    print(f"wrote {ROOT / 'summary.md'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["run", "recourse", "summary"],
                        required=True)
    parser.add_argument("--datasets", nargs="+",
                        default=["german_synth", "bar", "acs", "adult"])
    parser.add_argument("--dataset")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.stage == "recourse":
        recourse_one(args.dataset, args.seed)
    elif args.stage == "run":
        run(args.datasets)
    else:
        summarize(args.datasets)


if __name__ == "__main__":
    main()
