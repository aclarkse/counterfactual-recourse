"""Run and aggregate complete split/flow/classifier experiments across seeds.

Every seed writes to an isolated directory so that tensors, fitted mediator
models, classifiers, mediation estimates, and recourse sweeps cannot be mixed.

Example
-------
python -m evaluation.run_multiseed --datasets bar acs --seeds 42 43 44 45 46
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch


MODEL_SPECS = {
    "Logistic Reg.": {"outcome_slug": "logreg", "ablation_slug": "logistic_reg"},
    "MLP (64--32)": {"outcome_slug": "mlp", "ablation_slug": "mlp_6432"},
    "Random Forest": {
        "outcome_slug": "random_forest", "ablation_slug": "random_forest"
    },
}
DATASET_MODELS = {
    "acs": ("Logistic Reg.", "MLP (64--32)"),
    "bar": ("Logistic Reg.", "MLP (64--32)", "Random Forest"),
}
SCENARIOS = {
    "ordinary_actionable_recourse": lambda row: (
        row.get("is_ordinary_actionable_baseline", False)
        and np.isclose(row["eta"], 10.0)
    ),
    "transport_anchor_only": lambda row: (
        row.get("method") == "transport_anchor_only"
        and np.isclose(row["eta"], 10.0)
        and np.isclose(row["rho_anchor"], 10.0)
    ),
    "mediation_aware_recourse": lambda row: (
        row.get("method") == "mediation_aware_recourse"
        and row.get("is_effect_ratio_heuristic", False)
        and np.isclose(row["eta"], 10.0)
        and np.isclose(row["rho_anchor"], 10.0)
    ),
}
RECOURSE_METRICS = (
    "post_recourse_mediated_prediction_disparity",
    "factual_mediated_prediction_disparity_closure",
    "feasible_rate",
    "cost",
    "factual_closure_gain_vs_actionable_baseline",
    "cost_difference_vs_actionable_baseline",
    "feasibility_difference_vs_actionable_baseline",
)


def seed_paths(root: Path, dataset: str, seed: int) -> dict[str, Path]:
    base = root / dataset / f"seed_{seed}"
    return {
        "base": base,
        "tensors": base / "data" / "tensors.pt",
        "flows": base / "flows" / "flow_models.pt",
        "outcome": base / "outcome",
        "gaps": base / "gaps",
        "recourse": base / "recourse",
        "figures": base / "figures",
        "logs": base / "logs",
    }


def common_overrides(dataset: str, seed: int, paths: dict[str, Path]) -> list[str]:
    return [
        f"dataset={dataset}", f"seed={seed}",
        f"dataset.paths.tensors={paths['tensors']}",
        f"dataset.paths.flows={paths['flows']}",
        f"dataset.paths.outcome_dir={paths['outcome']}",
        f"dataset.paths.gaps_dir={paths['gaps']}",
        f"dataset.paths.recourse_dir={paths['recourse']}",
        f"dataset.paths.figures_dir={paths['figures']}",
    ]


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("\n$ " + " ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def _json_contains_models(path: Path, model_names: tuple[str, ...],
                          container: str | None = None) -> bool:
    if not path.exists():
        return False
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        if container is not None:
            data = data[container]
        return all(name in data for name in model_names)
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return False


def run_seed(dataset: str, seed: int, root: Path, resume: bool) -> None:
    paths = seed_paths(root, dataset, seed)
    overrides = common_overrides(dataset, seed, paths)
    model_names = DATASET_MODELS[dataset]
    flow_products = [paths["tensors"], paths["flows"]]
    outcome_products = [
        paths["outcome"] / f"{MODEL_SPECS[name]['outcome_slug']}.joblib"
        for name in model_names
    ]
    outcome_metrics = paths["outcome"] / "metrics.json"
    gap_product = paths["gaps"] / f"{dataset}_gender_gap.json"
    recourse_products = [
        paths["recourse"]
        / f"ablation_{dataset}_{MODEL_SPECS[name]['ablation_slug']}.json"
        for name in model_names
    ]
    stages = [
        (
            "flow",
            [sys.executable, "-m", "flows.train_flow", *overrides,
             "dataset.flow.reuse_tensors=false"],
            flow_products,
            lambda: all(path.exists() for path in flow_products),
        ),
        (
            "outcome",
            [sys.executable, "-m", "outcome.train_outcome", *overrides],
            outcome_products + [outcome_metrics],
            lambda: (
                all(path.exists() for path in outcome_products)
                and _json_contains_models(outcome_metrics, model_names, "models")
            ),
        ),
        (
            "gap",
            [sys.executable, "-m", "evaluation.estimate_gap", *overrides],
            [gap_product],
            lambda: _json_contains_models(gap_product, model_names),
        ),
        (
            "recourse",
            [sys.executable, "-m", "evaluation.ablate_recourse", *overrides],
            recourse_products,
            lambda: all(path.exists() for path in recourse_products),
        ),
    ]
    for stage, command, products, is_complete in stages:
        if resume and is_complete():
            print(f"[{dataset} seed {seed}] skipping completed {stage}", flush=True)
            continue
        print(f"[{dataset} seed {seed}] starting {stage}", flush=True)
        run_logged(command, paths["logs"] / f"{stage}.log")


def _point(value):
    return float(value[0] if isinstance(value, list) else value)


def _stats(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if not len(finite):
        return {"mean": float("nan"), "sd": float("nan"),
                "min": float("nan"), "max": float("nan"), "n_seeds": 0}
    return {
        "mean": float(finite.mean()),
        "sd": float(finite.std(ddof=1)) if len(finite) > 1 else 0.0,
        "min": float(finite.min()), "max": float(finite.max()),
        "n_seeds": int(len(finite)),
    }


def _one_row(rows: list[dict], predicate, description: str) -> dict:
    matches = [row for row in rows if predicate(row)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {description} row, found {len(matches)}")
    return matches[0]


def collect(root: Path, datasets: list[str], seeds: list[int]) -> dict:
    raw: list[dict] = []
    for dataset in datasets:
        for seed in seeds:
            paths = seed_paths(root, dataset, seed)
            flow = torch.load(paths["flows"], map_location="cpu", weights_only=False)
            with (paths["outcome"] / "metrics.json").open(encoding="utf-8") as f:
                outcome = json.load(f)
            with (paths["gaps"] / f"{dataset}_gender_gap.json").open(
                    encoding="utf-8") as f:
                gaps = json.load(f)
            for model in DATASET_MODELS[dataset]:
                slug = MODEL_SPECS[model]["ablation_slug"]
                with (paths["recourse"] / f"ablation_{dataset}_{slug}.json").open(
                        encoding="utf-8") as f:
                    ablation = json.load(f)
                base = {
                    "dataset": dataset, "seed": seed, "model": model,
                    "flow_best_val_nll": float(flow["best_val_nll"]),
                    "flow_best_epoch": int(flow["best_epoch"]),
                    "validation_auc": float(
                        outcome["models"][model]["validation"]["auc"]),
                    "validation_accuracy": float(
                        outcome["models"][model]["validation"]["accuracy"]),
                    "pure_nde": float(gaps[model]["overall"]["nde"]),
                    "pure_nie": float(gaps[model]["overall"]["nie"]),
                    "total_effect": float(gaps[model]["overall"]["te"]),
                    "addressability": float(
                        gaps[model]["overall"]["addressability"]),
                    "effect_ratio_heuristic": float(
                        gaps[model]["effect_ratio_heuristic"]),
                }
                for scenario, predicate in SCENARIOS.items():
                    selected = _one_row(ablation["rows"], predicate, scenario)
                    row = dict(base, scenario=scenario,
                               recipients=int(selected["n"]))
                    for metric in RECOURSE_METRICS:
                        row[metric] = _point(selected[metric])
                    raw.append(row)

    aggregate: list[dict] = []
    scalar_metrics = (
        "flow_best_val_nll", "flow_best_epoch", "validation_auc",
        "validation_accuracy", "pure_nde", "pure_nie", "total_effect",
        "addressability", "effect_ratio_heuristic", "recipients",
        *RECOURSE_METRICS,
    )
    for dataset in datasets:
        for model in DATASET_MODELS[dataset]:
            for scenario in SCENARIOS:
                subset = [row for row in raw if row["dataset"] == dataset
                          and row["model"] == model
                          and row["scenario"] == scenario]
                record = {"dataset": dataset, "model": model,
                          "scenario": scenario, "metrics": {}}
                for metric in scalar_metrics:
                    record["metrics"][metric] = _stats(
                        [row[metric] for row in subset]
                    )
                aggregate.append(record)
    return {
        "metadata": {
            "seeds": seeds,
            "n_seeds": len(seeds),
            "uncertainty": "sample standard deviation across complete refits",
            "within_seed_intervals": (
                "individual bootstrap with fitted models held fixed; retained "
                "in each seed's ablation JSON"
            ),
            "selection": (
                "eta=10; ordinary baseline has lambda=rho=0; full method uses "
                "lambda=|NDE/NIE| and rho=10"
            ),
        },
        "seed_level": raw,
        "aggregate": aggregate,
    }


def write_summary(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    rows = []
    for record in summary["aggregate"]:
        row = {key: record[key] for key in ("dataset", "model", "scenario")}
        for metric, stats in record["metrics"].items():
            for statistic, value in stats.items():
                row[f"{metric}_{statistic}"] = value
        rows.append(row)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Complete multi-seed split–flow–classifier results", "",
        f"Seeds: {', '.join(map(str, summary['metadata']['seeds']))}. Values are "
        "mean ± sample SD across complete refits.", "",
    ]
    lookup = {(row["dataset"], row["model"], row["scenario"]): row
              for row in summary["aggregate"]}
    for dataset in sorted({row["dataset"] for row in summary["aggregate"]}):
        lines += [f"## {dataset.upper()}", ""]
        for model in DATASET_MODELS[dataset]:
            lines += [f"### {model}", "",
                      "| Method | Post disparity | Factual closure | Feasible | Cost | Closure gain vs baseline |",
                      "|---|---:|---:|---:|---:|---:|"]
            for scenario in SCENARIOS:
                metrics = lookup[(dataset, model, scenario)]["metrics"]
                def cell(name: str, percent: bool = False) -> str:
                    factor = 100.0 if percent else 1.0
                    return (f"{factor * metrics[name]['mean']:.3f} ± "
                            f"{factor * metrics[name]['sd']:.3f}")
                lines.append(
                    f"| {scenario.replace('_', ' ')} | "
                    f"{cell('post_recourse_mediated_prediction_disparity')} | "
                    f"{cell('factual_mediated_prediction_disparity_closure', True)}% | "
                    f"{cell('feasible_rate', True)}% | {cell('cost')} | "
                    f"{cell('factual_closure_gain_vs_actionable_baseline', True)} pp |"
                )
            common = lookup[(dataset, model, "mediation_aware_recourse")]["metrics"]
            lines += ["", (
                f"Validation AUC: {common['validation_auc']['mean']:.3f} ± "
                f"{common['validation_auc']['sd']:.3f}; pure NIE: "
                f"{common['pure_nie']['mean']:+.3f} ± {common['pure_nie']['sd']:.3f}; "
                f"TE: {common['total_effect']['mean']:+.3f} ± "
                f"{common['total_effect']['sd']:.3f}."
            ), ""]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=("bar", "acs"),
                        default=["bar", "acs"])
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[42, 43, 44, 45, 46])
    parser.add_argument("--root", type=Path,
                        default=Path("outputs/multiseed"))
    parser.add_argument("--resume", action="store_true",
                        help="Skip stages whose complete output set exists.")
    parser.add_argument("--aggregate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex-matplotlib")
    if not args.aggregate_only:
        for dataset in args.datasets:
            for seed in args.seeds:
                run_seed(dataset, seed, args.root, args.resume)
    summary = collect(args.root, args.datasets, args.seeds)
    write_summary(summary, args.root)
    print(f"\nAggregate results written to {args.root / 'summary.md'}")


if __name__ == "__main__":
    main()
