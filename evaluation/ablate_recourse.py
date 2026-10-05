"""Independent eta/lambda recourse ablations with direct plug-in evaluation.

eta weights classifier shortfall, lambda weights pointwise direct-effect
invariance, and rho weights a conservative transport upper bound to the
conditional advantaged-group mediator distribution. All three are varied
independently.
"""

import csv
import json
import os
import warnings

warnings.filterwarnings("ignore")

import hydra
import joblib
import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.stats import spearmanr

from evaluation.compute_recourse import (
    compute_weights,
    precompute_interventional_candidates_batch,
)
from evaluation.candidate_pool import enumerate_pool, score_pool, sanity_check_proba
from evaluation.recourse_metrics import (
    add_candidate_geometry,
    attach_mediated_disparities,
    estimate_reference_terms,
    select_best,
    select_distribution_constrained,
    select_recourse_indices,
    select_validity_constrained,
)
from flows.diagnostics import make_sample_fns
from flows.models import load_flow_models
from flows.schema import MediatorSchema
from outcome.models import build_torch_pipeline


def _ci(values, n_boot, rng):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return [float("nan")] * 3
    indices = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[indices].mean(1)
    return [float(values.mean()), float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5))]


def _finite_summary(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {
            "n": 0, "mean": None, "sd": None, "min": None,
            "q25": None, "median": None, "q75": None, "q90": None,
            "max": None,
        }
    return {
        "n": int(len(finite)),
        "mean": float(finite.mean()),
        "sd": float(finite.std(ddof=1)) if len(finite) > 1 else 0.0,
        "min": float(finite.min()),
        "q25": float(np.percentile(finite, 25)),
        "median": float(np.median(finite)),
        "q75": float(np.percentile(finite, 75)),
        "q90": float(np.percentile(finite, 90)),
        "max": float(finite.max()),
    }


def compatibility_threshold(prediction_reference, threshold,
                            min_success_probability):
    """Per-recipient epsilon_gamma from each empirical reference law.

    epsilon_gamma = int_{1-gamma}^1 (tau - F_R^{-1}(u))_+ du, the smallest W1
    from R_z attainable by any gamma-valid score distribution. The empirical
    quantile function is a step function, so the integral is exact.
    """
    reference = np.sort(np.asarray(prediction_reference, dtype=float), axis=1)
    n_draws = reference.shape[1]
    upper = np.arange(1, n_draws + 1) / n_draws
    lower = upper - 1.0 / n_draws
    overlap = np.clip(upper - np.maximum(lower, 1.0 - min_success_probability),
                      0.0, None)
    return (overlap[None, :] * np.maximum(threshold - reference, 0.0)).sum(axis=1)


def compute_constraint_diagnostics(
        all_candidates, prediction_reference, threshold,
        min_success_probability, epsilon_grid, max_outcome_wasserstein=None):
    """Diagnose compatibility of validity with the prediction-W1 target."""
    prediction_reference = np.asarray(prediction_reference, dtype=float)
    reference_success = (prediction_reference >= threshold).mean(axis=1)
    reference_valid = reference_success >= min_success_probability
    epsilon_gamma = compatibility_threshold(
        prediction_reference, threshold, min_success_probability)

    minimum_w1 = []
    for candidates in all_candidates:
        valid = [
            candidate["outcome_wasserstein"]
            for candidate in candidates
            if candidate.get("success_probability_xi", 0.0)
            >= min_success_probability
            and np.isfinite(candidate.get("outcome_wasserstein", np.nan))
        ]
        minimum_w1.append(min(valid) if valid else np.nan)
    minimum_w1 = np.asarray(minimum_w1, dtype=float)
    validity_feasible = np.isfinite(minimum_w1)

    coverage = []
    for epsilon in sorted(set(float(value) for value in epsilon_grid)):
        feasible = validity_feasible & (minimum_w1 <= epsilon)

        def conditional_rate(mask):
            return (float(feasible[mask].mean()) if mask.any() else None)

        coverage.append({
            "epsilon": epsilon,
            "coverage": float(feasible.mean()),
            "n_feasible": int(feasible.sum()),
            "coverage_reference_valid": conditional_rate(reference_valid),
            "n_reference_valid": int(reference_valid.sum()),
            "coverage_reference_invalid": conditional_rate(~reference_valid),
            "n_reference_invalid": int((~reference_valid).sum()),
        })

    abstention_causes = None
    if max_outcome_wasserstein is not None:
        epsilon = float(max_outcome_wasserstein)
        no_recourse = ~validity_feasible
        incompatible = validity_feasible & (epsilon < epsilon_gamma)
        unreachable = (validity_feasible & (epsilon >= epsilon_gamma)
                       & (minimum_w1 > epsilon))
        abstention_causes = {
            "epsilon": epsilon,
            "no_recourse_rate": float(no_recourse.mean()),
            "target_incompatible_rate": float(incompatible.mean()),
            "unreachable_rate": float(unreachable.mean()),
            "feasible_rate": float((validity_feasible
                                    & (minimum_w1 <= epsilon)).mean()),
        }

    return {
        "n_recipients": int(len(all_candidates)),
        "classifier_threshold": float(threshold),
        "min_success_probability": float(min_success_probability),
        "reference_success_probability": _finite_summary(reference_success),
        "reference_validity_rate": float(reference_valid.mean()),
        "reference_validity_n": int(reference_valid.sum()),
        "validity_only_coverage": float(validity_feasible.mean()),
        "validity_only_n": int(validity_feasible.sum()),
        "minimum_attainable_w1_among_valid_actions": _finite_summary(minimum_w1),
        "compatibility_threshold": _finite_summary(epsilon_gamma),
        # Theorem: every valid plan has W1 >= epsilon_gamma; should be 0.
        "bound_violations": int(np.sum(
            validity_feasible & (minimum_w1 < epsilon_gamma - 1e-9))),
        "abstention_causes": abstention_causes,
        "coverage_vs_epsilon": coverage,
    }


def summarize(records, n_boot=1000, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    fields = ("cost", "shortfall", "direct_effect", "transport",
              "pre_recourse_mediated_prediction_disparity",
              "pre_recourse_factual_mediated_prediction_disparity",
              "post_recourse_mediated_prediction_disparity")
    result = {field: _ci([r[field] for r in records], n_boot, rng)
              for field in fields}
    if all("outcome_wasserstein" in r for r in records):
        result["outcome_wasserstein"] = _ci(
            [r["outcome_wasserstein"] for r in records], n_boot, rng
        )
    result["feasible_rate"] = _ci(
        [float(r["shortfall"] <= 1e-12) for r in records], n_boot, rng
    )
    result["constraint_feasible_rate"] = _ci(
        [float(r.get("constraint_feasible", False)) for r in records],
        n_boot, rng,
    )
    result["constraint_abstention_rate"] = _ci(
        [float(r.get("constraint_abstained", False)) for r in records],
        n_boot, rng,
    )
    pre = np.asarray([
        r["pre_recourse_factual_mediated_prediction_disparity"] for r in records
    ])
    post = np.asarray([
        r["post_recourse_mediated_prediction_disparity"] for r in records
    ])
    result["mean_absolute_gap_reduction"] = _ci(
        np.abs(pre) - np.abs(post), n_boot, rng)
    point = (1.0 - abs(post.mean()) / abs(pre.mean())
             if abs(pre.mean()) > 1e-12 else np.nan)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(records), len(records))
        denom = abs(pre[idx].mean())
        boot.append(1.0 - abs(post[idx].mean()) / denom
                    if denom > 1e-12 else np.nan)
    result["factual_mediated_prediction_disparity_closure"] = [
        float(point), float(np.nanpercentile(boot, 2.5)),
        float(np.nanpercentile(boot, 97.5))
    ]
    point_l1 = (1.0 - np.abs(post).mean() / np.abs(pre).mean()
                if np.abs(pre).mean() > 1e-12 else np.nan)
    boot_l1 = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(records), len(records))
        denominator = np.abs(pre[idx]).mean()
        boot_l1.append(1.0 - np.abs(post[idx]).mean() / denominator
                       if denominator > 1e-12 else np.nan)
    result["recipient_level_closure"] = [
        float(point_l1), float(np.nanpercentile(boot_l1, 2.5)),
        float(np.nanpercentile(boot_l1, 97.5)),
    ]
    result["overshoot_rate"] = _ci(
        ((pre * post) < 0.0).astype(float), n_boot, rng)
    result["overshoot_magnitude"] = _ci(
        np.maximum(-np.sign(pre) * post, 0.0), n_boot, rng)
    result["n"] = len(records)
    return result


def _paired_baseline_differences(records, baseline_records, n_boot=1000,
                                 rng_seed=0):
    """Paired CIs for closure, cost, and feasibility differences."""
    rng = np.random.default_rng(rng_seed)
    pre = np.asarray([
        r["pre_recourse_factual_mediated_prediction_disparity"]
        for r in records
    ], dtype=float)
    post = np.asarray([
        r["post_recourse_mediated_prediction_disparity"] for r in records
    ], dtype=float)
    post_baseline = np.asarray([
        r["post_recourse_mediated_prediction_disparity"]
        for r in baseline_records
    ], dtype=float)
    cost = np.asarray([r["cost"] for r in records], dtype=float)
    cost_baseline = np.asarray([r["cost"] for r in baseline_records], dtype=float)
    feasible = np.asarray([
        float(r.get("constraint_feasible", r["shortfall"] <= 1e-12))
        for r in records
    ])
    feasible_baseline = np.asarray([
        float(r.get("constraint_feasible", r["shortfall"] <= 1e-12))
        for r in baseline_records
    ])

    def closure(pre_values, post_values):
        denominator = abs(pre_values.mean())
        return (1.0 - abs(post_values.mean()) / denominator
                if denominator > 1e-12 else np.nan)

    point = np.array([
        closure(pre, post) - closure(pre, post_baseline),
        (cost - cost_baseline).mean(),
        (feasible - feasible_baseline).mean(),
        (np.abs(post_baseline) - np.abs(post)).mean(),
    ])
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(records), len(records))
        boots.append([
            closure(pre[idx], post[idx])
            - closure(pre[idx], post_baseline[idx]),
            (cost[idx] - cost_baseline[idx]).mean(),
            (feasible[idx] - feasible_baseline[idx]).mean(),
            (np.abs(post_baseline[idx]) - np.abs(post[idx])).mean(),
        ])
    boots = np.asarray(boots)
    return {
        "factual_closure_gain_vs_actionable_baseline": [
            float(point[0]), float(np.nanpercentile(boots[:, 0], 2.5)),
            float(np.nanpercentile(boots[:, 0], 97.5)),
        ],
        "cost_difference_vs_actionable_baseline": [
            float(point[1]), float(np.percentile(boots[:, 1], 2.5)),
            float(np.percentile(boots[:, 1], 97.5)),
        ],
        "feasibility_difference_vs_actionable_baseline": [
            float(point[2]), float(np.percentile(boots[:, 2], 2.5)),
            float(np.percentile(boots[:, 2], 97.5)),
        ],
        "paired_absolute_gap_gain_vs_actionable_baseline": [
            float(point[3]), float(np.percentile(boots[:, 3], 2.5)),
            float(np.percentile(boots[:, 3], 97.5)),
        ],
    }


def run_grid(all_candidates, references, factual_predictions, eta_grid,
             lambda_grid, rho_grid, disadvantaged_value, n_boot,
             constrained_config=None):
    rows = []
    selected_record_sets = []
    for rho in rho_grid:
        for eta in eta_grid:
            for lam in lambda_grid:
                records = []
                for i, candidates in enumerate(all_candidates):
                    best = dict(select_best(candidates, eta, lam, rho))
                    attach_mediated_disparities(
                        best, references["reference"][i],
                        references["natural_disadvantaged"][i],
                        factual_predictions[i], disadvantaged_value,
                    )
                    records.append(best)
                summary = summarize(records, n_boot=n_boot)
                summary.update(eta=float(eta), lambda_invariance=float(lam),
                               rho_anchor=float(rho))
                if lam == 0.0 and rho == 0.0:
                    method = "soft_objective_recourse"
                elif lam > 0.0 and rho == 0.0:
                    method = "direct_invariance_only"
                elif lam == 0.0 and rho > 0.0:
                    method = "transport_anchor_only"
                else:
                    method = "mediation_aware_recourse"
                summary["method"] = method
                summary["is_ordinary_actionable_baseline"] = False
                rows.append(summary)
                selected_record_sets.append(records)
    if constrained_config is not None:
        records = []
        for i, candidates in enumerate(all_candidates):
            best = dict(select_validity_constrained(
                candidates,
                constrained_config["min_success_probability"],
            ))
            attach_mediated_disparities(
                best, references["reference"][i],
                references["natural_disadvantaged"][i],
                factual_predictions[i], disadvantaged_value,
            )
            records.append(best)
        summary = summarize(records, n_boot=n_boot)
        summary.update(
            eta=0.0,
            lambda_invariance=0.0,
            rho_anchor=0.0,
            min_success_probability=float(
                constrained_config["min_success_probability"]),
            max_outcome_wasserstein=None,
            method="ordinary_actionable_recourse",
            is_ordinary_actionable_baseline=True,
        )
        rows.append(summary)
        selected_record_sets.append(records)

        records = []
        for i, candidates in enumerate(all_candidates):
            best = dict(select_distribution_constrained(
                candidates,
                constrained_config["min_success_probability"],
                constrained_config["max_outcome_wasserstein"],
            ))
            attach_mediated_disparities(
                best, references["reference"][i],
                references["natural_disadvantaged"][i],
                factual_predictions[i], disadvantaged_value,
            )
            records.append(best)
        summary = summarize(records, n_boot=n_boot)
        summary.update(
            eta=float(constrained_config["comparison_eta"]),
            lambda_invariance=0.0,
            rho_anchor=0.0,
            min_success_probability=float(
                constrained_config["min_success_probability"]),
            max_outcome_wasserstein=float(
                constrained_config["max_outcome_wasserstein"]),
            method="distribution_constrained_recourse",
            is_ordinary_actionable_baseline=False,
        )
        rows.append(summary)
        selected_record_sets.append(records)
    soft_baseline_records = {
        row["eta"]: records
        for row, records in zip(rows, selected_record_sets)
        if row["method"] == "soft_objective_recourse"
    }
    clean_baseline_records = next(
        records for row, records in zip(rows, selected_record_sets)
        if row["is_ordinary_actionable_baseline"]
    ) if constrained_config is not None else None
    for row, records in zip(rows, selected_record_sets):
        if row["method"] in (
                "ordinary_actionable_recourse",
                "distribution_constrained_recourse"):
            comparison_records = clean_baseline_records
        else:
            comparison_records = soft_baseline_records[row["eta"]]
        row.update(_paired_baseline_differences(
            records, comparison_records, n_boot=n_boot
        ))
    return rows


def _effect_ratio(path, model_name):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    model = data.get(model_name, {})
    if "effect_ratio_heuristic" in model:
        return float(model["effect_ratio_heuristic"])
    overall = model.get("overall", {})
    nie = overall.get("nie")
    nde = overall.get("nde")
    return float(abs(nde / nie)) if nie not in (None, 0) and nde is not None else None


def _slug(name):
    return (name.lower().replace(" ", "_").replace("(", "").replace(")", "")
            .replace("-", "").replace(".", ""))


def _flatten(row):
    flat = {"method": row["method"],
            "is_ordinary_actionable_baseline":
                row["is_ordinary_actionable_baseline"],
            "eta": row["eta"], "lambda_invariance": row["lambda_invariance"],
            "rho_anchor": row["rho_anchor"], "n": row["n"]}
    for key, value in row.items():
        if isinstance(value, list) and len(value) == 3:
            flat[key] = value[0]
            flat[key + "_lo"] = value[1]
            flat[key + "_hi"] = value[2]
    return flat


def _latex(rows, model_name, label):
    lines = [r"\begin{table}[htbp]", r"\centering",
             rf"\caption{{Independent recourse ablation for {model_name}. "
             r"$\eta$, $\lambda$, and $\rho$ weight shortfall, direct-effect "
             r"invariance, and conservative distributional anchoring.}}",
             rf"\label{{{label}}}", r"\begin{tabular}{lrrrccccccc}",
             r"\toprule",
             r"Method & $\eta$ & $\lambda$ & $\rho$ & Feasible & Direct gap & Transport UB & $D_{pre}$ & $D_{pre}^{fact}$ & $D_{post}$ & Factual closure \\",
             r"\midrule"]
    for row in rows:
        method = {
            "soft_objective_recourse": "Legacy soft objective",
            "ordinary_actionable_recourse": "Actionable baseline",
            "direct_invariance_only": "Invariance only",
            "transport_anchor_only": "Anchor only",
            "mediation_aware_recourse": "Full",
            "distribution_constrained_recourse": "Distribution constrained",
        }[row["method"]]
        lines.append(
            f"{method} & {row['eta']:g} & {row['lambda_invariance']:g} & "
            f"{row['rho_anchor']:g} & "
            f"{100*row['feasible_rate'][0]:.1f}\\% & "
            f"{row['direct_effect'][0]:+.3f} & {row['transport'][0]:.3f} & "
            f"{row['pre_recourse_mediated_prediction_disparity'][0]:+.3f} & "
            f"{row['pre_recourse_factual_mediated_prediction_disparity'][0]:+.3f} & "
            f"{row['post_recourse_mediated_prediction_disparity'][0]:+.3f} & "
            f"{100*row['factual_mediated_prediction_disparity_closure'][0]:+.1f}\\% \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def run(cfg):
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rec = cfg.dataset.recourse
    data = torch.load(cfg.dataset.paths.tensors, map_location="cpu", weights_only=False)
    scaler, vocab = data["scaler"], data["vocab"]
    disc_names = list(cfg.dataset.sfm.mediators_disc)
    cont_names = list(cfg.dataset.sfm.mediators_cont)
    mediator_specs = OmegaConf.to_container(rec.mediators, resolve=True)
    weights = compute_weights(data, disc_names, cont_names)
    g_phi, f_theta, _, flow_sfm_cfg = load_flow_models(
        cfg.dataset.paths.flows, device
    )
    if (bool(rec.get("propagate_descendants", True))
            and "mediator_layers" not in flow_sfm_cfg):
        raise RuntimeError(
            "The configured flow checkpoint predates explicit mediator blocks. "
            "Retrain with flows.train_flow for revised intervention ablations, "
            "or disable propagate_descendants only for legacy reproduction."
        )
    sample_fn, _ = make_sample_fns(g_phi, f_theta, scaler, device)
    schema = MediatorSchema.from_sfm_config(
        OmegaConf.to_container(cfg.dataset.sfm, resolve=True)
    )
    if (bool(rec.get("propagate_descendants", True))
            and MediatorSchema.from_sfm_config(flow_sfm_cfg).layers
            != schema.layers):
        raise RuntimeError(
            "Configured mediator_layers do not match the trained flow "
            "checkpoint. Retrain or restore the checkpoint's schema."
        )

    out_dir = cfg.dataset.paths.recourse_dir
    os.makedirs(out_dir, exist_ok=True)
    model_specs = OmegaConf.to_container(cfg.dataset.outcome.models, resolve=True)
    for spec in model_specs:
        name = spec["name"]
        model_slug = {"logreg": "logreg", "mlp": "mlp"}.get(spec["type"],
                                                               spec["type"])
        pipe = joblib.load(f"{cfg.dataset.paths.outcome_dir}/{model_slug}.joblib")
        idx = select_recourse_indices(
            data, pipe, scaler, int(rec.disadvantaged_value),
            int(rec.sweep_n_max) if int(rec.sweep_n_max) > 0 else None,
            rng_seed=int(cfg.seed),
        )
        print(f"\n{name}: {len(idx)} disadvantaged-group true negatives")
        if len(idx) == 0:
            continue
        references = estimate_reference_terms(
            pipe, scaler, data["Z_va"][idx], sample_fn, K=int(rec.reference_K),
            disadvantaged_value=int(rec.disadvantaged_value),
            advantaged_value=int(rec.advantaged_value),
        )
        propagate = bool(rec.get("propagate_descendants", True))
        if propagate:
            candidates = []
            for i, dataset_idx in enumerate(idx):
                candidates.append(precompute_interventional_candidates_batch(
                    pipe, scaler,
                    data["X_va"][dataset_idx].unsqueeze(0),
                    data["Z_va"][dataset_idx].unsqueeze(0),
                    data["Wd_va"][dataset_idx].unsqueeze(0),
                    data["Wc_va"][dataset_idx].unsqueeze(0),
                    disc_names, cont_names, mediator_specs, vocab, weights,
                    float(rec.threshold), float(rec.nu),
                    g_phi, f_theta, schema, device,
                    references["wd_reference"][i], references["wc_reference"][i],
                    references["prediction_reference"][i],
                    n_samples=int(rec.get("intervention_K", 32)),
                    same_level=str(rec.get(
                        "same_level_semantics", "preserve_factual"
                    )),
                ))
                if (i + 1) % 25 == 0 or i + 1 == len(idx):
                    print(f"    propagated candidates {i+1}/{len(idx)}", end="\r")
            print()
        else:
            pool = enumerate_pool(data, idx, scaler, disc_names, cont_names,
                                  mediator_specs, vocab, weights)
            torch_pipe = build_torch_pipeline(pipe).to(device)
            sanity_check_proba(pipe, torch_pipe, data, scaler, device=device)
            candidates, _, _ = score_pool(
                torch_pipe, pool, float(rec.threshold), float(rec.nu),
                slug="unused", save_dir=out_dir, n_max=len(idx), use_cache=False,
                device=device,
            )
        factual_predictions = []
        for i, (dataset_idx, individual) in enumerate(zip(idx, candidates)):
            wd = data["Wd_va"][dataset_idx].numpy()
            wc_std = data["Wc_va"][dataset_idx].numpy()
            wc = (scaler.inverse_transform(wc_std.reshape(1, -1))[0]
                  if scaler is not None and len(wc_std) else wc_std)
            if not propagate:
                add_candidate_geometry(
                    individual, wd, wc, references["wd_reference"][i],
                    references["wc_reference"][i], disc_names, cont_names,
                    mediator_specs,
                )
            factual = min(individual, key=lambda candidate: candidate["cost"])
            factual_predictions.append(factual["p_xi"])
        max_direct = max(c["direct_effect"] for individual in candidates
                         for c in individual)
        if max_direct <= 1e-10:
            print("  WARNING: classifier is insensitive to X; lambda has no "
                  "effect and the invariance method is out of scope.")

        eta_grid = [float(v) for v in rec.eta_grid]
        lambda_grid = [float(v) for v in rec.lambda_grid]
        heuristic = _effect_ratio(
            f"{cfg.dataset.paths.gaps_dir}/{cfg.dataset.name}_{cfg.dataset.gap.get('output_label', 'gender')}_gap.json", name
        )
        if heuristic is not None and np.isfinite(heuristic):
            lambda_grid = sorted(set(lambda_grid + [heuristic]))
            print(f"  marked heuristic λ=|NDE/NIE|={heuristic:.4g}; not a calibration")
        rho_grid = [float(v) for v in rec.rho_grid]
        constrained_config = None
        constraint_diagnostics = None
        if bool(rec.get("constrained_recourse", {}).get("enabled", False)):
            if not propagate:
                raise ValueError(
                    "Distribution-constrained recourse requires propagated "
                    "intervention draws."
                )
            constrained_config = {
                "min_success_probability": float(
                    rec.constrained_recourse.min_success_probability),
                "max_outcome_wasserstein": float(
                    rec.constrained_recourse.max_outcome_wasserstein),
                "comparison_eta": float(rec.eta),
            }
            epsilon_grid = list(rec.constrained_recourse.get(
                "epsilon_diagnostic_grid",
                [0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5],
            ))
            constraint_diagnostics = compute_constraint_diagnostics(
                candidates, references["prediction_reference"],
                float(rec.threshold),
                constrained_config["min_success_probability"],
                epsilon_grid,
                constrained_config["max_outcome_wasserstein"],
            )
            w1_summary = constraint_diagnostics[
                "minimum_attainable_w1_among_valid_actions"]
            print(
                "  constraint diagnostics: reference-valid="
                f"{100 * constraint_diagnostics['reference_validity_rate']:.1f}%, "
                "valid-action coverage="
                f"{100 * constraint_diagnostics['validity_only_coverage']:.1f}%, "
                f"median min-W1={w1_summary['median']}"
            )
            print("  coverage by epsilon: " + "; ".join(
                f"{item['epsilon']:g}="
                f"{100 * item['coverage']:.1f}%"
                for item in constraint_diagnostics["coverage_vs_epsilon"]
            ))
        rows = run_grid(candidates, references, factual_predictions, eta_grid,
                        lambda_grid, rho_grid, int(rec.disadvantaged_value),
                        int(rec.n_boot), constrained_config)
        gamma_sensitivity = []
        gamma_grid = (
            [float(v) for v in rec.constrained_recourse.get(
                "gamma_sensitivity_grid", [])]
            if constrained_config is not None else []
        )
        for gamma in gamma_grid:
            # Same candidate draws and references; only the validity level moves.
            gamma_config = dict(constrained_config,
                                min_success_probability=gamma)
            gamma_rows = run_grid(
                candidates, references, factual_predictions, [], [], [],
                int(rec.disadvantaged_value), int(rec.n_boot), gamma_config,
            )
            gamma_sensitivity.append({
                "min_success_probability": gamma,
                "constraint_diagnostics": compute_constraint_diagnostics(
                    candidates, references["prediction_reference"],
                    float(rec.threshold), gamma, epsilon_grid,
                    constrained_config["max_outcome_wasserstein"],
                ),
                "rows": gamma_rows,
            })
            by_method = {row["method"]: row for row in gamma_rows}
            print(
                f"  gamma={gamma:.2f}: closure ordinary="
                f"{100 * by_method['ordinary_actionable_recourse']['recipient_level_closure'][0]:+.1f}% "
                f"constrained="
                f"{100 * by_method['distribution_constrained_recourse']['recipient_level_closure'][0]:+.1f}% "
                f"abstain="
                f"{100 * by_method['distribution_constrained_recourse']['constraint_abstention_rate'][0]:.1f}%"
            )
        for row in rows:
            row["is_effect_ratio_heuristic"] = bool(
                heuristic is not None and
                np.isclose(row["lambda_invariance"], heuristic)
            )
        transport_values = np.asarray([row["transport"][0] for row in rows])
        disparity_magnitude = np.abs(
            np.asarray([
                row["post_recourse_mediated_prediction_disparity"][0]
                for row in rows
            ])
        )
        association = {
            "pearson_transport_vs_abs_mediated_post": float(
                np.corrcoef(transport_values, disparity_magnitude)[0, 1]
            ),
            "spearman_transport_vs_abs_mediated_post": float(
                spearmanr(transport_values, disparity_magnitude).statistic
            ),
        }
        print("  transport association with |D_post|: "
              f"Pearson={association['pearson_transport_vs_abs_mediated_post']:+.3f}, "
              f"Spearman={association['spearman_transport_vs_abs_mediated_post']:+.3f}")

        slug = _slug(name)
        json_path = f"{out_dir}/ablation_{cfg.dataset.name}_{slug}.json"
        csv_path = f"{out_dir}/ablation_{cfg.dataset.name}_{slug}.csv"
        tex_path = f"{out_dir}/ablation_{cfg.dataset.name}_{slug}.tex"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"model": name, "seed": int(cfg.seed),
                "artifact_version": "prediction_wasserstein_constraint_v3",
                "legacy_soft_objective": (
                    "cost + eta*shortfall + lambda*direct_effect + rho*transport"
                ),
                "threshold": float(rec.threshold), "nu": float(rec.nu),
                "reference_K": int(rec.reference_K),
                "intervention_K": int(rec.get("intervention_K", 1)),
                "intervention_semantics": (
                    "propagate_strict_descendants" if propagate else "joint_point"
                ),
                "same_level_semantics": str(rec.get(
                    "same_level_semantics", "preserve_factual"
                )),
                "transport_estimand": (
                    "independent_coupling_upper_bound" if propagate
                    else "joint_point_mass_w1"
                ),
                "ordinary_actionable_recourse_baseline": {
                    "min_success_probability": (
                        constrained_config["min_success_probability"]
                        if constrained_config is not None else None
                    ),
                    "lambda_invariance": 0.0,
                    "rho_anchor": 0.0,
                    "description": (
                        "minimum cost subject to the same factual chance-"
                        "validity constraint and abstention rule as the primary "
                        "method, without the prediction-Wasserstein constraint"
                    ),
                },
                "distribution_constrained_recourse": constrained_config,
                "constraint_diagnostics": constraint_diagnostics,
                "distribution_constrained_objective": (
                    "minimize cost subject to factual success probability and "
                    "recipient-conditional prediction-Wasserstein constraints; "
                    "return the factual plan and mark abstention if infeasible"
                ),
                "factual_closure_estimand": (
                    "1 - abs(mean(post_recourse_mediated_prediction_disparity)) "
                    "/ abs(mean(pre_recourse_factual_mediated_prediction_disparity))"
                ),
                "recipient_level_closure_estimand": (
                    "1 - mean(abs(post_recourse_mediated_prediction_disparity)) "
                    "/ mean(abs(pre_recourse_factual_mediated_prediction_disparity))"
                ),
                "interval_scope": "individual bootstrap; fitted models held fixed",
                "population": {
                "group": int(rec.disadvantaged_value), "Y": 0,
                "prediction": 0, "n": len(idx),
            }, "reference": (
                "prediction draws under W|X=advantaged,Z_recipient with "
                "the recipient's Z held fixed"
            ),
                "sweep_association": association,
                "gamma_sensitivity": gamma_sensitivity,
                "rows": rows}, f, indent=2)
        flat = [_flatten(row) | {"is_effect_ratio_heuristic":
                                 row["is_effect_ratio_heuristic"]}
                for row in rows]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat[0]))
            writer.writeheader(); writer.writerows(flat)
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(_latex(rows, name, f"tab:{cfg.dataset.name}_{slug}_ablation"))
        print(f"  Saved {json_path}, {csv_path}, {tex_path}")


@hydra.main(config_path="../conf", config_name="config", version_base="1.1")
def main(cfg):
    run(cfg)


if __name__ == "__main__":
    main()
