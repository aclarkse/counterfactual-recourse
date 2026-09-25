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
    select_recourse_indices,
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


def summarize(records, n_boot=1000, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    fields = ("cost", "shortfall", "direct_effect", "transport",
              "pre_recourse_mediated_prediction_disparity",
              "pre_recourse_factual_mediated_prediction_disparity",
              "post_recourse_mediated_prediction_disparity")
    result = {field: _ci([r[field] for r in records], n_boot, rng)
              for field in fields}
    result["feasible_rate"] = _ci(
        [float(r["shortfall"] <= 1e-12) for r in records], n_boot, rng
    )
    pre = np.asarray([
        r["pre_recourse_factual_mediated_prediction_disparity"] for r in records
    ])
    post = np.asarray([
        r["post_recourse_mediated_prediction_disparity"] for r in records
    ])
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
        float(r["shortfall"] <= 1e-12) for r in records
    ])
    feasible_baseline = np.asarray([
        float(r["shortfall"] <= 1e-12) for r in baseline_records
    ])

    def closure(pre_values, post_values):
        denominator = abs(pre_values.mean())
        return (1.0 - abs(post_values.mean()) / denominator
                if denominator > 1e-12 else np.nan)

    point = np.array([
        closure(pre, post) - closure(pre, post_baseline),
        (cost - cost_baseline).mean(),
        (feasible - feasible_baseline).mean(),
    ])
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(records), len(records))
        boots.append([
            closure(pre[idx], post[idx])
            - closure(pre[idx], post_baseline[idx]),
            (cost[idx] - cost_baseline[idx]).mean(),
            (feasible[idx] - feasible_baseline[idx]).mean(),
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
    }


def run_grid(all_candidates, references, factual_predictions, eta_grid,
             lambda_grid, rho_grid, disadvantaged_value, n_boot):
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
                    method = "ordinary_actionable_recourse"
                elif lam > 0.0 and rho == 0.0:
                    method = "direct_invariance_only"
                elif lam == 0.0 and rho > 0.0:
                    method = "transport_anchor_only"
                else:
                    method = "mediation_aware_recourse"
                summary["method"] = method
                summary["is_ordinary_actionable_baseline"] = (
                    method == "ordinary_actionable_recourse"
                )
                rows.append(summary)
                selected_record_sets.append(records)
    baseline_records = {
        row["eta"]: records
        for row, records in zip(rows, selected_record_sets)
        if row["is_ordinary_actionable_baseline"]
    }
    for row, records in zip(rows, selected_record_sets):
        row.update(_paired_baseline_differences(
            records, baseline_records[row["eta"]], n_boot=n_boot
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
            "ordinary_actionable_recourse": "Actionable baseline",
            "direct_invariance_only": "Invariance only",
            "transport_anchor_only": "Anchor only",
            "mediation_aware_recourse": "Full",
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
            f"{cfg.dataset.paths.gaps_dir}/{cfg.dataset.name}_gender_gap.json", name
        )
        if heuristic is not None and np.isfinite(heuristic):
            lambda_grid = sorted(set(lambda_grid + [heuristic]))
            print(f"  marked heuristic λ=|NDE/NIE|={heuristic:.4g}; not a calibration")
        rho_grid = [float(v) for v in rec.rho_grid]
        rows = run_grid(candidates, references, factual_predictions, eta_grid,
                        lambda_grid, rho_grid, int(rec.disadvantaged_value),
                        int(rec.n_boot))
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
                "objective": "cost + eta*shortfall + lambda*direct_effect + rho*transport",
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
                    "lambda_invariance": 0.0,
                    "rho_anchor": 0.0,
                    "description": (
                        "minimum-cost validity recourse with the same action "
                        "set and descendant-propagation semantics"
                    ),
                },
                "factual_closure_estimand": (
                    "1 - abs(mean(post_recourse_mediated_prediction_disparity)) "
                    "/ abs(mean(pre_recourse_factual_mediated_prediction_disparity))"
                ),
                "interval_scope": "individual bootstrap; fitted models held fixed",
                "population": {
                "group": int(rec.disadvantaged_value), "Y": 0,
                "prediction": 0, "n": len(idx),
            }, "reference": "direct draws W|X=advantaged,Z_recipient",
                "sweep_association": association, "rows": rows}, f, indent=2)
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
