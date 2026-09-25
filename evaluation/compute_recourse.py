"""
evaluation/compute_recourse.py — Generic Hydra entry point for counterfactual recourse.

Run: python -m evaluation.compute_recourse [dataset=acs|bar]
"""

import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import joblib

import hydra
from omegaconf import OmegaConf

from data.build_tensors import stack_sfm_features
from flows.interventions import sample_intervention_batch
from flows.models import load_flow_models
from flows.diagnostics import make_sample_fns
from flows.schema import MediatorSchema
from evaluation.recourse_metrics import (
    add_candidate_geometry,
    add_sampled_candidate_geometry,
    attach_mediated_disparities,
    estimate_reference_terms,
    select_best as _select_best,
    select_recourse_indices,
)

SAVE_DIR = "outputs/recourse"


# ── Cost weights from training data ──────────────────────────────────────────

def compute_weights(data: dict, disc_names: list, cont_names: list) -> dict:
    """
    Compute α_j cost weights from the training split.

    For continuous mediators: 1 / Var(wc_natural)
    For discrete mediators:   1 / H(entropy of empirical distribution)
    """
    scaler = data["scaler"]
    vocab  = data["vocab"]
    Wc_tr  = data["Wc_tr"]
    Wd_tr  = data["Wd_tr"]
    weights = {}

    if cont_names and Wc_tr.shape[1] > 0:
        wc_nat = scaler.inverse_transform(Wc_tr.numpy()) if scaler is not None else Wc_tr.numpy()
        for i, name in enumerate(cont_names):
            weights[name] = 1.0 / float(np.var(wc_nat[:, i]))

    for i, name in enumerate(disc_names):
        n_cats = vocab[name]
        counts = np.bincount(Wd_tr[:, i].numpy(), minlength=n_cats).astype(float)
        p = counts / counts.sum()
        weights[name] = 1.0 / float(-np.sum(p * np.log(p + 1e-12)))

    return weights


# ── Vectorised candidate generation ──────────────────────────────────────────

def _build_candidate_arrays(xi_val, zi_np, wdi_np, wci_natural,
                             disc_names, cont_names, mediator_specs,
                             vocab, weights):
    """
    Enumerate all candidate (W') combinations for one individual.

    Returns (feat_xi, feat_xcf, cost_col, change_data) where change_data
    is a dict with keys: delta_{name} for all mediators, changed_{name} for
    discrete mediators.
    """
    xi_cf_val = 1.0 - xi_val

    # Build per-mediator grids
    disc_grids = []
    disc_cur   = []
    for i, name in enumerate(disc_names):
        spec    = mediator_specs[name]
        n_cats  = vocab[name]
        cur_val = int(wdi_np[i])
        disc_cur.append(cur_val)
        actionable = spec.get("actionable", "free")
        if actionable == "monotone_up":
            grid = list(range(cur_val, n_cats))
        else:
            grid = list(range(n_cats))
        disc_grids.append(grid)

    cont_grids = []
    cont_cur   = []
    for i, name in enumerate(cont_names):
        spec    = mediator_specs[name]
        clip    = spec.get("clip", [None, None])
        step    = spec.get("step", 1.0)
        cur_val = float(wci_natural[i])
        cont_cur.append(cur_val)
        clip_lo = clip[0] if clip[0] is not None else cur_val
        clip_hi = clip[1] if clip[1] is not None else cur_val
        grid = np.clip(
            np.arange(cur_val, clip_hi + step, step),
            clip_lo, clip_hi,
        ).astype(np.float32)
        cont_grids.append(grid)

    # Cartesian product
    all_grids = disc_grids + cont_grids
    if not all_grids:
        # No mediators — single no-op candidate
        n_disc = len(disc_names)
        n_cont = len(cont_names)
        n_feat = 1 + len(zi_np) + n_disc + n_cont  # x + z + disc + cont
        feat_xi  = np.zeros((1, n_feat), dtype=np.float32)
        feat_xi[0, 0] = xi_val
        feat_xi[0, 1:1+len(zi_np)] = zi_np
        feat_xcf = feat_xi.copy()
        feat_xcf[0, 0] = xi_cf_val
        cost_col = np.zeros(1, dtype=np.float32)
        change_data = {}
        for nm in disc_names:
            change_data[f"delta_{nm}"] = np.zeros(1, dtype=np.float32)
            change_data[f"changed_{nm}"] = np.zeros(1, dtype=np.int32)
        for nm in cont_names:
            change_data[f"delta_{nm}"] = np.zeros(1, dtype=np.float32)
        return feat_xi, feat_xcf, cost_col, change_data

    meshes = np.meshgrid(*all_grids, indexing='ij')
    shape  = meshes[0].shape
    n      = int(np.prod(shape))

    # Flatten all grid dimensions
    flat = [m.ravel() for m in meshes]

    # Build cost column
    cost_col = np.zeros(n, dtype=np.float32)
    for i, name in enumerate(disc_names):
        cur_val = disc_cur[i]
        alpha   = weights[name]
        spec    = mediator_specs[name]
        actionable = spec.get("actionable", "free")
        vals_flat  = flat[i]
        if actionable == "monotone_up":
            delta = (vals_flat - cur_val).astype(float)
            # cost: 1 if delta > 0
            cost_col += alpha * (delta > 0).astype(np.float32)
        else:
            changed = (vals_flat != cur_val).astype(np.float32)
            cost_col += alpha * changed

    for i, name in enumerate(cont_names):
        cur_val = cont_cur[i]
        alpha   = weights[name]
        vals_flat = flat[len(disc_names) + i]
        delta     = (vals_flat - cur_val).astype(np.float32)
        cost_col += alpha * delta * delta

    # Feature matrix layout: [xi | zi | disc vals | cont vals (natural units)]
    n_z    = len(zi_np)
    n_disc = len(disc_names)
    n_cont = len(cont_names)
    n_feat = 1 + n_z + n_disc + n_cont

    feat_xi         = np.empty((n, n_feat), dtype=np.float32)
    feat_xi[:, 0]   = xi_val
    feat_xi[:, 1:1+n_z] = zi_np[None, :]
    for i in range(n_disc):
        feat_xi[:, 1 + n_z + i] = flat[i]
    for i in range(n_cont):
        feat_xi[:, 1 + n_z + n_disc + i] = flat[len(disc_names) + i]

    feat_xcf       = feat_xi.copy()
    feat_xcf[:, 0] = xi_cf_val

    # Build change_data
    change_data = {}
    for i, name in enumerate(disc_names):
        cur_val = disc_cur[i]
        vals_flat = flat[i]
        change_data[f"delta_{name}"]   = (vals_flat - cur_val).astype(np.float32)
        change_data[f"changed_{name}"] = (vals_flat != cur_val).astype(np.int32)
    for i, name in enumerate(cont_names):
        cur_val = cont_cur[i]
        vals_flat = flat[len(disc_names) + i]
        change_data[f"delta_{name}"] = (vals_flat - cur_val).astype(np.float32)

    return feat_xi, feat_xcf, cost_col, change_data


def precompute_candidates_batch(pipe, scaler, xi, zi, wdi, wci,
                                disc_names, cont_names, mediator_specs,
                                vocab, weights, threshold, nu):
    """
    Evaluate every candidate for one individual.

    Builds feature matrices and calls pipe.predict_proba twice (xi, xi_cf).

    Returns list[dict] — one dict per candidate with keys:
        cost, shortfall, p_xi, p_xcf, delta_{name} (all), changed_{name} (disc)
    """
    xi_val    = float(xi[0, 0].item())
    zi_np     = zi[0].numpy().astype(np.float32)
    wdi_np    = wdi[0].numpy()
    wci_raw   = wci[0].numpy()

    # Convert wci from scaler space to natural units
    if scaler is not None and len(wci_raw) > 0:
        wci_natural = scaler.inverse_transform(wci_raw.reshape(1, -1))[0]
    else:
        wci_natural = wci_raw

    soft_thr = threshold - nu

    feat_xi, feat_xcf, cost_col, change_data = _build_candidate_arrays(
        xi_val, zi_np, wdi_np, wci_natural,
        disc_names, cont_names, mediator_specs, vocab, weights,
    )

    p_xi  = pipe.predict_proba(feat_xi)[:, 1]
    p_xcf = pipe.predict_proba(feat_xcf)[:, 1]
    sf    = np.maximum(0.0, soft_thr - p_xi) + np.maximum(0.0, soft_thr - p_xcf)

    n = len(cost_col)
    return [
        dict(
            cost=float(cost_col[i]),
            shortfall=float(sf[i]),
            p_xi=float(p_xi[i]),
            p_xcf=float(p_xcf[i]),
            **{k: (int(v[i]) if v.dtype == np.int32 else float(v[i]))
               for k, v in change_data.items()},
        )
        for i in range(n)
    ]


def _predict_proba_batched(pipe, features, batch_size=100_000):
    return np.concatenate([
        pipe.predict_proba(features[start:start + batch_size])[:, 1]
        for start in range(0, len(features), batch_size)
    ])


def precompute_interventional_candidates_batch(
        pipe, scaler, xi, zi, wdi, wci,
        disc_names, cont_names, mediator_specs, vocab, weights,
        threshold, nu, g_phi, f_theta, schema, device,
        reference_wd, reference_wc, n_samples=32,
        same_level="preserve_factual"):
    """Evaluate direct action plans after propagating strict descendants.

    Candidate grids specify *directly manipulated* values.  Coordinates left
    at their factual value are not silently interpreted as joint hard
    interventions: later causal blocks are sampled from their learned
    mechanisms, while unacted coordinates in the same unordered block obey
    the explicitly selected ``same_level`` semantics.
    """
    xi_val = float(xi[0, 0].item())
    zi_np = zi[0].numpy().astype(np.float32)
    wdi_np = wdi[0].numpy()
    wci_raw = wci[0].numpy()
    if scaler is not None and len(wci_raw):
        wci_natural = scaler.inverse_transform(wci_raw.reshape(1, -1))[0]
    else:
        wci_natural = wci_raw

    feat_xi, _, cost_col, change_data = _build_candidate_arrays(
        xi_val, zi_np, wdi_np, wci_natural,
        disc_names, cont_names, mediator_specs, vocab, weights,
    )
    n_plans = len(cost_col)
    n_z, n_disc, n_cont = len(zi_np), len(disc_names), len(cont_names)
    mediator_start = 1 + n_z
    target_wd_np = feat_xi[:, mediator_start:mediator_start + n_disc].astype(np.int64)
    target_wc_nat = feat_xi[:, mediator_start + n_disc:].astype(np.float32)
    mask_wd_np = target_wd_np != wdi_np[None, :]
    mask_wc_np = ~np.isclose(target_wc_nat, wci_natural[None, :], atol=1e-7)
    if n_cont and scaler is not None:
        target_wc_std = scaler.transform(target_wc_nat).astype(np.float32)
    else:
        target_wc_std = target_wc_nat

    repeat_rows = lambda tensor: tensor.repeat(n_plans, 1)
    samples = sample_intervention_batch(
        g_phi, f_theta, schema,
        repeat_rows(xi).to(device), repeat_rows(zi).to(device),
        repeat_rows(wdi).to(device), repeat_rows(wci).to(device),
        torch.as_tensor(target_wd_np, dtype=torch.long, device=device),
        torch.as_tensor(target_wc_std, dtype=torch.float32, device=device),
        torch.as_tensor(mask_wd_np, dtype=torch.bool, device=device),
        torch.as_tensor(mask_wc_np, dtype=torch.bool, device=device),
        n_samples=n_samples, same_level=same_level,
    )
    sampled_wd = samples.w_disc.cpu()
    sampled_wc = samples.w_cont.cpu()
    sampled_x = samples.x.cpu()
    sampled_z = samples.z.cpu()
    feat_actual = stack_sfm_features(
        sampled_x, sampled_z, sampled_wd, sampled_wc, scaler
    )
    feat_counterfactual = stack_sfm_features(
        1.0 - sampled_x, sampled_z, sampled_wd, sampled_wc, scaler
    )
    p_actual_draws = _predict_proba_batched(pipe, feat_actual).reshape(n_plans, n_samples)
    p_cf_draws = _predict_proba_batched(pipe, feat_counterfactual).reshape(n_plans, n_samples)
    soft_thr = threshold - nu
    shortfall = (
        np.maximum(0.0, soft_thr - p_actual_draws)
        + np.maximum(0.0, soft_thr - p_cf_draws)
    ).mean(axis=1)

    candidates = []
    for i in range(n_plans):
        candidates.append(dict(
            cost=float(cost_col[i]),
            shortfall=float(shortfall[i]),
            p_xi=float(p_actual_draws[i].mean()),
            p_xcf=float(p_cf_draws[i].mean()),
            direct_effect=float(np.abs(p_cf_draws[i] - p_actual_draws[i]).mean()),
            success_probability_xi=float((p_actual_draws[i] >= soft_thr).mean()),
            success_probability_xcf=float((p_cf_draws[i] >= soft_thr).mean()),
            outcome_sd_xi=float(p_actual_draws[i].std()),
            outcome_sd_xcf=float(p_cf_draws[i].std()),
            intervention_semantics="propagate_strict_descendants",
            same_level_semantics=same_level,
            **{key: (int(value[i]) if value.dtype == np.int32 else float(value[i]))
               for key, value in change_data.items()},
        ))

    sampled_wd_np = sampled_wd.numpy().reshape(n_plans, n_samples, n_disc)
    sampled_wc_std = sampled_wc.numpy()
    if n_cont and scaler is not None:
        sampled_wc_nat = scaler.inverse_transform(sampled_wc_std)
    else:
        sampled_wc_nat = sampled_wc_std
    sampled_wc_nat = sampled_wc_nat.reshape(n_plans, n_samples, n_cont)
    add_sampled_candidate_geometry(
        candidates, sampled_wd_np, sampled_wc_nat,
        reference_wd, reference_wc,
        disc_names, cont_names, mediator_specs,
    )
    return candidates


def select_best(candidates: list, eta: float, lambda_invariance: float = 0.0,
                rho_anchor: float = 0.0) -> dict:
    """Backward-compatible export of the correctly separated objective."""
    return _select_best(candidates, eta, lambda_invariance, rho_anchor)


# ── Batch recourse ────────────────────────────────────────────────────────────

def run_recourse_batch(pipe, scaler, data, disc_names, cont_names,
                       mediator_specs, vocab, weights,
                       threshold, nu, eta, lambda_invariance, rho_anchor,
                       sample_fn, g_phi=None, f_theta=None, schema=None,
                       device=None, propagate_descendants=True,
                       intervention_K=32,
                       same_level_semantics="preserve_factual",
                       reference_K=200, disadvantaged_value=0,
                       advantaged_value=1, n_max=None, rng_seed=0):
    """
    Generate recourse for disadvantaged-group true negatives only and evaluate
    the post-recourse mediated disparity on those recipients' Z distribution.
    """
    X_va  = data["X_va"]
    Z_va  = data["Z_va"]
    Wd_va = data["Wd_va"]
    Wc_va = data["Wc_va"]
    tn_idx = select_recourse_indices(
        data, pipe, scaler, disadvantaged_value, n_max, rng_seed,
    )
    print(f"  Disadvantaged-group true negatives: {len(tn_idx):,}")
    references = estimate_reference_terms(
        pipe, scaler, Z_va[tn_idx], sample_fn, K=reference_K,
        disadvantaged_value=disadvantaged_value,
        advantaged_value=advantaged_value,
    )

    results = []
    t0      = time.time()

    for k, i in enumerate(tn_idx):
        wd_fact = Wd_va[i].numpy()
        wc_raw = Wc_va[i].numpy()
        wc_fact = (scaler.inverse_transform(wc_raw.reshape(1, -1))[0]
                   if scaler is not None and len(wc_raw) else wc_raw)
        if propagate_descendants:
            if any(value is None for value in (g_phi, f_theta, schema, device)):
                raise ValueError(
                    "Flow models, schema, and device are required when "
                    "propagate_descendants=True."
                )
            cands = precompute_interventional_candidates_batch(
                pipe, scaler,
                X_va[i].unsqueeze(0), Z_va[i].unsqueeze(0),
                Wd_va[i].unsqueeze(0), Wc_va[i].unsqueeze(0),
                disc_names, cont_names, mediator_specs, vocab, weights,
                threshold, nu, g_phi, f_theta, schema, device,
                references["wd_reference"][k], references["wc_reference"][k],
                n_samples=intervention_K,
                same_level=same_level_semantics,
            )
        else:
            cands = precompute_candidates_batch(
                pipe, scaler,
                X_va[i].unsqueeze(0), Z_va[i].unsqueeze(0),
                Wd_va[i].unsqueeze(0), Wc_va[i].unsqueeze(0),
                disc_names, cont_names, mediator_specs, vocab, weights,
                threshold, nu,
            )
            add_candidate_geometry(
                cands, wd_fact, wc_fact,
                references["wd_reference"][k], references["wc_reference"][k],
                disc_names, cont_names, mediator_specs,
            )
        best = dict(select_best(cands, eta, lambda_invariance, rho_anchor))
        factual = min(cands, key=lambda c: c["cost"])
        factual_prediction = factual["p_xi"]
        attach_mediated_disparities(
            best, references["reference"][k],
            references["natural_disadvantaged"][k], factual_prediction,
            disadvantaged_value,
        )
        best["group"] = disadvantaged_value
        results.append(best)

        if (k + 1) % 50 == 0 or (k + 1) == len(tn_idx):
            elapsed = time.time() - t0
            rate    = (k + 1) / max(elapsed, 1e-6)
            eta_s   = (len(tn_idx) - k - 1) / max(rate, 1e-6)
            print(f"    {k+1:5d}/{len(tn_idx)}  "
                  f"({rate:.1f} ind/s  ETA {eta_s:.0f}s)", end="\r")
    print()

    return results


# ── Aggregation ───────────────────────────────────────────────────────────────

def _bootstrap_ci(arr, n_boot=1000, alpha=0.05, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    boots = np.array([
        arr[rng.choice(len(arr), len(arr), replace=True)].mean()
        for _ in range(n_boot)
    ])
    return (float(arr.mean()),
            float(np.percentile(boots, 100 * alpha / 2)),
            float(np.percentile(boots, 100 * (1 - alpha / 2))))


def _bootstrap_absolute_closure(pre, post, n_boot=1000, alpha=0.05, rng=None):
    """Fraction of the absolute population disparity closed after recourse."""
    if rng is None:
        rng = np.random.default_rng(0)
    pre = np.asarray(pre, dtype=float)
    post = np.asarray(post, dtype=float)
    if len(pre) == 0 or abs(pre.mean()) <= 1e-12:
        return (float("nan"),) * 3
    point = 1.0 - abs(post.mean()) / abs(pre.mean())
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(pre), len(pre))
        denominator = abs(pre[idx].mean())
        boots.append(
            1.0 - abs(post[idx].mean()) / denominator
            if denominator > 1e-12 else np.nan
        )
    return (float(point),
            float(np.nanpercentile(boots, 100 * alpha / 2)),
            float(np.nanpercentile(boots, 100 * (1 - alpha / 2))))


def aggregate(results: list, disc_names: list, cont_names: list,
              mediator_specs: dict, n_boot: int = 1000) -> dict:
    """
    Aggregate recourse results. For each mediator, bootstrap the right metric:
      - monotone_up discrete and all continuous: delta_{name}
      - free discrete: changed_{name}
    """
    n = len(results)
    if n == 0:
        nan3 = (float("nan"),) * 3
        base = {"n": 0, "n_feasible": 0, "feasible_rate": nan3,
                "cost": nan3, "shortfall": nan3, "direct_effect": nan3,
                "transport": nan3,
                "pre_recourse_mediated_prediction_disparity": nan3,
                "pre_recourse_factual_mediated_prediction_disparity": nan3,
                "post_recourse_mediated_prediction_disparity": nan3,
                "factual_mediated_prediction_disparity_closure": nan3}
        for name in disc_names + cont_names:
            base[name] = nan3
        return base

    rng  = np.random.default_rng(0)
    feas = np.array([float(r["shortfall"] == 0.0) for r in results])
    d_cost = np.array([r["cost"]       for r in results], dtype=float)
    d_sf   = np.array([r["shortfall"]  for r in results], dtype=float)
    d_direct = np.array([r["direct_effect"] for r in results], dtype=float)
    d_transport = np.array([r["transport"] for r in results], dtype=float)
    d_pre = np.array([
        r["pre_recourse_mediated_prediction_disparity"] for r in results
    ], dtype=float)
    d_pre_fact = np.array([
        r["pre_recourse_factual_mediated_prediction_disparity"] for r in results
    ], dtype=float)
    d_post = np.array([
        r["post_recourse_mediated_prediction_disparity"] for r in results
    ], dtype=float)

    out = {
        "n":             n,
        "n_feasible":    int(feas.sum()),
        "feasible_rate": _bootstrap_ci(feas,   n_boot, rng=rng),
        "cost":          _bootstrap_ci(d_cost, n_boot, rng=rng),
        "shortfall":     _bootstrap_ci(d_sf,   n_boot, rng=rng),
        "direct_effect": _bootstrap_ci(d_direct, n_boot, rng=rng),
        "transport": _bootstrap_ci(d_transport, n_boot, rng=rng),
        "pre_recourse_mediated_prediction_disparity": _bootstrap_ci(
            d_pre, n_boot, rng=rng
        ),
        "pre_recourse_factual_mediated_prediction_disparity": _bootstrap_ci(
            d_pre_fact, n_boot, rng=rng
        ),
        "post_recourse_mediated_prediction_disparity": _bootstrap_ci(
            d_post, n_boot, rng=rng
        ),
        "factual_mediated_prediction_disparity_closure": _bootstrap_absolute_closure(
            d_pre_fact, d_post, n_boot, rng=rng
        ),
    }

    for name in disc_names:
        spec = mediator_specs[name]
        actionable = spec.get("actionable", "free")
        if actionable == "free":
            arr = np.array([r[f"changed_{name}"] for r in results], dtype=float)
        else:
            arr = np.array([r[f"delta_{name}"] for r in results], dtype=float)
        out[name] = _bootstrap_ci(arr, n_boot, rng=rng)

    for name in cont_names:
        arr = np.array([r[f"delta_{name}"] for r in results], dtype=float)
        out[name] = _bootstrap_ci(arr, n_boot, rng=rng)

    return out


def stratify(results: list, disc_names: list, cont_names: list,
             mediator_specs: dict, n_boot: int = 1000) -> dict:
    # Recourse is intentionally restricted to one disadvantaged group. Keeping
    # only an overall recipient summary avoids an undefined advantaged-group
    # "post-recourse" estimand.
    return {"recipients": aggregate(results, disc_names, cont_names,
                                     mediator_specs, n_boot)}


# ── LaTeX table ───────────────────────────────────────────────────────────────

def _ci_cell(mean, lo, hi, fmt=".2f"):
    return rf"${mean:{fmt}}\ [{lo:{fmt}},\ {hi:{fmt}}]$"


def _pct_cell(mean, lo, hi):
    return rf"${100*mean:.1f}\ [{100*lo:.1f},\ {100*hi:.1f}]\%$"


def make_recourse_table(all_stats: dict, model_names: list,
                        disc_names: list, cont_names: list,
                        mediator_specs: dict,
                        caption: str, label: str) -> str:
    # Build dynamic column headers from mediator config
    med_headers = []
    for name in disc_names + cont_names:
        display = mediator_specs[name].get("display", name)
        med_headers.append(display)

    n_med_cols = len(med_headers)
    col_spec = "l" + "c" * (10 + n_med_cols)
    header_cells = " & ".join(med_headers)

    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        r"  \setlength{\tabcolsep}{4pt}",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
        f"    Model & $n$ & Feasible ($S{{=}}0$) & "
        f"{header_cells}"
        r" & Cost & Shortfall & Direct gap & Transport UB & $D_{\rm pre}$ & $D_{\rm pre}^{\rm factual}$ & $D_{\rm post}$ & Factual closure \\",
        r"    \midrule",
    ]
    for mi, name in enumerate(model_names):
        if mi > 0:
            lines.append(r"    \midrule")
        s = all_stats[name]["recipients"]
        med_cells = []
        for mname in disc_names:
            spec = mediator_specs[mname]
            med_cells.append(_pct_cell(*s[mname]) if spec.get("actionable", "free") == "free"
                             else _ci_cell(*s[mname]))
        for mname in cont_names:
            med_cells.append(_ci_cell(*s[mname], fmt=".1f"))
        lines.append(
            f"    {name} & {s['n']} & {_pct_cell(*s['feasible_rate'])} & "
            f"{' & '.join(med_cells)} & {_ci_cell(*s['cost'], fmt='.3f')} & "
            f"{_ci_cell(*s['shortfall'], fmt='.4f')} & "
            f"{_ci_cell(*s['direct_effect'], fmt='.4f')} & "
            f"{_ci_cell(*s['transport'], fmt='.4f')} & "
            f"{_ci_cell(*s['pre_recourse_mediated_prediction_disparity'], fmt='.4f')} & "
            f"{_ci_cell(*s['pre_recourse_factual_mediated_prediction_disparity'], fmt='.4f')} & "
            f"{_ci_cell(*s['post_recourse_mediated_prediction_disparity'], fmt='.4f')} & "
            f"{_pct_cell(*s['factual_mediated_prediction_disparity_closure'])} \\\\"
        )
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ── Human-readable summary ────────────────────────────────────────────────────

def print_summary(all_stats: dict, model_names: list,
                  disc_names: list, cont_names: list,
                  mediator_specs: dict) -> str:
    lines = []
    for name in model_names:
        hdr = f"\n{'='*60}\n{name}\n{'='*60}"
        print(hdr)
        lines.append(hdr)
        for key, label_str in [("recipients", "Recipients")]:
            s = all_stats[name][key]
            fr  = s["feasible_rate"]
            med_parts = []
            for mname in disc_names:
                spec = mediator_specs[mname]
                actionable = spec.get("actionable", "free")
                if actionable == "free":
                    med_parts.append(f"{mname}%={100*s[mname][0]:.1f}")
                else:
                    med_parts.append(f"Δ{mname}={s[mname][0]:.2f}")
            for mname in cont_names:
                med_parts.append(f"Δ{mname}={s[mname][0]:.1f}")
            med_str = "  ".join(med_parts)
            row = (
                f"  [{label_str:7s}]  n={s['n']:5d}  "
                f"feasible={fr[0]:.1%} [{fr[1]:.1%},{fr[2]:.1%}]  "
                f"{med_str}  "
                f"cost={s['cost'][0]:.3f}  "
                f"shortfall={s['shortfall'][0]:.4f}  "
                f"direct={s['direct_effect'][0]:+.4f}  "
                f"transport_UB={s['transport'][0]:.4f}  "
                f"D_pre={s['pre_recourse_mediated_prediction_disparity'][0]:+.4f}  "
                f"D_pre_factual={s['pre_recourse_factual_mediated_prediction_disparity'][0]:+.4f}  "
                f"D_post={s['post_recourse_mediated_prediction_disparity'][0]:+.4f}  "
                f"factual_closure={s['factual_mediated_prediction_disparity_closure'][0]:+.1%}"
            )
            print(row)
            lines.append(row)
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

@hydra.main(config_path="../conf", config_name="config", version_base="1.1")
def main(cfg):
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    tensors_path   = cfg.dataset.paths.tensors
    outcome_dir    = cfg.dataset.paths.outcome_dir
    recourse_dir   = cfg.dataset.paths.recourse_dir
    dataset_name   = cfg.dataset.name

    data    = torch.load(tensors_path, map_location="cpu", weights_only=False)
    scaler  = data["scaler"]
    vocab   = data["vocab"]

    disc_names     = list(cfg.dataset.sfm.mediators_disc)
    cont_names     = list(cfg.dataset.sfm.mediators_cont)
    mediator_specs = OmegaConf.to_container(cfg.dataset.recourse.mediators, resolve=True)

    weights = compute_weights(data, disc_names, cont_names)

    print("Cost weights (from training data):")
    for name, w in weights.items():
        print(f"  α_{name} = {w:.6f}")

    model_specs    = OmegaConf.to_container(cfg.dataset.outcome.models, resolve=True)
    recourse_cfg   = cfg.dataset.recourse
    type_to_slug   = {"logreg": "logreg", "mlp": "mlp"}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g_phi, f_theta, _, flow_sfm_cfg = load_flow_models(
        cfg.dataset.paths.flows, device
    )
    if (bool(recourse_cfg.get("propagate_descendants", True))
            and "mediator_layers" not in flow_sfm_cfg):
        raise RuntimeError(
            "The configured flow checkpoint predates explicit mediator blocks. "
            "Retrain with flows.train_flow for revised intervention results, "
            "or set dataset.recourse.propagate_descendants=false only to "
            "reproduce the legacy joint-point analysis."
        )
    sample_fn, _ = make_sample_fns(g_phi, f_theta, scaler, device)
    schema = MediatorSchema.from_sfm_config(
        OmegaConf.to_container(cfg.dataset.sfm, resolve=True)
    )
    if (bool(recourse_cfg.get("propagate_descendants", True))
            and MediatorSchema.from_sfm_config(flow_sfm_cfg).layers
            != schema.layers):
        raise RuntimeError(
            "Configured mediator_layers do not match the trained flow "
            "checkpoint. Retrain or restore the checkpoint's schema."
        )
    eta = float(recourse_cfg.eta)
    lambda_invariance = float(recourse_cfg.lambda_invariance)
    rho_anchor = float(recourse_cfg.rho_anchor)

    n_max_cfg = int(recourse_cfg.n_max)
    n_max     = n_max_cfg if n_max_cfg > 0 else None

    all_stats   = {}
    model_names = []

    for spec in model_specs:
        name = spec["name"]
        slug = type_to_slug.get(spec["type"], spec["type"])
        pipe = joblib.load(f"{outcome_dir}/{slug}.joblib")
        print(f"\n[{name}]  η={eta:.4f}  λ={lambda_invariance:.4f}  "
              f"ρ={rho_anchor:.4f}  τ={recourse_cfg.threshold}  "
              f"ν={recourse_cfg.nu}  soft_thr={recourse_cfg.threshold - recourse_cfg.nu:.3f}  "
              f"propagate={bool(recourse_cfg.get('propagate_descendants', True))}")
        results = run_recourse_batch(
            pipe, scaler, data,
            disc_names, cont_names, mediator_specs, vocab, weights,
            threshold=float(recourse_cfg.threshold),
            nu=float(recourse_cfg.nu),
            eta=eta,
            lambda_invariance=lambda_invariance,
            rho_anchor=rho_anchor,
            sample_fn=sample_fn,
            g_phi=g_phi,
            f_theta=f_theta,
            schema=schema,
            device=device,
            propagate_descendants=bool(
                recourse_cfg.get("propagate_descendants", True)
            ),
            intervention_K=int(recourse_cfg.get("intervention_K", 32)),
            same_level_semantics=str(recourse_cfg.get(
                "same_level_semantics", "preserve_factual"
            )),
            reference_K=int(recourse_cfg.reference_K),
            disadvantaged_value=int(recourse_cfg.disadvantaged_value),
            advantaged_value=int(recourse_cfg.advantaged_value),
            n_max=n_max,
            rng_seed=int(cfg.seed),
        )
        all_stats[name] = stratify(results, disc_names, cont_names,
                                   mediator_specs, n_boot=int(recourse_cfg.n_boot))
        model_names.append(name)

    summary = print_summary(all_stats, model_names, disc_names, cont_names, mediator_specs)
    os.makedirs(recourse_dir, exist_ok=True)

    tex = make_recourse_table(
        all_stats, model_names, disc_names, cont_names, mediator_specs,
        caption=(
            f"Recourse for disadvantaged-group {dataset_name} true negatives. "
            rf"Objective weights: $\eta={eta:g}$, "
            rf"$\lambda={lambda_invariance:g}$, $\rho={rho_anchor:g}$. "
            rf"Direct actions propagate strict descendants using "
            rf"$K={int(recourse_cfg.get('intervention_K', 32))}$ draws; "
            r"transport is an independent-coupling upper bound. "
            r"$D_{\rm post}$ is a direct plug-in post-recourse mediated disparity. "
            r"Values: mean with 95\% bootstrap CI."
        ),
        label=f"tab:{dataset_name}_recourse",
    )

    tex_path = f"{recourse_dir}/{dataset_name}_recourse.tex"
    txt_path = f"{recourse_dir}/{dataset_name}_recourse.txt"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"\nSaved → {tex_path}")
    print(f"Saved → {txt_path}")
    print("\n" + tex)


if __name__ == "__main__":
    main()
