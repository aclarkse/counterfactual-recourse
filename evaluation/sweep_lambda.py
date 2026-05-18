"""
evaluation/sweep_lambda.py — Generic λ sensitivity sweep with PyTorch-accelerated inference.

Run: python -m evaluation.sweep_lambda [dataset=acs|bar]
     python -m evaluation.sweep_lambda dataset=acs recourse.n_max=300
"""

import json
import os
import pickle
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import joblib

import hydra
from omegaconf import OmegaConf

from data.build_tensors import stack_sfm_features
from outcome.models import build_torch_pipeline
from evaluation.compute_recourse import (
    compute_weights,
    _build_candidate_arrays,
    select_best,
    _bootstrap_ci,
)

INF_ETA  = 1e9          # proxy for λ = ∞
_BATCH   = 500_000      # max rows per PyTorch forward pass (memory guard)


# ── Sanity checks ─────────────────────────────────────────────────────────────

def sanity_check_proba(pipe, torch_pipe, data, scaler,
                       n_check: int = 500, tol: float = 1e-5,
                       device=None) -> float:
    """
    Verify |p_torch − p_sklearn|_max ≤ tol on n_check validation rows.
    Returns the max absolute difference.
    """
    X   = data["X_va"][:n_check]
    Z   = data["Z_va"][:n_check]
    Wd  = data["Wd_va"][:n_check]
    Wc  = data["Wc_va"][:n_check]

    feat   = stack_sfm_features(X, Z, Wd, Wc, scaler)
    p_sk   = pipe.predict_proba(feat)[:, 1]

    feat_t = torch.tensor(feat, dtype=torch.float32)
    if device is not None:
        feat_t = feat_t.to(device)
    torch_pipe.eval()
    with torch.no_grad():
        p_pt = torch_pipe(feat_t).cpu().numpy()

    diff = np.abs(p_sk - p_pt)
    print(f"    predict_proba agreement:  max={diff.max():.2e}  "
          f"mean={diff.mean():.2e}  (n={n_check})")
    if diff.max() > tol:
        raise RuntimeError(
            f"PyTorch model disagrees with sklearn:  "
            f"max|Δp| = {diff.max():.2e} > tol = {tol:.2e}"
        )
    return float(diff.max())


def sanity_check_sweep(pipe, torch_pipe, data, scaler,
                       disc_names, cont_names, mediator_specs,
                       vocab, weights, threshold, nu, leb,
                       n_check: int = 50, device=None) -> None:
    """
    On n_check true negatives, verify that best candidate selected by
    PyTorch at λ = λ_EB matches sklearn to within cost ≤ 1e-3 and the same
    discrete choices.
    """
    from evaluation.compute_recourse import precompute_candidates_batch

    X_va  = data["X_va"]
    Z_va  = data["Z_va"]
    Wd_va = data["Wd_va"]
    Wc_va = data["Wc_va"]
    Y_va  = data["Y_va"].numpy().astype(int)

    feat_all = stack_sfm_features(X_va, Z_va, Wd_va, Wc_va, scaler)
    yhat_all = pipe.predict(feat_all)
    tn_idx   = np.where((Y_va == 0) & (yhat_all == 0))[0][:n_check]

    cost_diffs  = []
    disc_match  = []
    torch_pipe.eval()

    for i in tn_idx:
        # sklearn candidates
        cands_sk = precompute_candidates_batch(
            pipe, scaler,
            X_va[i].unsqueeze(0), Z_va[i].unsqueeze(0),
            Wd_va[i].unsqueeze(0), Wc_va[i].unsqueeze(0),
            disc_names, cont_names, mediator_specs, vocab, weights,
            threshold, nu,
        )
        best_sk = select_best(cands_sk, leb)

        # PyTorch candidates (single individual)
        cands_pt = _precompute_one_torch(
            torch_pipe, scaler,
            X_va[i].unsqueeze(0), Z_va[i].unsqueeze(0),
            Wd_va[i].unsqueeze(0), Wc_va[i].unsqueeze(0),
            disc_names, cont_names, mediator_specs, vocab, weights,
            threshold, nu, device,
        )
        best_pt = select_best(cands_pt, leb)

        cost_diffs.append(abs(best_sk["cost"] - best_pt["cost"]))
        # Compare all disc mediator deltas
        disc_ok = all(
            best_sk.get(f"delta_{nm}", best_sk.get(f"changed_{nm}", None)) ==
            best_pt.get(f"delta_{nm}", best_pt.get(f"changed_{nm}", None))
            for nm in disc_names
        )
        disc_match.append(disc_ok)

    match_rate    = sum(disc_match) / len(disc_match)
    max_cost_diff = max(cost_diffs) if cost_diffs else 0.0
    print(f"    Sweep agreement (n={len(tn_idx)}, λ=λ_EB):  "
          f"discrete match={match_rate:.1%}  "
          f"max|Δcost|={max_cost_diff:.2e}")
    if match_rate < 0.95 or max_cost_diff > 1e-3:
        print("    WARNING: sweep results differ more than expected.")


# ── λ_EB from gap JSON ────────────────────────────────────────────────────────

def load_lambda_eb(gap_json: str) -> dict:
    with open(gap_json, encoding="utf-8") as f:
        snap = json.load(f)
    result = {}
    for model, strata in snap.items():
        nde = strata["overall"]["nde"]
        nie = strata["overall"]["nie"]
        if abs(nie) < 1e-12:
            raise ValueError(f"NIE ≈ 0 for '{model}' — λ_EB undefined.")
        result[model] = abs(nde / nie)
        print(f"  {model:20s}  NDE={nde:+.4f}  NIE={nie:+.4f}  "
              f"λ_EB={result[model]:.4f}")
    return result


# ── Candidate cache I/O ───────────────────────────────────────────────────────

def _cache_path(save_dir: str, slug: str, n_max: int,
                threshold: float, nu: float) -> str:
    # "scored_…_y0" prefix: distinct from legacy per-model-TN caches so
    # stale pools (built before the shared Y=0 eval set) are never reused.
    return (f"{save_dir}/scored_{slug}_y0"
            f"_n{n_max}_tau{threshold:.2f}_nu{nu:.2f}.pkl")


# ── PyTorch global-batch candidate generation ─────────────────────────────────

def _precompute_one_torch(torch_pipe, scaler, xi, zi, wdi, wci,
                           disc_names, cont_names, mediator_specs,
                           vocab, weights, threshold, nu, device):
    """Single-individual PyTorch candidate pool (used by sanity check)."""
    xi_val   = float(xi[0, 0].item())
    zi_np    = zi[0].numpy().astype(np.float32)
    wdi_np   = wdi[0].numpy()
    wci_raw  = wci[0].numpy()

    if scaler is not None and len(wci_raw) > 0:
        wci_natural = scaler.inverse_transform(wci_raw.reshape(1, -1))[0]
    else:
        wci_natural = wci_raw

    soft_thr = threshold - nu

    feat_xi, feat_xcf, cost_col, change_data = _build_candidate_arrays(
        xi_val, zi_np, wdi_np, wci_natural,
        disc_names, cont_names, mediator_specs, vocab, weights,
    )
    n = len(cost_col)

    with torch.no_grad():
        t_xi  = torch.tensor(feat_xi,  dtype=torch.float32, device=device)
        t_xcf = torch.tensor(feat_xcf, dtype=torch.float32, device=device)
        p_xi  = torch_pipe(t_xi).cpu().numpy()
        p_xcf = torch_pipe(t_xcf).cpu().numpy()

    sf = np.maximum(0.0, soft_thr - p_xi) + np.maximum(0.0, soft_thr - p_xcf)

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


def _torch_predict_batched(torch_pipe, feat_np: np.ndarray, device) -> np.ndarray:
    """Run torch_pipe on feat_np in _BATCH-sized chunks to guard GPU memory."""
    n      = len(feat_np)
    out    = np.empty(n, dtype=np.float32)
    torch_pipe.eval()
    with torch.no_grad():
        for start in range(0, n, _BATCH):
            end    = min(start + _BATCH, n)
            t      = torch.tensor(feat_np[start:end], dtype=torch.float32,
                                  device=device)
            out[start:end] = torch_pipe(t).cpu().numpy()
    return out


def select_eval_idx(data, n_max: int, rng_seed: int = 0) -> np.ndarray:
    """
    Model-independent evaluation set: all individuals with true label Y=0,
    subsampled to n_max with a fixed seed. The SAME indices are reused for
    every model so the cross-model comparison is not contaminated by each
    classifier's own decision boundary.
    """
    Y_va   = data["Y_va"].numpy().astype(int)
    y0_idx = np.where(Y_va == 0)[0]
    rng    = np.random.default_rng(rng_seed)
    if len(y0_idx) > n_max:
        eval_idx = np.sort(rng.choice(y0_idx, n_max, replace=False))
        print(f"  Eval set: {len(eval_idx):,} of {len(y0_idx):,} Y=0 "
              f"individuals (fixed subsample, seed={rng_seed})")
    else:
        eval_idx = y0_idx
        print(f"  Eval set: all {len(eval_idx):,} Y=0 individuals")
    return eval_idx


def enumerate_pool(data, eval_idx, scaler,
                   disc_names, cont_names, mediator_specs, vocab, weights):
    """
    Enumerate the candidate pool ONCE (model-independent). The feature
    matrices, costs, and change metadata depend only on (x, z, w) and the
    cost weights — not on any classifier — so a single enumeration is shared
    by every model, guaranteeing an identical candidate pool.

    Returns a dict holding the stacked feature matrices and per-individual
    metadata, including factual_pos (the zero-cost w'=w candidate index).
    """
    X_va, Z_va, Wd_va, Wc_va = (data["X_va"], data["Z_va"],
                                 data["Wd_va"], data["Wc_va"])

    print(f"  Enumerating shared candidate pool for {len(eval_idx)} individuals...")
    t0            = time.time()
    all_feats_xi  = []
    all_feats_xcf = []
    per_ind_meta  = []   # (cost_col, change_data)
    ind_sizes     = []
    sex_labels    = []
    factual_pos   = []   # index of the w'=w (zero-cost) candidate per individual

    for idx in eval_idx:
        xi_val  = float(X_va[idx, 0].item())
        zi_np   = Z_va[idx].numpy().astype(np.float32)
        wdi_np  = Wd_va[idx].numpy()
        wci_raw = Wc_va[idx].numpy()

        if scaler is not None and len(wci_raw) > 0:
            wci_natural = scaler.inverse_transform(wci_raw.reshape(1, -1))[0]
        else:
            wci_natural = wci_raw

        feat_xi, feat_xcf, cost_col, change_data = _build_candidate_arrays(
            xi_val, zi_np, wdi_np, wci_natural,
            disc_names, cont_names, mediator_specs, vocab, weights,
        )
        all_feats_xi.append(feat_xi)
        all_feats_xcf.append(feat_xcf)
        per_ind_meta.append((cost_col, change_data))
        ind_sizes.append(len(cost_col))
        sex_labels.append(int(xi_val))
        # The factual point (w'=w) is the unique zero-cost candidate.
        factual_pos.append(int(np.argmin(cost_col)))

    enum_time   = time.time() - t0
    total_cands = sum(ind_sizes)
    print(f"  Enumeration done in {enum_time:.1f}s  "
          f"(total candidates: {total_cands:,})")

    return {
        "global_xi":    np.vstack(all_feats_xi),
        "global_xcf":   np.vstack(all_feats_xcf),
        "per_ind_meta": per_ind_meta,
        "ind_sizes":    ind_sizes,
        "sex_labels":   sex_labels,
        "factual_pos":  factual_pos,
        "total_cands":  total_cands,
    }


def score_pool(torch_pipe, pool, threshold, nu, slug,
               save_dir, n_max, use_cache=True, device=None):
    """
    Score the shared candidate pool with one model in TWO global PyTorch
    forward passes, compute shortfalls, and derive each individual's
    pre-recourse NIE from the factual (w'=w) candidate.

    Returns (all_candidates, sex_labels, nie_pre) where nie_pre[k] is the
    signed indirect-effect gap for individual k BEFORE any intervention.
    """
    os.makedirs(save_dir, exist_ok=True)
    cache_file = _cache_path(save_dir, slug, n_max, threshold, nu)

    sex_labels = pool["sex_labels"]
    if use_cache and os.path.exists(cache_file):
        print(f"  Loading scored cache: {cache_file}")
        with open(cache_file, "rb") as f:
            obj = pickle.load(f)
        if obj.get("eval_tag") == "y0" and "nie_pre" in obj:
            n_cands = len(obj["all_candidates"][0]) if obj["all_candidates"] else 0
            print(f"  Loaded {len(obj['all_candidates'])} individuals, "
                  f"{n_cands} candidates each")
            return obj["all_candidates"], obj["sex_labels"], obj["nie_pre"]
        print("  Cache stale (old eval set / format) — recomputing.")

    print(f"  Running global PyTorch inference on "
          f"{pool['total_cands']:,} candidates × 2 sexes...")
    t1    = time.time()
    p_xi  = _torch_predict_batched(torch_pipe, pool["global_xi"],  device)
    p_xcf = _torch_predict_batched(torch_pipe, pool["global_xcf"], device)
    print(f"  Inference done in {time.time() - t1:.1f}s")

    soft_thr = threshold - nu
    sf       = np.maximum(0.0, soft_thr - p_xi) + np.maximum(0.0, soft_thr - p_xcf)

    ind_sizes   = pool["ind_sizes"]
    factual_pos = pool["factual_pos"]
    offsets     = np.concatenate([[0], np.cumsum(ind_sizes)])

    all_candidates = []
    nie_pre        = []
    for k, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        c, change_data = pool["per_ind_meta"][k]
        n   = end - start
        sex = sex_labels[k]
        all_candidates.append([
            dict(
                cost=float(c[j]),
                shortfall=float(sf[start + j]),
                p_xi=float(p_xi[start + j]),
                p_xcf=float(p_xcf[start + j]),
                **{key: (int(val[j]) if val.dtype == np.int32 else float(val[j]))
                   for key, val in change_data.items()},
            )
            for j in range(n)
        ])
        # Pre-recourse NIE: signed gap at the factual candidate, same
        # group-orientation convention as nie_post.
        f       = factual_pos[k]
        p1f     = p_xcf[start + f] if sex == 0 else p_xi[start + f]
        p0f     = p_xi[start + f]  if sex == 0 else p_xcf[start + f]
        nie_pre.append(float(p1f - p0f))

    if use_cache:
        with open(cache_file, "wb") as f:
            pickle.dump({"all_candidates": all_candidates,
                         "sex_labels":     sex_labels,
                         "nie_pre":        nie_pre,
                         "eval_tag":       "y0"}, f, protocol=4)
        print(f"  Cache saved → {cache_file}  "
              f"({os.path.getsize(cache_file)/1e6:.1f} MB)")

    return all_candidates, sex_labels, nie_pre


# ── λ sweep ───────────────────────────────────────────────────────────────────

def run_sweep(all_candidates: list, sex_labels: list,
              nie_pre: list, lambda_grid: list) -> list:
    """Re-rank cached candidates at each λ — zero additional model calls.

    nie_pre[k] (the factual w'=w gap) is λ-independent and attached to every
    record so the fractional reduction ΔNIE/NIE_pre can be aggregated.
    """
    results = []
    for eta in lambda_grid:
        per_lambda = []
        for cands, sex, npre in zip(all_candidates, sex_labels, nie_pre):
            best     = dict(select_best(cands, eta))   # shallow copy
            p_g1     = best["p_xcf"] if sex == 0 else best["p_xi"]
            p_g0     = best["p_xi"]  if sex == 0 else best["p_xcf"]
            best["nie_post"] = float(p_g1 - p_g0)
            best["nie_pre"]  = float(npre)
            per_lambda.append(best)
        results.append(per_lambda)
    return results


# ── Aggregation ───────────────────────────────────────────────────────────────

def _ratio_reduction_ci(pre: np.ndarray, post: np.ndarray,
                         n_boot: int, rng) -> tuple:
    """
    Fractional NIE reduction as a population-level ratio of means:

        1 − mean(NIE_post) / mean(NIE_pre)

    (signed; NIE is a property of the (classifier, data) pair). The CI
    resamples individuals JOINTLY so the numerator and denominator stay
    paired. Returns (point, lo, hi); nan if mean(NIE_pre) ≈ 0.
    """
    m = len(pre)
    if m == 0 or abs(pre.mean()) < 1e-12:
        return float("nan"), float("nan"), float("nan")
    point = 1.0 - post.mean() / pre.mean()
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx     = rng.choice(m, m, replace=True)
        denom   = pre[idx].mean()
        boots[b] = (1.0 - post[idx].mean() / denom
                    if abs(denom) > 1e-12 else np.nan)
    lo = float(np.nanpercentile(boots, 2.5))
    hi = float(np.nanpercentile(boots, 97.5))
    return float(point), lo, hi


def _agg(records: list, n_boot: int, rng,
         disc_names: list, cont_names: list, mediator_specs: dict) -> dict:
    if not records:
        nan3 = (float("nan"),) * 3
        base = dict(n=0, feasible_rate=nan3, cost=nan3, shortfall=nan3,
                    nie_post=nan3, nie_pre=nan3, nie_red=nan3)
        for nm in disc_names + cont_names:
            base[nm] = nan3
        return base

    feas   = np.array([float(r["shortfall"] == 0.0) for r in records])
    d_cost = np.array([r["cost"]      for r in records], dtype=float)
    d_sf   = np.array([r["shortfall"] for r in records], dtype=float)
    d_nie  = np.array([r["nie_post"]  for r in records], dtype=float)
    d_pre  = np.array([r["nie_pre"]   for r in records], dtype=float)

    out = {
        "n":             len(records),
        "feasible_rate": _bootstrap_ci(feas,   n_boot, rng=rng),
        "cost":          _bootstrap_ci(d_cost, n_boot, rng=rng),
        "shortfall":     _bootstrap_ci(d_sf,   n_boot, rng=rng),
        "nie_post":      _bootstrap_ci(d_nie,  n_boot, rng=rng),
        "nie_pre":       _bootstrap_ci(d_pre,  n_boot, rng=rng),
        "nie_red":       _ratio_reduction_ci(d_pre, d_nie, n_boot, rng),
    }

    for name in disc_names:
        spec = mediator_specs[name]
        actionable = spec.get("actionable", "free")
        if actionable == "free":
            arr = np.array([r[f"changed_{name}"] for r in records], dtype=float)
        else:
            arr = np.array([r[f"delta_{name}"] for r in records], dtype=float)
        out[name] = _bootstrap_ci(arr, n_boot, rng=rng)

    for name in cont_names:
        arr = np.array([r[f"delta_{name}"] for r in records], dtype=float)
        out[name] = _bootstrap_ci(arr, n_boot, rng=rng)

    return out


def aggregate_sweep(sweep_results: list, sex_labels: list,
                    lambda_grid: list, disc_names: list, cont_names: list,
                    mediator_specs: dict, n_boot: int = 1000) -> dict:
    rng = np.random.default_rng(0)
    sex = np.array(sex_labels)
    out = {"overall": [], "female": [], "male": []}

    for per_ind in sweep_results:
        out["overall"].append(_agg(per_ind, n_boot, rng,
                                   disc_names, cont_names, mediator_specs))
        out["female"].append(_agg(
            [r for r, s in zip(per_ind, sex) if s == 0],
            n_boot, rng, disc_names, cont_names, mediator_specs,
        ))
        out["male"].append(_agg(
            [r for r, s in zip(per_ind, sex) if s == 1],
            n_boot, rng, disc_names, cont_names, mediator_specs,
        ))

    return out


# ── LaTeX table ───────────────────────────────────────────────────────────────

def _ci(mean, lo, hi, fmt=".3f"):
    return rf"${mean:{fmt}}\ [{lo:{fmt}},\ {hi:{fmt}}]$"


def _pct(mean, lo, hi):
    return rf"${100*mean:.1f}\ [{100*lo:.1f},\ {100*hi:.1f}]\%$"


def _eta_label(eta: float) -> str:
    """Absolute η value (the grid is common across models)."""
    if eta == 0:
        return r"$0$"
    if eta >= INF_ETA / 2:
        return r"$\infty$"
    return rf"${eta:.4g}$"


def _is_leb(eta: float, leb: float) -> bool:
    return 0 < eta < INF_ETA / 2 and abs(eta - leb) <= 1e-9 * max(1.0, leb)


def make_sweep_table(agg: dict, lambda_grid: list, leb: float,
                     stratum: str, model_name: str,
                     disc_names: list, cont_names: list, mediator_specs: dict,
                     caption: str, label: str) -> str:
    rows = agg[stratum]

    # Build dynamic mediator column headers
    med_headers = []
    for name in disc_names + cont_names:
        display = mediator_specs[name].get("display", name)
        med_headers.append(display)

    n_med = len(med_headers)
    # 8 + n_med columns: η | n | Feasible | <mediators> | Cost | Shortfall
    #                    | NIE_pre | NIE_post | ΔNIE/NIE_pre
    col_spec = "l" + "c" * (7 + n_med)
    med_header_str = " & ".join(med_headers)

    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        r"  \setlength{\tabcolsep}{4pt}",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
        f"    $\\eta$ & $n$ & Feasible ($S{{=}}0$) "
        f"& {med_header_str} & Cost & Shortfall $S$"
        r" & $\mathrm{NIE}_{\mathrm{pre}}$ & $\mathrm{NIE}_{\mathrm{post}}$"
        r" & $\Delta\mathrm{NIE}/\mathrm{NIE}_{\mathrm{pre}}$ \\",
        r"    \midrule",
    ]
    for eta, row in zip(lambda_grid, rows):
        is_leb = _is_leb(eta, leb)
        lbl    = _eta_label(eta)
        if is_leb:
            lbl += r"$^{\dagger}$"

        med_cells = []
        for name in disc_names:
            spec = mediator_specs[name]
            actionable = spec.get("actionable", "free")
            if actionable == "free":
                med_cells.append(_pct(*row[name]))
            else:
                med_cells.append(_ci(*row[name]))
        for name in cont_names:
            med_cells.append(_ci(*row[name], fmt=".1f"))
        med_str = " & ".join(med_cells)

        lines.append(
            f"    {lbl} & {row['n']} & "
            f"{_pct(*row['feasible_rate'])} & "
            f"{med_str} & "
            f"{_ci(*row['cost'])} & "
            f"{_ci(*row['shortfall'], fmt='.4f')} & "
            f"{_ci(*row['nie_pre'], fmt='.4f')} & "
            f"{_ci(*row['nie_post'], fmt='.4f')} & "
            f"{_ci(*row['nie_red'], fmt='.3f')} \\\\"
        )
        if is_leb:
            lines.append(r"    \midrule")

    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    lines.append(
        rf"% $^{{\dagger}}$ $\eta = \lambda_{{\mathrm{{EB}}}} "
        rf"= {leb:.4f}$ for {model_name}."
    )
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

@hydra.main(config_path="../conf", config_name="config", version_base="1.1")
def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tensors_path = cfg.dataset.paths.tensors
    outcome_dir  = cfg.dataset.paths.outcome_dir
    gaps_dir     = cfg.dataset.paths.gaps_dir
    recourse_dir = cfg.dataset.paths.recourse_dir
    dataset_name = cfg.dataset.name

    disc_names     = list(cfg.dataset.sfm.mediators_disc)
    cont_names     = list(cfg.dataset.sfm.mediators_cont)
    mediator_specs = OmegaConf.to_container(cfg.dataset.recourse.mediators, resolve=True)
    model_specs    = OmegaConf.to_container(cfg.dataset.outcome.models, resolve=True)
    recourse_cfg   = cfg.dataset.recourse
    type_to_slug   = {"logreg": "logreg", "mlp": "mlp"}

    gap_json_path = f"{gaps_dir}/{dataset_name}_gender_gap.json"
    print("\nLoading λ_EB:")
    leb_by_model = load_lambda_eb(gap_json_path)

    data    = torch.load(tensors_path, map_location="cpu", weights_only=False)
    scaler  = data["scaler"]
    vocab   = data["vocab"]
    weights = compute_weights(data, disc_names, cont_names)

    print(f"Cost weights: " + "  ".join(f"α_{n}={w:.5f}" for n, w in weights.items()))
    print(f"Soft constraint: τ={recourse_cfg.threshold}  ν={recourse_cfg.nu}  "
          f"n_max={recourse_cfg.sweep_n_max}")

    os.makedirs(recourse_dir, exist_ok=True)
    txt_lines = []
    n_max_cfg = int(recourse_cfg.sweep_n_max)
    n_max     = n_max_cfg if n_max_cfg > 0 else 10**12

    # ── Common absolute η grid (identical for every model) ──
    # Union of each model's λ_EB-anchored points so η is held fixed across
    # models and every model's λ_EB lands exactly on a grid row.
    anchors = {0.0, float(INF_ETA)}
    for lv in leb_by_model.values():
        for c in (0.25, 0.5, 1.0, 2.0, 4.0):
            anchors.add(c * lv)
    lambda_grid = sorted(anchors)
    print("\nCommon η grid (shared across models): "
          + ", ".join("∞" if v >= INF_ETA / 2 else f"{v:.4g}"
                       for v in lambda_grid))
    print("λ_EB per model: "
          + "  ".join(f"{m}={v:.4f}" for m, v in leb_by_model.items()))

    # ── Shared evaluation set + candidate pool (model-independent) ──
    eval_idx = select_eval_idx(data, n_max, rng_seed=0)
    pool     = enumerate_pool(
        data, eval_idx, scaler,
        disc_names, cont_names, mediator_specs, vocab, weights,
    )

    for spec in model_specs:
        name = spec["name"]
        slug_model = type_to_slug.get(spec["type"], spec["type"])
        pipe = joblib.load(f"{outcome_dir}/{slug_model}.joblib")

        leb = leb_by_model.get(name)
        if leb is None:
            for k, v in leb_by_model.items():
                if k.lower().replace(" ", "") == name.lower().replace(" ", ""):
                    leb = v
                    break
        if leb is None:
            print(f"\nWARNING: no λ_EB for '{name}' — skipping.")
            continue

        slug_file = (name.lower()
                     .replace(" ", "_").replace("(", "").replace(")", "")
                     .replace("-", "").replace(".", ""))

        print(f"\n{'='*60}")
        print(f"Model: {name}   λ_EB = {leb:.4f}")
        print(f"{'='*60}")

        # ── Build PyTorch pipeline ──
        print("Building PyTorch pipeline...")
        torch_pipe = build_torch_pipeline(pipe).to(device)

        # ── Sanity check (a): predict_proba agreement ──
        print("  Sanity check (a) — predict_proba agreement:")
        sanity_check_proba(pipe, torch_pipe, data, scaler,
                           n_check=500, tol=1e-5, device=device)

        # ── Sanity check (b): sweep agreement at λ_EB on 50 individuals ──
        print("  Sanity check (b) — sweep agreement at λ=λ_EB (n=50):")
        sanity_check_sweep(
            pipe, torch_pipe, data, scaler,
            disc_names, cont_names, mediator_specs,
            vocab, weights,
            threshold=float(recourse_cfg.threshold),
            nu=float(recourse_cfg.nu),
            leb=leb, n_check=50, device=device,
        )

        # ── Score the shared pool with this model ──
        print("\nScoring shared candidate pool...")
        t0 = time.time()
        all_candidates, sex_labels, nie_pre = score_pool(
            torch_pipe, pool,
            threshold=float(recourse_cfg.threshold),
            nu=float(recourse_cfg.nu),
            slug=slug_file,
            save_dir=recourse_dir,
            n_max=n_max,
            use_cache=True,
            device=device,
        )
        n_cands = len(all_candidates[0]) if all_candidates else 0
        print(f"  Total wall time: {time.time()-t0:.1f}s  "
              f"({len(all_candidates)} individuals, {n_cands} candidates each)")

        # ── λ sweep over the common grid ──
        print("Running λ sweep (re-ranking cached candidates)...")
        sweep_results = run_sweep(all_candidates, sex_labels, nie_pre, lambda_grid)
        agg = aggregate_sweep(
            sweep_results, sex_labels, lambda_grid,
            disc_names, cont_names, mediator_specs,
            n_boot=int(recourse_cfg.n_boot),
        )

        # ── Print summary ──
        hdr = f"\n{name}  (λ_EB = {leb:.4f}, marked † below)"
        print(hdr)
        txt_lines.append(hdr)
        col_hdr = (f"  {'η':>16s}  {'Feasible':>9}  "
                   + "  ".join(f"{nm:>8}" for nm in disc_names + cont_names)
                   + f"  {'Cost':>7}  {'Shortfall':>10}  "
                   f"{'NIE_pre':>9}  {'NIE_post':>9}  {'dNIE/NIE0':>10}")
        print(col_hdr)
        txt_lines.append(col_hdr)

        for eta, row in zip(lambda_grid, agg["overall"]):
            if eta >= INF_ETA / 2:
                lbl = "inf"
            else:
                lbl = f"{eta:.4g}"
            if _is_leb(eta, leb):
                lbl += "†"
            med_parts = []
            for nm in disc_names:
                med_parts.append(f"{100*row[nm][0]:7.1f}%")
            for nm in cont_names:
                med_parts.append(f"{row[nm][0]:8.2f}")
            med_str = "  ".join(med_parts)
            row_str = (
                f"  {lbl:>16s}  "
                f"{row['feasible_rate'][0]:8.1%}  "
                f"{med_str}  "
                f"{row['cost'][0]:7.4f}  "
                f"{row['shortfall'][0]:10.5f}  "
                f"{row['nie_pre'][0]:+9.4f}  "
                f"{row['nie_post'][0]:+9.4f}  "
                f"{row['nie_red'][0]:+10.3f}"
            )
            print(row_str)
            txt_lines.append(row_str)

        # ── Save LaTeX tables ──
        for stratum, stratum_label in [("overall", "Overall"),
                                       ("female",  "Female"),
                                       ("male",    "Male")]:
            cap = (
                rf"$\lambda$ sensitivity sweep ({stratum_label}, "
                rf"\texttt{{{name}}} outcome model). "
                rf"Evaluation set and candidate pool are held fixed across "
                rf"models (all $Y{{=}}0$ individuals, $n={len(eval_idx)}$); "
                rf"only the classifier varies, so $\mathrm{{NIE}}$ is a "
                rf"property of the (classifier, data) pair. $\eta$ is a "
                rf"common absolute grid; "
                rf"$^{{\dagger}}$ marks $\eta=\lambda_{{\mathrm{{EB}}}}="
                rf"{leb:.3f}=|$NDE$|/|$NIE$|$ for this model. "
                rf"$\Delta\mathrm{{NIE}}/\mathrm{{NIE}}_{{\mathrm{{pre}}}}="
                rf"1-\overline{{\mathrm{{NIE}}_{{\mathrm{{post}}}}}}/"
                rf"\overline{{\mathrm{{NIE}}_{{\mathrm{{pre}}}}}}$. "
                rf"Values: mean with 95\% bootstrap CI."
            )
            lbl = f"tab:sweep_lambda_{slug_file}_{stratum}"
            tex = make_sweep_table(
                agg, lambda_grid, leb, stratum, name,
                disc_names, cont_names, mediator_specs,
                cap, lbl,
            )
            path = f"{recourse_dir}/sweep_lambda_{slug_file}_{stratum}.tex"
            with open(path, "w", encoding="utf-8") as f:
                f.write(tex)
            print(f"  Saved → {path}")

    # ── Text summary ──
    txt_path = f"{recourse_dir}/sweep_lambda_{dataset_name}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines))
    print(f"\nText summary → {txt_path}")


if __name__ == "__main__":
    main()
