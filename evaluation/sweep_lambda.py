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
    return (f"{save_dir}/candidates_{slug}"
            f"_n{n_max}_tau{threshold:.2f}_nu{nu:.2f}.pkl")


def _save_cache(path: str, all_candidates: list, sex_labels: list) -> None:
    with open(path, "wb") as f:
        pickle.dump({"all_candidates": all_candidates,
                     "sex_labels":     sex_labels}, f, protocol=4)
    print(f"  Cache saved → {path}  ({os.path.getsize(path)/1e6:.1f} MB)")


def _load_cache(path: str):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj["all_candidates"], obj["sex_labels"]


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


def precompute_all_tn(torch_pipe, scaler, data,
                      disc_names, cont_names, mediator_specs,
                      vocab, weights, threshold, nu, n_max, slug,
                      save_dir, use_cache=True, device=None, rng_seed=0):
    """
    Build ALL candidate pools in TWO global PyTorch forward passes:
      1. Enumerate candidates per individual (CPU, no model).
      2. Stack into one giant feature matrix → two PyTorch calls (xi, xi_cf).
      3. Compute shortfalls, split back per individual, cache.
    """
    os.makedirs(save_dir, exist_ok=True)
    cache_file = _cache_path(save_dir, slug, n_max, threshold, nu)

    if use_cache and os.path.exists(cache_file):
        print(f"  Loading candidate cache: {cache_file}")
        all_candidates, sex_labels = _load_cache(cache_file)
        sample = all_candidates[0][0] if all_candidates else {}
        if "p_xi" not in sample:
            print("  Cache missing p_xi/p_xcf (old format) — recomputing.")
        else:
            n_cands = len(all_candidates[0]) if all_candidates else 0
            print(f"  Loaded {len(all_candidates)} individuals, "
                  f"{n_cands} candidates each")
            return all_candidates, sex_labels

    # Identify true negatives
    X_va  = data["X_va"]
    Z_va  = data["Z_va"]
    Wd_va = data["Wd_va"]
    Wc_va = data["Wc_va"]
    Y_va  = data["Y_va"].numpy().astype(int)

    feat_all = stack_sfm_features(X_va, Z_va, Wd_va, Wc_va, scaler)
    with torch.no_grad():
        p_all = _torch_predict_batched(
            torch_pipe, feat_all.astype(np.float32), device,
        )
    yhat_all = (p_all >= 0.5).astype(int)
    tn_mask  = (Y_va == 0) & (yhat_all == 0)
    tn_idx   = np.where(tn_mask)[0]
    print(f"  True negatives: {len(tn_idx):,} / {len(Y_va):,}")

    rng = np.random.default_rng(rng_seed)
    if len(tn_idx) > n_max:
        tn_idx = rng.choice(tn_idx, n_max, replace=False)
        print(f"  Subsampled to {n_max:,}")

    # Phase 1: enumerate candidates (no model calls)
    print(f"  Enumerating candidates for {len(tn_idx)} individuals...")
    t0            = time.time()
    all_feats_xi  = []
    all_feats_xcf = []
    per_ind_meta  = []   # (cost_col, change_data)
    ind_sizes     = []
    sex_labels    = []

    for idx in tn_idx:
        xi_val   = float(X_va[idx, 0].item())
        zi_np    = Z_va[idx].numpy().astype(np.float32)
        wdi_np   = Wd_va[idx].numpy()
        wci_raw  = Wc_va[idx].numpy()

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

    enum_time   = time.time() - t0
    total_cands = sum(ind_sizes)
    print(f"  Enumeration done in {enum_time:.1f}s  "
          f"(total candidates: {total_cands:,})")

    # Phase 2: TWO global PyTorch forward passes
    print(f"  Running global PyTorch inference on {total_cands:,} candidates × 2 sexes...")
    t1         = time.time()
    global_xi  = np.vstack(all_feats_xi)   # (total_cands, n_feat)
    global_xcf = np.vstack(all_feats_xcf)  # (total_cands, n_feat)

    p_xi  = _torch_predict_batched(torch_pipe, global_xi,  device)
    p_xcf = _torch_predict_batched(torch_pipe, global_xcf, device)
    infer_time = time.time() - t1
    print(f"  Inference done in {infer_time:.1f}s")

    # Phase 3: compute shortfalls + split per individual
    soft_thr = threshold - nu
    sf       = np.maximum(0.0, soft_thr - p_xi) + np.maximum(0.0, soft_thr - p_xcf)

    offsets        = np.concatenate([[0], np.cumsum(ind_sizes)])
    all_candidates = []
    for k, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        c, change_data = per_ind_meta[k]
        n = end - start
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

    if use_cache:
        _save_cache(cache_file, all_candidates, sex_labels)

    return all_candidates, sex_labels


# ── λ sweep ───────────────────────────────────────────────────────────────────

def run_sweep(all_candidates: list, sex_labels: list,
              lambda_grid: list) -> list:
    """Re-rank cached candidates at each λ — zero additional model calls."""
    results = []
    for eta in lambda_grid:
        per_lambda = []
        for cands, sex in zip(all_candidates, sex_labels):
            best     = dict(select_best(cands, eta))   # shallow copy
            p_g1     = best["p_xcf"] if sex == 0 else best["p_xi"]
            p_g0     = best["p_xi"]  if sex == 0 else best["p_xcf"]
            best["nie_post"] = float(p_g1 - p_g0)
            per_lambda.append(best)
        results.append(per_lambda)
    return results


# ── Aggregation ───────────────────────────────────────────────────────────────

def _agg(records: list, n_boot: int, rng,
         disc_names: list, cont_names: list, mediator_specs: dict) -> dict:
    if not records:
        nan3 = (float("nan"),) * 3
        base = dict(n=0, feasible_rate=nan3, cost=nan3, shortfall=nan3, nie_post=nan3)
        for nm in disc_names + cont_names:
            base[nm] = nan3
        return base

    feas   = np.array([float(r["shortfall"] == 0.0) for r in records])
    d_cost = np.array([r["cost"]      for r in records], dtype=float)
    d_sf   = np.array([r["shortfall"] for r in records], dtype=float)
    d_nie  = np.array([r["nie_post"]  for r in records], dtype=float)

    out = {
        "n":             len(records),
        "feasible_rate": _bootstrap_ci(feas,   n_boot, rng=rng),
        "cost":          _bootstrap_ci(d_cost, n_boot, rng=rng),
        "shortfall":     _bootstrap_ci(d_sf,   n_boot, rng=rng),
        "nie_post":      _bootstrap_ci(d_nie,  n_boot, rng=rng),
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


def _lambda_label(eta: float, leb: float) -> str:
    if eta == 0:
        return "$0$"
    if eta >= INF_ETA / 2:
        return r"$\infty$"
    ratios  = {0.25: r"$\lambda_{\mathrm{EB}}/4$",
               0.5:  r"$\lambda_{\mathrm{EB}}/2$",
               1.0:  r"$\lambda_{\mathrm{EB}}$",
               2.0:  r"$2\lambda_{\mathrm{EB}}$",
               4.0:  r"$4\lambda_{\mathrm{EB}}$"}
    nearest = min(ratios, key=lambda r: abs(r - eta / leb))
    return ratios[nearest]


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
    col_spec = "l" + "c" * (3 + n_med)
    med_header_str = " & ".join(med_headers)

    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        r"  \setlength{\tabcolsep}{4pt}",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
        f"    $\\lambda$ & $n_{{\\mathrm{{TN}}}}$ & Feasible ($S{{=}}0$) "
        f"& {med_header_str} & Cost & Shortfall $S$"
        r" & $\mathrm{NIE}_{\mathrm{post}}$ \\",
        r"    \midrule",
    ]
    for eta, row in zip(lambda_grid, rows):
        lbl = _lambda_label(eta, leb)

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
            f"{_ci(*row['nie_post'], fmt='.4f')} \\\\"
        )
        if eta > 0 and eta < INF_ETA / 2 and abs(eta / leb - 1.0) < 0.01:
            lines.append(r"    \midrule")

    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
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
    n_max = int(recourse_cfg.sweep_n_max)

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

        slug_file   = (name.lower()
                       .replace(" ", "_").replace("(", "").replace(")", "")
                       .replace("-", "").replace(".", ""))
        lambda_grid = [0.0, leb/4, leb/2, leb, 2*leb, 4*leb, INF_ETA]

        print(f"\n{'='*60}")
        print(f"Model: {name}   λ_EB = {leb:.4f}")
        print(f"λ grid: {[f'{v:.4f}' if v < INF_ETA else '∞' for v in lambda_grid]}")
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

        # ── Precompute / load candidate pools ──
        print("\nPrecomputing / loading candidate pools...")
        t0 = time.time()
        all_candidates, sex_labels = precompute_all_tn(
            torch_pipe, scaler, data,
            disc_names, cont_names, mediator_specs,
            vocab, weights,
            threshold=float(recourse_cfg.threshold),
            nu=float(recourse_cfg.nu),
            n_max=n_max,
            slug=slug_file,
            save_dir=recourse_dir,
            use_cache=True,
            device=device,
        )
        n_cands = len(all_candidates[0]) if all_candidates else 0
        print(f"  Total wall time: {time.time()-t0:.1f}s  "
              f"({len(all_candidates)} individuals, {n_cands} candidates each)")

        # ── λ sweep ──
        print("Running λ sweep (re-ranking cached candidates)...")
        sweep_results = run_sweep(all_candidates, sex_labels, lambda_grid)
        agg = aggregate_sweep(
            sweep_results, sex_labels, lambda_grid,
            disc_names, cont_names, mediator_specs,
            n_boot=int(recourse_cfg.n_boot),
        )

        # ── Print summary ──
        hdr = f"\n{name}  (λ_EB = {leb:.4f})"
        print(hdr)
        txt_lines.append(hdr)
        col_hdr = (f"  {'λ':>16s}  {'Feasible':>9}  "
                   + "  ".join(f"{nm:>8}" for nm in disc_names + cont_names)
                   + f"  {'Cost':>7}  {'Shortfall':>10}  {'NIE_post':>9}")
        print(col_hdr)
        txt_lines.append(col_hdr)

        for eta, row in zip(lambda_grid, agg["overall"]):
            lbl = "∞" if eta >= INF_ETA / 2 else f"{eta:.4f}"
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
                f"{row['nie_post'][0]:+9.4f}"
            )
            print(row_str)
            txt_lines.append(row_str)

        # ── Save LaTeX tables ──
        for stratum, stratum_label in [("overall", "Overall"),
                                       ("female",  "Female"),
                                       ("male",    "Male")]:
            cap = (
                rf"$\lambda$ sensitivity sweep ({stratum_label} true negatives, "
                rf"\texttt{{{name}}} outcome model). "
                rf"$\lambda_{{\mathrm{{EB}}}} = {leb:.3f}$ "
                rf"$= |$NDE$|/|$NIE$|$. "
                rf"$n = {n_max}$ TN individuals (subsample). "
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
