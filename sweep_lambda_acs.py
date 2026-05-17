"""
sweep_lambda_acs.py — λ sensitivity sweep with PyTorch-accelerated inference.

The recourse objective is

    min_{w'} cost(w, w') + λ · S(w')

    S(w') = max(0, τ−ν − p(x_i,   w', z))
           + max(0, τ−ν − p(1−x_i, w', z))

The empirical calibration λ_EB = |NDE| / |NIE| is loaded from the gap JSON.

Performance design
------------------
  1. sklearn weights are extracted into frozen PyTorch nn.Linear / nn.Sequential
     (verified against sklearn predict_proba to ≤ 1e-5 before the sweep).
  2. For each model, ALL candidate pools across ALL TN individuals are
     evaluated in TWO global PyTorch forward passes (one per sex value),
     replacing O(n_individuals) separate sklearn calls.
  3. The candidate pool (costs + shortfalls) is cached to disk; the λ sweep
     itself is a sequence of per-individual argmin operations — zero additional
     model evaluations.
  4. Run on --n-max 250 (default) for the sweep figure.  Use
     compute_recourse_acs.py (sklearn, full validation set) for the main table.

Sanity checks
-------------
  Before the sweep, the script verifies:
    (a) |p_torch − p_sklearn|_max ≤ 1e-5 on 500 validation rows.
    (b) At λ = λ_EB and ν = 0.05, the best candidate selected by PyTorch
        matches the sklearn version on a 50-individual subsample
        (Δcost ≤ 1e-4, same discrete choices).

Requires
--------
  outputs/gaps/acs_gender_gap.json
  outputs/data/acs_tensors.pt
  outputs/outcome/acs/logreg.joblib
  outputs/outcome/acs/mlp.joblib

Outputs
-------
  outputs/recourse/candidates_{slug}_n{N}_tau{t}_nu{v}.pkl  (candidate cache)
  outputs/recourse/sweep_lambda_{slug}_{stratum}.tex
  outputs/recourse/sweep_lambda_acs.txt

Usage
-----
  python sweep_lambda_acs.py
  python sweep_lambda_acs.py --n-max 300 --nu 0.05
  python sweep_lambda_acs.py --no-cache      # recompute candidate pools
"""

import argparse
import json
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib

from train_outcome_acs import build_features
from compute_recourse_acs import (
    WKHP_CLIP, WKHP_STEP,
    compute_weights,
    precompute_candidates_batch,   # sklearn — used only for sanity check
    select_best,
    _bootstrap_ci,
)

SAVE_DIR = "outputs/recourse"
INF_ETA  = 1e9          # proxy for λ = ∞
_BATCH   = 500_000      # max rows per PyTorch forward pass (memory guard)


# ── PyTorch pipeline: sklearn weights → frozen nn.Module ──────────────────────

class _ACSPreprocess(nn.Module):
    """
    Replicates the ACS ColumnTransformer in PyTorch.

    Input  feat: (N, 6)  — [SEX, AGEP, POBP_US, SCHL_GRP, OCCP_GRP, WKHP_h]
    Output       (N, 2 + 2 + n_schl + n_occp)

    Output column order matches sklearn's ColumnTransformer (transformers in
    declaration order):
        pass  → [SEX, POBP_US]
        scale → [AGEP_scaled, WKHP_scaled]
        ohe_schl → one-hot SCHL_GRP  (n_schl cols)
        ohe_occp → one-hot OCCP_GRP  (n_occp cols)
    """

    def __init__(self, prep, n_schl: int, n_occp: int):
        super().__init__()
        scale_tr = prep.named_transformers_["scale"]
        # scale_tr was fitted on [AGEP (col 1), WKHP (col 5)] in that order
        self.register_buffer("mean_",  torch.tensor(scale_tr.mean_,  dtype=torch.float32))
        self.register_buffer("scale_", torch.tensor(scale_tr.scale_, dtype=torch.float32))
        self.n_schl = n_schl
        self.n_occp = n_occp

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        sex  = feat[:, 0:1]
        pobp = feat[:, 2:3]
        agep = (feat[:, 1:2] - self.mean_[0]) / self.scale_[0]
        wkhp = (feat[:, 5:6] - self.mean_[1]) / self.scale_[1]
        schl = F.one_hot(feat[:, 3].long(), self.n_schl).float()
        occp = F.one_hot(feat[:, 4].long(), self.n_occp).float()
        return torch.cat([sex, pobp, agep, wkhp, schl, occp], dim=1)


class TorchLogReg(nn.Module):
    """LogisticRegression pipeline: frozen preprocessing + frozen linear."""

    def __init__(self, prep, clf, n_schl: int, n_occp: int):
        super().__init__()
        self.preprocess = _ACSPreprocess(prep, n_schl, n_occp)
        n_in = 2 + 2 + n_schl + n_occp
        self.linear = nn.Linear(n_in, 1)
        # clf.coef_: (1, n_in),  clf.intercept_: (1,)
        self.linear.weight.data = torch.tensor(clf.coef_,       dtype=torch.float32)
        self.linear.bias.data   = torch.tensor(clf.intercept_,  dtype=torch.float32)
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(self.preprocess(feat))).squeeze(1)


class TorchMLP(nn.Module):
    """MLPClassifier pipeline: frozen preprocessing + frozen MLP."""

    def __init__(self, prep, clf, n_schl: int, n_occp: int):
        super().__init__()
        self.preprocess = _ACSPreprocess(prep, n_schl, n_occp)
        n_in = 2 + 2 + n_schl + n_occp

        layers = []
        in_dim = n_in
        # clf.coefs_[i]: (in, out) — sklearn layout; nn.Linear wants (out, in) for weight
        for i, (W, b) in enumerate(zip(clf.coefs_, clf.intercepts_)):
            out_dim = W.shape[1]
            lin     = nn.Linear(in_dim, out_dim)
            lin.weight.data = torch.tensor(W.T, dtype=torch.float32)
            lin.bias.data   = torch.tensor(b,   dtype=torch.float32)
            for p in lin.parameters():
                p.requires_grad_(False)
            layers.append(lin)
            if i < len(clf.coefs_) - 1:
                layers.append(nn.ReLU())
            in_dim = out_dim

        self.net = nn.Sequential(*layers)
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # sklearn MLPClassifier binary: output activation = logistic (sigmoid)
        return torch.sigmoid(self.net(self.preprocess(feat))).squeeze(1)


def build_torch_pipeline(pipe, vocab: dict) -> nn.Module:
    """
    Convert a fitted sklearn Pipeline (prep + clf) into a frozen PyTorch Module.
    Supports LogisticRegression and MLPClassifier.
    """
    n_schl = vocab["SCHL_GRP"]
    n_occp = vocab["OCCP_GRP"]
    prep   = pipe.named_steps["prep"]
    clf    = pipe.named_steps["clf"]
    if hasattr(clf, "coefs_"):
        return TorchMLP(prep, clf, n_schl, n_occp)
    return TorchLogReg(prep, clf, n_schl, n_occp)


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

    feat   = build_features(X, Z, Wd, Wc, scaler)       # (n_check, 6) numpy
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


def sanity_check_sweep(pipe, torch_pipe, data, scaler, vocab, weights,
                       threshold: float, nu: float, leb: float,
                       n_check: int = 50, device=None) -> None:
    """
    On n_check true negatives, verify that the best candidate selected by
    PyTorch at λ = λ_EB matches sklearn to within cost ≤ 1e-4 and the same
    discrete choices (Δedu, occ_change).
    """
    X_va  = data["X_va"]
    Z_va  = data["Z_va"]
    Wd_va = data["Wd_va"]
    Wc_va = data["Wc_va"]
    Y_va  = data["Y_va"].numpy().astype(int)

    feat_all = build_features(X_va, Z_va, Wd_va, Wc_va, scaler)
    yhat_all = pipe.predict(feat_all)
    tn_idx   = np.where((Y_va == 0) & (yhat_all == 0))[0][:n_check]

    cost_diffs, disc_match = [], []
    torch_pipe.eval()

    for i in tn_idx:
        # sklearn candidates
        cands_sk = precompute_candidates_batch(
            pipe, scaler,
            X_va[i].unsqueeze(0), Z_va[i].unsqueeze(0),
            Wd_va[i].unsqueeze(0), Wc_va[i].unsqueeze(0),
            vocab, weights, threshold, nu,
        )
        best_sk = select_best(cands_sk, leb)

        # PyTorch candidates (global batch of size 1 individual)
        cands_pt = _precompute_one_torch(
            torch_pipe, scaler,
            X_va[i].unsqueeze(0), Z_va[i].unsqueeze(0),
            Wd_va[i].unsqueeze(0), Wc_va[i].unsqueeze(0),
            vocab, weights, threshold, nu, device,
        )
        best_pt = select_best(cands_pt, leb)

        cost_diffs.append(abs(best_sk["cost"] - best_pt["cost"]))
        disc_match.append(
            best_sk["delta_schl"] == best_pt["delta_schl"]
            and best_sk["occ_change"] == best_pt["occ_change"]
        )

    match_rate   = sum(disc_match) / len(disc_match)
    max_cost_diff = max(cost_diffs)
    print(f"    Sweep agreement (n={len(tn_idx)}, λ=λ_EB):  "
          f"discrete match={match_rate:.1%}  "
          f"max|Δcost|={max_cost_diff:.2e}")
    if match_rate < 0.95 or max_cost_diff > 1e-3:
        print("    WARNING: sweep results differ more than expected.")


# ── λ_EB from gap JSON ────────────────────────────────────────────────────────

def load_lambda_eb(gap_json: str) -> dict[str, float]:
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

def _cache_path(slug: str, n_max: int, threshold: float, nu: float) -> str:
    return (f"{SAVE_DIR}/candidates_{slug}"
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

def _build_candidate_arrays(scaler, xi_val, zi_vals, wdi_vals, wci_vals,
                             vocab, weights, threshold, nu):
    """
    Enumerate (SCHL', OCCP', WKHP') candidates for ONE individual and return
    raw numpy arrays (no model calls).

    Returns
    -------
    feat_xi  : (n, 6)  raw feature matrix with actual sex
    feat_xcf : (n, 6)  same but with counterfactual sex
    cost_col : (n,)    total cost for each candidate
    ds_col   : (n,)    delta_schl
    oc_col   : (n,)    occ_change
    dw_col   : (n,)    delta_wkhp
    """
    n_schl     = vocab["SCHL_GRP"]
    n_occp     = vocab["OCCP_GRP"]
    schl_cur   = int(wdi_vals[0])
    occp_cur   = int(wdi_vals[1])
    wkhp_cur_h = float(np.clip(scaler.inverse_transform(
                     wci_vals.reshape(1, 1))[0, 0], *WKHP_CLIP))
    agep       = float(zi_vals[0])
    pobp       = float(zi_vals[1])
    xi_cf_val  = 1.0 - xi_val

    alpha_wkhp = weights["WKHP"]
    alpha_schl = weights["SCHL_GRP"]
    alpha_occp = weights["OCCP_GRP"]

    wkhp_h_grid = np.clip(
        np.arange(wkhp_cur_h, WKHP_CLIP[1] + WKHP_STEP, WKHP_STEP),
        *WKHP_CLIP,
    ).astype(np.float32)
    dw_grid = (wkhp_h_grid - wkhp_cur_h).astype(np.float32)
    n_wkhp  = len(wkhp_h_grid)

    schl_range = range(schl_cur, n_schl)
    n          = len(schl_range) * n_occp * n_wkhp

    schl_col = np.empty(n, dtype=np.float32)
    occp_col = np.empty(n, dtype=np.float32)
    wkhp_col = np.empty(n, dtype=np.float32)
    ds_col   = np.empty(n, dtype=np.int32)
    oc_col   = np.empty(n, dtype=np.int32)
    dw_col   = np.empty(n, dtype=np.float32)
    cost_col = np.empty(n, dtype=np.float32)

    k = 0
    for schl in schl_range:
        ds          = schl - schl_cur
        schl_term   = alpha_schl * int(ds > 0)
        for occp in range(n_occp):
            oc         = int(occp != occp_cur)
            base_cost  = schl_term + alpha_occp * oc
            end        = k + n_wkhp
            schl_col[k:end] = float(schl)
            occp_col[k:end] = float(occp)
            wkhp_col[k:end] = wkhp_h_grid
            ds_col[k:end]   = ds
            oc_col[k:end]   = oc
            dw_col[k:end]   = dw_grid
            cost_col[k:end] = base_cost + alpha_wkhp * dw_grid * dw_grid
            k = end

    feat_xi         = np.empty((n, 6), dtype=np.float32)
    feat_xi[:, 0]   = xi_val
    feat_xi[:, 1]   = agep
    feat_xi[:, 2]   = pobp
    feat_xi[:, 3]   = schl_col
    feat_xi[:, 4]   = occp_col
    feat_xi[:, 5]   = wkhp_col

    feat_xcf        = feat_xi.copy()
    feat_xcf[:, 0]  = xi_cf_val

    return feat_xi, feat_xcf, cost_col, ds_col, oc_col, dw_col


def _precompute_one_torch(torch_pipe, scaler, xi, zi, wdi, wci,
                          vocab, weights, threshold, nu, device):
    """Single-individual PyTorch candidate pool (used by sanity check)."""
    xi_val  = float(xi[0, 0].item())
    zi_vals = zi[0].numpy()
    wdi_vals = wdi[0].numpy()
    wci_vals = wci[0].numpy()

    feat_xi, feat_xcf, cost_col, ds_col, oc_col, dw_col = _build_candidate_arrays(
        scaler, xi_val, zi_vals, wdi_vals, wci_vals,
        vocab, weights, threshold, nu,
    )
    n        = len(cost_col)
    soft_thr = threshold - nu

    with torch.no_grad():
        t_xi  = torch.tensor(feat_xi,  dtype=torch.float32, device=device)
        t_xcf = torch.tensor(feat_xcf, dtype=torch.float32, device=device)
        p_xi  = torch_pipe(t_xi).cpu().numpy()
        p_xcf = torch_pipe(t_xcf).cpu().numpy()

    sf = np.maximum(0.0, soft_thr - p_xi) + np.maximum(0.0, soft_thr - p_xcf)

    return [
        {"cost": float(cost_col[i]), "shortfall": float(sf[i]),
         "delta_schl": int(ds_col[i]), "occ_change": int(oc_col[i]),
         "delta_wkhp": float(dw_col[i])}
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


def precompute_all_tn(torch_pipe, scaler, data, vocab, weights,
                      threshold, nu, n_max, slug,
                      use_cache=True, device=None, rng_seed=0):
    """
    Build ALL candidate pools in TWO global PyTorch forward passes:
      1. Enumerate candidates per individual (CPU, no model).
      2. Stack into one giant feature matrix → two PyTorch calls (xi, xi_cf).
      3. Compute shortfalls, split back per individual, cache.
    """
    os.makedirs(SAVE_DIR, exist_ok=True)
    cache_file = _cache_path(slug, n_max, threshold, nu)

    if use_cache and os.path.exists(cache_file):
        print(f"  Loading candidate cache: {cache_file}")
        all_candidates, sex_labels = _load_cache(cache_file)
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

    feat_all = build_features(X_va, Z_va, Wd_va, Wc_va, scaler)
    # Use sklearn pipe stored on the torch_pipe to get predictions for TN identification
    # We need a sklearn pipe — pass it separately or use torch for TN identification too
    # Simple: use torch pipeline for TN identification
    with torch.no_grad():
        p_all = _torch_predict_batched(
            torch_pipe,
            feat_all.astype(np.float32),
            device,
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
    t0           = time.time()
    all_feats_xi  = []
    all_feats_xcf = []
    per_ind_meta  = []   # (cost_col, ds_col, oc_col, dw_col)
    ind_sizes     = []
    sex_labels    = []

    for idx in tn_idx:
        xi_val   = float(X_va[idx, 0].item())
        zi_vals  = Z_va[idx].numpy()
        wdi_vals = Wd_va[idx].numpy()
        wci_vals = Wc_va[idx].numpy()

        feat_xi, feat_xcf, cost_col, ds_col, oc_col, dw_col = _build_candidate_arrays(
            scaler, xi_val, zi_vals, wdi_vals, wci_vals,
            vocab, weights, threshold, nu,
        )
        all_feats_xi.append(feat_xi)
        all_feats_xcf.append(feat_xcf)
        per_ind_meta.append((cost_col, ds_col, oc_col, dw_col))
        ind_sizes.append(len(cost_col))
        sex_labels.append(int(xi_val))

    enum_time = time.time() - t0
    total_cands = sum(ind_sizes)
    print(f"  Enumeration done in {enum_time:.1f}s  "
          f"(total candidates: {total_cands:,})")

    # Phase 2: TWO global PyTorch forward passes
    print(f"  Running global PyTorch inference on {total_cands:,} candidates × 2 sexes...")
    t1          = time.time()
    global_xi   = np.vstack(all_feats_xi)   # (total_cands, 6)
    global_xcf  = np.vstack(all_feats_xcf)  # (total_cands, 6)

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
        c, ds, oc, dw = per_ind_meta[k]
        n = end - start
        all_candidates.append([
            {
                "cost":       float(c[j]),
                "shortfall":  float(sf[start + j]),
                "delta_schl": int(ds[j]),
                "occ_change": int(oc[j]),
                "delta_wkhp": float(dw[j]),
            }
            for j in range(n)
        ])

    if use_cache:
        _save_cache(cache_file, all_candidates, sex_labels)

    return all_candidates, sex_labels


# ── λ sweep ───────────────────────────────────────────────────────────────────

def run_sweep(all_candidates: list, sex_labels: list,
              lambda_grid: list[float]) -> list[list[dict]]:
    """Re-rank cached candidates at each λ — zero additional model calls."""
    return [
        [select_best(cands, eta) for cands in all_candidates]
        for eta in lambda_grid
    ]


# ── Aggregation ───────────────────────────────────────────────────────────────

def _agg(records: list[dict], n_boot: int, rng) -> dict:
    if not records:
        nan3 = (float("nan"),) * 3
        return dict(n=0, feasible_rate=nan3, delta_schl=nan3,
                    occ_change=nan3, delta_wkhp=nan3, cost=nan3, shortfall=nan3)

    feas   = np.array([float(r["shortfall"] == 0.0) for r in records])
    d_schl = np.array([r["delta_schl"] for r in records], dtype=float)
    d_occ  = np.array([r["occ_change"] for r in records], dtype=float)
    d_wkhp = np.array([r["delta_wkhp"] for r in records], dtype=float)
    d_cost = np.array([r["cost"]       for r in records], dtype=float)
    d_sf   = np.array([r["shortfall"]  for r in records], dtype=float)

    return {
        "n":             len(records),
        "feasible_rate": _bootstrap_ci(feas,   n_boot, rng=rng),
        "delta_schl":    _bootstrap_ci(d_schl, n_boot, rng=rng),
        "occ_change":    _bootstrap_ci(d_occ,  n_boot, rng=rng),
        "delta_wkhp":    _bootstrap_ci(d_wkhp, n_boot, rng=rng),
        "cost":          _bootstrap_ci(d_cost, n_boot, rng=rng),
        "shortfall":     _bootstrap_ci(d_sf,   n_boot, rng=rng),
    }


def aggregate_sweep(sweep_results: list[list[dict]],
                    sex_labels: list[int],
                    lambda_grid: list[float],
                    n_boot: int = 1000) -> dict:
    rng = np.random.default_rng(0)
    sex = np.array(sex_labels)
    out = {"overall": [], "female": [], "male": []}

    for per_ind in sweep_results:
        out["overall"].append(_agg(per_ind, n_boot, rng))
        out["female"].append(_agg([r for r, s in zip(per_ind, sex) if s == 0],
                                  n_boot, rng))
        out["male"].append(_agg([r for r, s in zip(per_ind, sex) if s == 1],
                                 n_boot, rng))

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


def make_sweep_table(agg: dict, lambda_grid: list[float], leb: float,
                     stratum: str, model_name: str,
                     caption: str, label: str) -> str:
    rows  = agg[stratum]
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{lccccccc}",
        r"    \toprule",
        r"    $\lambda$ & $n_{\mathrm{TN}}$ & Feasible ($S{=}0$) "
        r"& $\Delta\mathrm{Edu}$ & Occ.\ chg & $\Delta\mathrm{WKHP}$ & Cost & Shortfall $S$ \\",
        r"    \midrule",
    ]
    for eta, row in zip(lambda_grid, rows):
        lbl = _lambda_label(eta, leb)
        lines.append(
            f"    {lbl} & {row['n']} & "
            f"{_pct(*row['feasible_rate'])} & "
            f"{_ci(*row['delta_schl'])} & "
            f"{_pct(*row['occ_change'])} & "
            f"{_ci(*row['delta_wkhp'], fmt='.1f')} & "
            f"{_ci(*row['cost'])} & "
            f"{_ci(*row['shortfall'], fmt='.4f')} \\\\"
        )
        if eta > 0 and eta < INF_ETA / 2 and abs(eta / leb - 1.0) < 0.01:
            lines.append(r"    \midrule")

    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\nLoading λ_EB:")
    leb_by_model = load_lambda_eb(args.gap_json)

    data    = torch.load(args.tensors, map_location="cpu", weights_only=False)
    scaler  = data["scaler"]
    vocab   = data["vocab"]
    weights = compute_weights(data)
    print(f"Cost weights: α_WKHP={weights['WKHP']:.5f}  "
          f"α_SCHL={weights['SCHL_GRP']:.5f}  α_OCCP={weights['OCCP_GRP']:.5f}")
    print(f"Soft constraint: τ={args.threshold}  ν={args.nu}  n_max={args.n_max}")

    pipe_lr  = joblib.load(args.logreg)
    pipe_mlp = joblib.load(args.mlp)
    models   = [(pipe_lr, "Logistic Reg."), (pipe_mlp, "MLP (64--32)")]

    os.makedirs(SAVE_DIR, exist_ok=True)
    txt_lines = []

    for pipe, name in models:
        leb = leb_by_model.get(name)
        if leb is None:
            for k, v in leb_by_model.items():
                if k.lower().replace(" ", "") == name.lower().replace(" ", ""):
                    leb = v
                    break
        if leb is None:
            print(f"\nWARNING: no λ_EB for '{name}' — skipping.")
            continue

        slug        = (name.lower()
                       .replace(" ", "_").replace("(", "").replace(")", "")
                       .replace("-", "").replace(".", ""))
        lambda_grid = [0.0, leb/4, leb/2, leb, 2*leb, 4*leb, INF_ETA]

        print(f"\n{'='*60}")
        print(f"Model: {name}   λ_EB = {leb:.4f}")
        print(f"λ grid: {[f'{v:.4f}' if v < INF_ETA else '∞' for v in lambda_grid]}")
        print(f"{'='*60}")

        # ── Build PyTorch pipeline ──
        print("Building PyTorch pipeline...")
        torch_pipe = build_torch_pipeline(pipe, vocab).to(device)

        # ── Sanity check (a): predict_proba agreement ──
        print("  Sanity check (a) — predict_proba agreement:")
        sanity_check_proba(pipe, torch_pipe, data, scaler,
                           n_check=500, tol=1e-5, device=device)

        # ── Sanity check (b): sweep agreement at λ_EB on 50 individuals ──
        print("  Sanity check (b) — sweep agreement at λ=λ_EB (n=50):")
        sanity_check_sweep(pipe, torch_pipe, data, scaler, vocab, weights,
                           threshold=args.threshold, nu=args.nu,
                           leb=leb, n_check=50, device=device)

        # ── Precompute / load candidate pools ──
        print("\nPrecomputing / loading candidate pools...")
        t0 = time.time()
        all_candidates, sex_labels = precompute_all_tn(
            torch_pipe, scaler, data, vocab, weights,
            threshold=args.threshold,
            nu=args.nu,
            n_max=args.n_max,
            slug=slug,
            use_cache=not args.no_cache,
            device=device,
        )
        n_cands = len(all_candidates[0]) if all_candidates else 0
        print(f"  Total wall time: {time.time()-t0:.1f}s  "
              f"({len(all_candidates)} individuals, {n_cands} candidates each)")

        # ── λ sweep (cheap: re-rank stored candidates) ──
        print("Running λ sweep (re-ranking cached candidates)...")
        sweep_results = run_sweep(all_candidates, sex_labels, lambda_grid)
        agg           = aggregate_sweep(sweep_results, sex_labels, lambda_grid,
                                        n_boot=args.n_boot)

        # ── Print summary ──
        hdr = f"\n{name}  (λ_EB = {leb:.4f})"
        print(hdr)
        txt_lines.append(hdr)
        col_hdr = (f"  {'λ':>16s}  {'Feasible':>9}  {'ΔEdu':>6}  "
                   f"{'Occ%':>6}  {'ΔWKHP':>7}  {'Cost':>7}  {'Shortfall':>10}")
        print(col_hdr)
        txt_lines.append(col_hdr)

        for eta, row in zip(lambda_grid, agg["overall"]):
            lbl = "∞" if eta >= INF_ETA / 2 else f"{eta:.4f}"
            row_str = (
                f"  {lbl:>16s}  "
                f"{row['feasible_rate'][0]:8.1%}  "
                f"{row['delta_schl'][0]:6.2f}  "
                f"{100*row['occ_change'][0]:5.1f}%  "
                f"{row['delta_wkhp'][0]:7.2f}  "
                f"{row['cost'][0]:7.4f}  "
                f"{row['shortfall'][0]:10.5f}"
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
                rf"$n = {args.n_max}$ TN individuals (subsample). "
                rf"Feasible: $S=0$ at the optimal solution. "
                rf"Values: mean with 95\% bootstrap CI."
            )
            lbl = f"tab:sweep_lambda_{slug}_{stratum}"
            tex = make_sweep_table(agg, lambda_grid, leb, stratum,
                                   name, cap, lbl)
            path = f"{SAVE_DIR}/sweep_lambda_{slug}_{stratum}.tex"
            with open(path, "w", encoding="utf-8") as f:
                f.write(tex)
            print(f"  Saved → {path}")

    # ── Text summary ──
    txt_path = f"{SAVE_DIR}/sweep_lambda_acs.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines))
    print(f"\nText summary → {txt_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="λ sensitivity sweep (PyTorch)")
    p.add_argument("--gap-json",  default="outputs/gaps/acs_gender_gap.json")
    p.add_argument("--tensors",   default="outputs/data/acs_tensors.pt")
    p.add_argument("--logreg",    default="outputs/outcome/acs/logreg.joblib")
    p.add_argument("--mlp",       default="outputs/outcome/acs/mlp.joblib")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--nu",        type=float, default=0.05)
    p.add_argument("--n-max",     type=int,   default=250,
                   help="TN individuals for the sweep (default 250)")
    p.add_argument("--n-boot",    type=int,   default=1000)
    p.add_argument("--no-cache",  action="store_true",
                   help="Ignore existing cache and recompute")
    main(p.parse_args())
