"""
compute_recourse_acs.py — Minimum-cost counterfactual recourse for ACS Income.

Targets true negatives (y = 0, ŷ = 0) and finds the minimum-cost intervention
on actionable mediators W = (SCHL_GRP, OCCP_GRP, WKHP) under the soft
local-fairness-invariant recourse objective.

Candidate generation is fully vectorised: for each individual, all
(SCHL', OCCP', WKHP-grid) combinations are evaluated with two bulk
predict_proba calls (one per sex value), replacing thousands of single-row
calls.

Actionability constraints
--------------------------
  SCHL_GRP : monotone — can only increase (education is irreversible)
  OCCP_GRP : free     — any occupation is reachable
  WKHP     : monotone — can only increase, capped at 60 h/week

Objective
---------
  min_{w'} cost(w, w') + η · S(w')

  cost = α_WKHP · (ΔWKHP)²
       + α_SCHL · I(SCHL' > SCHL)
       + α_OCCP · I(OCCP' ≠ OCCP)

  S(w') = max(0, τ−ν − p(x_i,   w', z))   local fairness invariance:
         + max(0, τ−ν − p(1−x_i, w', z))   shortfall at BOTH sex values

  Weights α_j (from training split):
    α_WKHP = 1 / Var(WKHP_train)     continuous → inverse empirical variance
    α_SCHL = 1 / H(SCHL_GRP_train)   discrete   → inverse Shannon entropy
    α_OCCP = 1 / H(OCCP_GRP_train)   discrete   → inverse Shannon entropy

By default (--gap-json provided) η = λ_EB = |NDE| / |NIE| per model.

Outputs
-------
  outputs/recourse/acs_recourse.tex
  outputs/recourse/acs_recourse.txt

Usage
-----
  python compute_recourse_acs.py
  python compute_recourse_acs.py --gap-json outputs/gaps/acs_gender_gap.json
  python compute_recourse_acs.py --eta 2.5 --n-max 2000
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import joblib

from train_outcome_acs import build_features

SAVE_DIR  = "outputs/recourse"
WKHP_CLIP = (1.0, 60.0)
WKHP_STEP = 0.5          # hours — dense grid resolution


# ── Cost weights from training data ──────────────────────────────────────────

def _shannon_entropy(counts: np.ndarray) -> float:
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p + 1e-12)))


def compute_weights(data: dict) -> dict:
    """
    Compute α_j cost weights from the training split.

    Returns dict with keys 'WKHP', 'SCHL_GRP', 'OCCP_GRP'.
    """
    scaler    = data["scaler"]
    vocab     = data["vocab"]
    Wc_tr     = data["Wc_tr"]
    Wd_tr     = data["Wd_tr"]

    wkhp_h    = np.clip(scaler.inverse_transform(Wc_tr.numpy()).ravel(), *WKHP_CLIP)
    weights   = {"WKHP": 1.0 / float(np.var(wkhp_h))}

    for col_idx, col in enumerate(vocab.keys()):
        n_cats        = vocab[col]
        counts        = np.bincount(Wd_tr[:, col_idx].numpy(),
                                    minlength=n_cats).astype(float)
        weights[col]  = 1.0 / _shannon_entropy(counts)

    return weights


# ── Vectorised candidate generation ──────────────────────────────────────────

def precompute_candidates_batch(pipe, scaler, xi, zi, wdi, wci,
                                vocab, weights, threshold, nu):
    """
    Evaluate every (SCHL', OCCP', WKHP') candidate for one individual.

    Builds a (n_candidates, 6) feature matrix and calls pipe.predict_proba
    twice (once for xi, once for xi_cf) instead of calling it per candidate.

    Feature layout matches train_outcome_acs.build_features:
        [SEX(0), AGEP(1), POBP_US(2), SCHL_GRP(3), OCCP_GRP(4), WKHP_h(5)]

    Returns
    -------
    list[dict] — one dict per candidate with keys:
        cost, shortfall, delta_schl, occ_change, delta_wkhp
    """
    n_schl     = vocab["SCHL_GRP"]
    n_occp     = vocab["OCCP_GRP"]
    schl_cur   = int(wdi[0, 0].item())
    occp_cur   = int(wdi[0, 1].item())
    wkhp_cur_h = float(np.clip(scaler.inverse_transform(wci.numpy())[0, 0], *WKHP_CLIP))
    xi_val     = float(xi[0, 0].item())
    xi_cf_val  = 1.0 - xi_val
    agep       = float(zi[0, 0].item())
    pobp       = float(zi[0, 1].item())
    soft_thr   = threshold - nu

    alpha_wkhp = weights["WKHP"]
    alpha_schl = weights["SCHL_GRP"]
    alpha_occp = weights["OCCP_GRP"]

    wkhp_h_grid = np.clip(
        np.arange(wkhp_cur_h, WKHP_CLIP[1] + WKHP_STEP, WKHP_STEP),
        *WKHP_CLIP,
    ).astype(np.float32)
    n_wkhp  = len(wkhp_h_grid)
    dw_grid = (wkhp_h_grid - wkhp_cur_h).astype(np.float32)

    schl_range = range(schl_cur, n_schl)          # education only increases
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
        ds         = schl - schl_cur
        schl_term  = alpha_schl * int(ds > 0)
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

    # Feature matrix: two rows sets — one per sex value
    feat        = np.empty((n, 6), dtype=np.float32)
    feat[:, 0]  = xi_val
    feat[:, 1]  = agep
    feat[:, 2]  = pobp
    feat[:, 3]  = schl_col
    feat[:, 4]  = occp_col
    feat[:, 5]  = wkhp_col

    feat_cf     = feat.copy()
    feat_cf[:, 0] = xi_cf_val

    p_xi  = pipe.predict_proba(feat)[:, 1]
    p_xcf = pipe.predict_proba(feat_cf)[:, 1]
    sf    = np.maximum(0.0, soft_thr - p_xi) + np.maximum(0.0, soft_thr - p_xcf)

    return [
        {
            "cost":       float(cost_col[i]),
            "shortfall":  float(sf[i]),
            "delta_schl": int(ds_col[i]),
            "occ_change": int(oc_col[i]),
            "delta_wkhp": float(dw_col[i]),
        }
        for i in range(n)
    ]


def select_best(candidates: list[dict], eta: float) -> dict:
    """Return the candidate minimising  cost + η · shortfall."""
    return min(candidates, key=lambda c: c["cost"] + eta * c["shortfall"])


# ── Batch recourse ────────────────────────────────────────────────────────────

def run_recourse_batch(pipe, scaler, data, vocab, weights,
                       threshold, nu, eta, n_max=None, rng_seed=0):
    """
    Identify true negatives, precompute candidate grids (vectorised),
    and select the best intervention at penalty weight η = eta.

    Parameters
    ----------
    n_max : int | None
        Maximum TN individuals to process.  None = all (full validation set).
    """
    X_va  = data["X_va"]
    Z_va  = data["Z_va"]
    Wd_va = data["Wd_va"]
    Wc_va = data["Wc_va"]
    Y_va  = data["Y_va"].numpy().astype(int)

    feat_all = build_features(X_va, Z_va, Wd_va, Wc_va, scaler)
    yhat_all = pipe.predict(feat_all)
    tn_mask  = (Y_va == 0) & (yhat_all == 0)
    tn_idx   = np.where(tn_mask)[0]
    print(f"  True negatives: {len(tn_idx):,} / {len(Y_va):,}")

    rng = np.random.default_rng(rng_seed)
    if n_max is not None and len(tn_idx) > n_max:
        tn_idx = rng.choice(tn_idx, n_max, replace=False)
        print(f"  Subsampled to {n_max:,}")

    results = []
    t0      = time.time()

    for k, i in enumerate(tn_idx):
        cands       = precompute_candidates_batch(
            pipe, scaler,
            X_va[i].unsqueeze(0), Z_va[i].unsqueeze(0),
            Wd_va[i].unsqueeze(0), Wc_va[i].unsqueeze(0),
            vocab, weights, threshold, nu,
        )
        best        = select_best(cands, eta)
        best["sex"] = int(X_va[i, 0].item())
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


def aggregate(results: list[dict], n_boot: int = 1000) -> dict:
    """
    Aggregate recourse results for a list of individual dicts.

    Feasibility: shortfall == 0 at the optimal solution (both p(xi,w') and
    p(1−xi,w') are ≥ τ−ν).  All statistics are over ALL individuals.
    """
    n = len(results)
    if n == 0:
        nan3 = (float("nan"),) * 3
        return {"n": 0, "n_feasible": 0, "feasible_rate": nan3,
                "delta_schl": nan3, "occ_change": nan3,
                "delta_wkhp": nan3, "cost": nan3, "shortfall": nan3}

    rng    = np.random.default_rng(0)
    feas   = np.array([float(r["shortfall"] == 0.0) for r in results])
    d_schl = np.array([r["delta_schl"] for r in results], dtype=float)
    d_occ  = np.array([r["occ_change"] for r in results], dtype=float)
    d_wkhp = np.array([r["delta_wkhp"] for r in results], dtype=float)
    d_cost = np.array([r["cost"]       for r in results], dtype=float)
    d_sf   = np.array([r["shortfall"]  for r in results], dtype=float)

    return {
        "n":             n,
        "n_feasible":    int(feas.sum()),
        "feasible_rate": _bootstrap_ci(feas,   n_boot, rng=rng),
        "delta_schl":    _bootstrap_ci(d_schl, n_boot, rng=rng),
        "occ_change":    _bootstrap_ci(d_occ,  n_boot, rng=rng),
        "delta_wkhp":    _bootstrap_ci(d_wkhp, n_boot, rng=rng),
        "cost":          _bootstrap_ci(d_cost, n_boot, rng=rng),
        "shortfall":     _bootstrap_ci(d_sf,   n_boot, rng=rng),
    }


def stratify(results: list[dict], n_boot: int = 1000) -> dict:
    female = [r for r in results if r["sex"] == 0]
    male   = [r for r in results if r["sex"] == 1]
    return {
        "overall": aggregate(results, n_boot),
        "female":  aggregate(female,  n_boot),
        "male":    aggregate(male,    n_boot),
    }


# ── LaTeX table ───────────────────────────────────────────────────────────────

def _ci_cell(mean, lo, hi, fmt=".2f"):
    return rf"${mean:{fmt}}\ [{lo:{fmt}},\ {hi:{fmt}}]$"


def _pct_cell(mean, lo, hi):
    return rf"${100*mean:.1f}\ [{100*lo:.1f},\ {100*hi:.1f}]\%$"


def make_recourse_table(all_stats: dict, model_names: list[str],
                        caption: str, label: str) -> str:
    strata = [("overall", "Overall"), ("female", "Female"), ("male", "Male")]
    lines  = [
        r"\begin{table}[htbp]",
        r"  \centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{llccccccc}",
        r"    \toprule",
        r"    Model & Group & $n_{\mathrm{TN}}$ & Feasible ($S{=}0$) & "
        r"$\Delta\mathrm{Edu}$ & Occ.\ change & $\Delta\mathrm{WKHP}$ (h/wk)"
        r" & Cost & Shortfall $S$ \\",
        r"    \midrule",
    ]
    for mi, name in enumerate(model_names):
        if mi > 0:
            lines.append(r"    \midrule")
        stats = all_stats[name]
        first = True
        for key, label_str in strata:
            s = stats.get(key)
            if s is None:
                continue
            model_cell = name if first else ""
            first = False
            lines.append(
                f"    {model_cell} & {label_str} & {s['n']} & "
                f"{_pct_cell(*s['feasible_rate'])} & "
                f"{_ci_cell(*s['delta_schl'])} & "
                f"{_pct_cell(*s['occ_change'])} & "
                f"{_ci_cell(*s['delta_wkhp'], fmt='.1f')} & "
                f"{_ci_cell(*s['cost'], fmt='.3f')} & "
                f"{_ci_cell(*s['shortfall'], fmt='.4f')} \\\\"
            )
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ── Human-readable summary ────────────────────────────────────────────────────

def print_summary(all_stats: dict, model_names: list[str]) -> str:
    lines = []
    for name in model_names:
        hdr = f"\n{'='*60}\n{name}\n{'='*60}"
        print(hdr)
        lines.append(hdr)
        for key, label_str in [("overall", "Overall"), ("female", "Female"),
                                ("male", "Male")]:
            s = all_stats[name].get(key)
            if s is None:
                continue
            fr  = s["feasible_rate"]
            row = (
                f"  [{label_str:7s}]  n={s['n']:5d}  "
                f"feasible={fr[0]:.1%} [{fr[1]:.1%},{fr[2]:.1%}]  "
                f"Δedu={s['delta_schl'][0]:.2f}  "
                f"occ%={100*s['occ_change'][0]:.1f}  "
                f"Δwkhp={s['delta_wkhp'][0]:.1f}  "
                f"cost={s['cost'][0]:.3f}  "
                f"shortfall={s['shortfall'][0]:.4f}"
            )
            print(row)
            lines.append(row)
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    data    = torch.load(args.tensors, map_location="cpu", weights_only=False)
    scaler  = data["scaler"]
    vocab   = data["vocab"]
    weights = compute_weights(data)

    print("Cost weights (from training data):")
    print(f"  α_WKHP = {weights['WKHP']:.6f}  "
          f"α_SCHL = {weights['SCHL_GRP']:.6f}  "
          f"α_OCCP = {weights['OCCP_GRP']:.6f}")

    pipe_lr  = joblib.load(args.logreg)
    pipe_mlp = joblib.load(args.mlp)
    models   = [(pipe_lr, "Logistic Reg."), (pipe_mlp, "MLP (64--32)")]

    # Load per-model λ_EB from gap JSON (takes priority over --eta)
    lam_by_model = {}
    if args.gap_json and os.path.exists(args.gap_json):
        with open(args.gap_json, encoding="utf-8") as f:
            snap = json.load(f)
        for mname, strata in snap.items():
            nde = strata["overall"]["nde"]
            nie = strata["overall"]["nie"]
            if abs(nie) > 1e-12:
                lam_by_model[mname] = abs(nde / nie)
        if lam_by_model:
            print("λ_EB loaded from gap JSON:")
            for mname, lv in lam_by_model.items():
                print(f"  {mname}: {lv:.4f}")
    else:
        print(f"Using fixed λ = {args.eta} for all models.")

    n_max = args.n_max if args.n_max > 0 else None

    all_stats   = {}
    model_names = []

    for pipe, name in models:
        eta = lam_by_model.get(name, args.eta)
        print(f"\n[{name}]  λ={eta:.4f}  τ={args.threshold}  "
              f"ν={args.nu}  soft_thr={args.threshold - args.nu:.3f}")
        results         = run_recourse_batch(
            pipe, scaler, data, vocab, weights,
            threshold=args.threshold, nu=args.nu, eta=eta,
            n_max=n_max,
        )
        all_stats[name] = stratify(results, n_boot=args.n_boot)
        model_names.append(name)

    summary = print_summary(all_stats, model_names)
    os.makedirs(SAVE_DIR, exist_ok=True)

    tex = make_recourse_table(
        all_stats, model_names,
        caption=(
            r"Minimum-cost counterfactual recourse for ACS Income true negatives "
            r"($y{=}0$, $\hat{y}{=}0$). "
            r"Actionability: Education ($\Delta\mathrm{Edu} \geq 0$), "
            r"Occupation (free), Hours ($\Delta\mathrm{WKHP} \geq 0$, "
            r"$\leq 60$\,h/wk). "
            r"$\lambda = \lambda_{\mathrm{EB}} = |\mathrm{NDE}|/|\mathrm{NIE}|$ "
            r"per model. "
            r"Feasible ($S{=}0$): $p(x_i,w',z) \geq \tau{-}\nu$ and "
            r"$p(1{-}x_i,w',z) \geq \tau{-}\nu$ (local fairness invariance). "
            r"Values: mean with 95\% bootstrap CI."
        ),
        label="tab:acs_recourse",
    )

    tex_path = f"{SAVE_DIR}/acs_recourse.tex"
    txt_path = f"{SAVE_DIR}/acs_recourse.txt"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"\nSaved → {tex_path}")
    print(f"Saved → {txt_path}")
    print("\n" + tex)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="ACS Income counterfactual recourse")
    p.add_argument("--tensors",   default="outputs/data/acs_tensors.pt")
    p.add_argument("--logreg",    default="outputs/outcome/acs/logreg.joblib")
    p.add_argument("--mlp",       default="outputs/outcome/acs/mlp.joblib")
    p.add_argument("--gap-json",  default="outputs/gaps/acs_gender_gap.json",
                   help="JSON from estimate_gender_gap_acs.py; sets λ=λ_EB per model")
    p.add_argument("--eta",       type=float, default=10.0,
                   help="Fallback penalty weight λ if --gap-json is absent (default 10.0)")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Decision threshold τ (default 0.5)")
    p.add_argument("--nu",        type=float, default=0.05,
                   help="Slack ν: shortfall activates when p < τ−ν (default 0.05)")
    p.add_argument("--n-max",     type=int,   default=0,
                   help="Max TN individuals. 0 = all (default).")
    p.add_argument("--n-boot",    type=int,   default=1000,
                   help="Bootstrap iterations for CIs (default 1000)")
    main(p.parse_args())
