"""
estimate_gender_gap_acs.py — Mediated vs direct gender income gap on ACS Income.

Decomposes the observed gender income gap into:

  NDE  (Natural Direct Effect)   — direct Sex → Income path
  NIE  (Natural Indirect Effect) — indirect paths via mediators
                                    (Education, Occupation, Hours worked)
  TE   (Total Effect)            = NDE + NIE

Estimator
---------
  For each individual i with confounders Z = z_i, always sampling from the
  Female baseline P(W | X = 0, Z = z_i):

    1. Draw K samples  W_k ~ P(W | X = Female, Z = z_i)
    2. NDE_i = (1/K) Σ_k [ f(Male,   W_k, z_i) - f(Female, W_k, z_i) ]
    3. NIE_i = Σ_k r̄_k f(Female, W_k, z_i)  -  (1/K) Σ_k f(Female, W_k, z_i)
               r̄_k ∝ P(W_k | Male, z_i) / P(W_k | Female, z_i)

  Population averages are taken over all (subsampled) validation individuals.
  Stratified estimates split by each individual's observed sex.

Outputs
-------
  outputs/gaps/acs_gender_gap_overall.tex    — overall NDE / NIE / TE table
  outputs/gaps/acs_gender_gap_stratified.tex — stratified by observed sex
  outputs/gaps/acs_gender_gap.txt            — human-readable summary

Usage
-----
  python estimate_gender_gap_acs.py
  python estimate_gender_gap_acs.py --K 1000 --n-inst 1000 --n-boot 2000
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import joblib

from train_acs_flow import load_flow_models
from train_outcome_acs import build_features
from inspect_acs_model import make_sample_fns

SAVE_DIR = "outputs/gaps"
X0, X1 = 0.0, 1.0   # Female = 0, Male = 1


# ── Outcome model wrapper ─────────────────────────────────────────────────────

def make_outcome_fn(pipe, scaler):
    """
    Wrap an sklearn Pipeline into a callable:
      (x, wd, wc_std, z) → Tensor of shape (N,)

    x       : (N, 1)  SEX in {0, 1}
    wd      : (N, 2)  [SCHL_GRP, OCCP_GRP] — integer category indices as float
    wc_std  : (N, 1)  WKHP in QuantileTransformer output space (N(0,1))
    z       : (N, 2)  [AGEP, POBP_US]

    build_features inverts the scaler internally (no extra clipping needed;
    the QT was fitted on clipped [1, 60] values so inverse maps to that range).
    """
    def fn(x: torch.Tensor, wd: torch.Tensor,
           wc_std: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        feat  = build_features(x, z, wd, wc_std, scaler)   # (N, 6) numpy
        probs = pipe.predict_proba(feat)[:, 1].astype(np.float32)
        return torch.tensor(probs)
    return fn


# ── Batched NDE / NIE estimator ───────────────────────────────────────────────

def estimate_nde_nie(
    Z_val: torch.Tensor,
    sample_fn,
    log_prob_fn,
    outcome_fn,
    K: int = 500,
    batch_size: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate per-individual NDE and NIE by sampling W ~ P(W | X=Female, Z=z_i)
    for every individual in Z_val.

    Returns nde, nie — both arrays of shape (N,).
    """
    N     = len(Z_val)
    X_ref = torch.zeros(N, 1)   # always Female reference for sampling
    nde_list, nie_list = [], []

    for start in range(0, N, batch_size):
        end  = min(start + batch_size, N)
        xi   = X_ref[start:end]         # (b, 1)
        zi   = Z_val[start:end]         # (b, 2)
        b    = len(xi)

        with torch.no_grad():
            out    = sample_fn(xi, zi, K=K)
            wd     = out["w_disc"]           # (b*K, 2)
            wc     = out["w_cont"]           # (b*K, 1)
            z_rep  = out["z_rep"]            # (b*K, 2)
            log_p0 = out["log_prob"]         # (b*K,)

            x0_rep = torch.full((b * K, 1), X0)
            x1_rep = torch.full((b * K, 1), X1)

            f_x1 = outcome_fn(x1_rep, wd, wc, z_rep).view(b, K)  # (b, K)
            f_x0 = outcome_fn(x0_rep, wd, wc, z_rep).view(b, K)  # (b, K)

            nde_i = (f_x1 - f_x0).mean(dim=1)                    # (b,)
            nde_list.extend(nde_i.numpy().tolist())

            log_p1   = log_prob_fn(wd, wc, x1_rep, z_rep).view(b, K)  # (b, K)
            log_p0_2d = log_p0.view(b, K)                              # (b, K)

            log_r = log_p1 - log_p0_2d
            log_r = log_r - log_r.max(dim=1, keepdim=True).values  # stabilise
            r     = torch.exp(log_r)
            r_bar = r / r.sum(dim=1, keepdim=True)                 # (b, K)

            nie_i = (r_bar * f_x0).sum(dim=1) - f_x0.mean(dim=1)  # (b,)
            nie_list.extend(nie_i.numpy().tolist())

    return np.array(nde_list), np.array(nie_list)


# ── Bootstrap CI ──────────────────────────────────────────────────────────────

def bootstrap_ci(arr: np.ndarray, n_boot: int = 2000,
                 alpha: float = 0.05) -> tuple[float, float, float]:
    """Return (mean, lower, upper) percentile bootstrap CI."""
    rng   = np.random.default_rng(0)
    means = np.array([
        arr[rng.choice(len(arr), len(arr), replace=True)].mean()
        for _ in range(n_boot)
    ])
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return float(arr.mean()), float(lo), float(hi)


# ── Raw observable gap ────────────────────────────────────────────────────────

def compute_raw_gap(pipe, scaler, data: dict) -> tuple[float, float, float]:
    """
    E[ f(Male observed data) ] - E[ f(Female observed data) ]
    using each group's actual mediator and confounder values.
    """
    X_va  = data["X_va"]
    Z_va  = data["Z_va"]
    Wd_va = data["Wd_va"]
    Wc_va = data["Wc_va"]

    mask_f = (X_va[:, 0] == 0).numpy()
    mask_m = (X_va[:, 0] == 1).numpy()

    feat_f = build_features(X_va[mask_f], Z_va[mask_f], Wd_va[mask_f], Wc_va[mask_f], scaler)
    feat_m = build_features(X_va[mask_m], Z_va[mask_m], Wd_va[mask_m], Wc_va[mask_m], scaler)

    p_f = float(pipe.predict_proba(feat_f)[:, 1].mean())
    p_m = float(pipe.predict_proba(feat_m)[:, 1].mean())
    return p_m - p_f, p_m, p_f


# ── Full gap decomposition for one outcome model ──────────────────────────────

def compute_gap_stats(
    pipe, scaler, data: dict,
    sample_fn, log_prob_fn,
    K: int, n_inst: int, n_boot: int,
) -> dict:
    """
    Run the full gap decomposition for a single outcome model.

    Returns a results dict with keys:
      'raw', 'p_male', 'p_female'
      'overall': {'nde', 'nie', 'te'}  (each a (mean, lo, hi) triple)
      'female' : {'nde', 'nie', 'te'}
      'male'   : {'nde', 'nie', 'te'}
    """
    X_va = data["X_va"]
    Z_va = data["Z_va"]

    # Subsample validation set
    idx   = torch.randperm(len(X_va))[:n_inst]
    X_sub = X_va[idx]
    Z_sub = Z_va[idx]

    outcome_fn = make_outcome_fn(pipe, scaler)

    print(f"    Estimating NDE/NIE (K={K}, n={n_inst})...")
    t0 = time.time()
    nde, nie = estimate_nde_nie(Z_sub, sample_fn, log_prob_fn, outcome_fn, K=K)
    te        = nde + nie
    print(f"    Done in {time.time() - t0:.1f}s")

    mask_f = (X_sub[:, 0] == 0).numpy().astype(bool)
    mask_m = (X_sub[:, 0] == 1).numpy().astype(bool)

    def _ci_triple(arr):
        return bootstrap_ci(arr, n_boot)

    res = {
        "overall": {
            "nde": _ci_triple(nde),
            "nie": _ci_triple(nie),
            "te":  _ci_triple(te),
            "n":   len(nde),
        },
    }

    for label, mask in [("female", mask_f), ("male", mask_m)]:
        if mask.sum() >= 10:
            nde_g, nie_g, te_g = nde[mask], nie[mask], te[mask]
            res[label] = {
                "nde": _ci_triple(nde_g),
                "nie": _ci_triple(nie_g),
                "te":  _ci_triple(te_g),
                "n":   int(mask.sum()),
            }

    raw, p_m, p_f = compute_raw_gap(pipe, scaler, data)
    res["raw"]      = raw
    res["p_male"]   = p_m
    res["p_female"] = p_f

    return res


# ── LaTeX table helpers ───────────────────────────────────────────────────────

def _cell(mean: float, lo: float, hi: float, d: int = 4) -> str:
    return rf"${mean:+.{d}f}\ [{lo:+.{d}f},\ {hi:+.{d}f}]$"


def _raw_cell(v: float, d: int = 4) -> str:
    return f"${v:+.{d}f}$"


def make_overall_table(
    all_results: list[dict],
    model_names: list[str],
    caption: str,
    label: str,
) -> str:
    """One row per outcome model — overall estimates only."""
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        r"  \begin{tabular}{lcccc}",
        r"    \toprule",
        r"    Model & Raw gap & NDE & NIE & TE \\",
        r"    \midrule",
    ]
    for name, res in zip(model_names, all_results):
        ov = res["overall"]
        lines.append(
            f"    {name} & {_raw_cell(res['raw'])} & "
            f"{_cell(*ov['nde'])} & {_cell(*ov['nie'])} & {_cell(*ov['te'])} \\\\"
        )
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def make_stratified_table(
    all_results: list[dict],
    model_names: list[str],
    caption: str,
    label: str,
) -> str:
    """Two rows per outcome model — Female and Male strata."""
    strata = [("female", "Female"), ("male", "Male")]
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        r"  \begin{tabular}{llccc}",
        r"    \toprule",
        r"    Model & Stratum & NDE & NIE & TE \\",
        r"    \midrule",
    ]
    for i, (name, res) in enumerate(zip(model_names, all_results)):
        if i > 0:
            lines.append(r"    \midrule")
        first = True
        for key, label_str in strata:
            if key not in res:
                continue
            s = res[key]
            model_cell = name if first else ""
            lines.append(
                f"    {model_cell} & {label_str} & "
                f"{_cell(*s['nde'])} & {_cell(*s['nie'])} & {_cell(*s['te'])} \\\\"
            )
            first = False
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ── Human-readable summary ────────────────────────────────────────────────────

def print_and_collect_summary(
    all_results: list[dict],
    model_names: list[str],
) -> str:
    lines = []
    for name, res in zip(model_names, all_results):
        header = f"{'='*60}\n{name}\n{'='*60}"
        print(f"\n{header}")
        lines.append(header)

        raw_line = (
            f"Raw gap : {res['raw']:+.4f}  "
            f"(P_male={res['p_male']:.4f}, P_female={res['p_female']:.4f})"
        )
        print(f"  {raw_line}")
        lines.append(raw_line)

        for key, label_str in [("overall", "Overall"), ("female", "Female"), ("male", "Male")]:
            if key not in res:
                continue
            s   = res[key]
            n   = res[key]["n"]
            row = (
                f"  [{label_str:7s}] n={n:5d}  "
                f"NDE={s['nde'][0]:+.4f} [{s['nde'][1]:+.4f}, {s['nde'][2]:+.4f}]  "
                f"NIE={s['nie'][0]:+.4f} [{s['nie'][1]:+.4f}, {s['nie'][2]:+.4f}]  "
                f"TE={s['te'][0]:+.4f}  [{s['te'][1]:+.4f}, {s['te'][2]:+.4f}]"
            )
            print(row)
            lines.append(row)
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load flow models
    g_phi, f_theta, scaler, _ = load_flow_models(args.flow, device)
    sample_fn, log_prob_fn    = make_sample_fns(g_phi, f_theta, scaler, device)

    # Load tensors + outcome models
    data     = torch.load(args.tensors, map_location="cpu", weights_only=False)
    pipe_lr  = joblib.load(args.logreg)
    pipe_mlp = joblib.load(args.mlp)

    models       = [(pipe_lr, "Logistic Reg."), (pipe_mlp, "MLP (64--32)")]
    all_results  = []
    model_names  = []

    for pipe, name in models:
        print(f"\n[{name}]")
        res = compute_gap_stats(
            pipe, scaler, data,
            sample_fn, log_prob_fn,
            K=args.K, n_inst=args.n_inst, n_boot=args.n_boot,
        )
        all_results.append(res)
        model_names.append(name)

    # Print + collect text summary
    summary = print_and_collect_summary(all_results, model_names)

    os.makedirs(SAVE_DIR, exist_ok=True)

    # Overall LaTeX table
    tex_overall = make_overall_table(
        all_results, model_names,
        caption=(
            r"Gender income gap decomposition on ACS Income (validation set). "
            r"NDE = Natural Direct Effect (direct Sex $\to$ Income path); "
            r"NIE = Natural Indirect Effect (via Education, Occupation, Hours worked); "
            r"TE = Total Effect $=$ NDE $+$ NIE. "
            r"Values are means with 95\% bootstrap confidence intervals "
            r"($K=" + str(args.K) + r"$ IS samples per individual)."
        ),
        label="tab:acs_gender_gap_overall",
    )

    # Stratified LaTeX table
    tex_strat = make_stratified_table(
        all_results, model_names,
        caption=(
            r"Gender income gap decomposition stratified by the individual's "
            r"observed sex. W is always sampled from the Female baseline "
            r"$P(W \mid X{=}\text{Female}, Z)$. "
            r"95\% bootstrap confidence intervals in brackets."
        ),
        label="tab:acs_gender_gap_stratified",
    )

    path_ov    = f"{SAVE_DIR}/acs_gender_gap_overall.tex"
    path_st    = f"{SAVE_DIR}/acs_gender_gap_stratified.tex"
    path_txt   = f"{SAVE_DIR}/acs_gender_gap.txt"

    with open(path_ov,  "w", encoding="utf-8") as f:
        f.write(tex_overall)
    with open(path_st,  "w", encoding="utf-8") as f:
        f.write(tex_strat)
    with open(path_txt, "w", encoding="utf-8") as f:
        f.write(summary)

    # JSON snapshot used by sweep_lambda_acs.py to read λ_EB = |NDE/NIE|
    path_json = f"{SAVE_DIR}/acs_gender_gap.json"
    snapshot = {}
    for name, res in zip(model_names, all_results):
        snapshot[name] = {
            stratum: {
                "nde": res[stratum]["nde"][0],
                "nie": res[stratum]["nie"][0],
                "te":  res[stratum]["te"][0],
            }
            for stratum in ("overall", "female", "male")
            if stratum in res
        }
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)

    print(f"\nSaved → {path_ov}")
    print(f"Saved → {path_st}")
    print(f"Saved → {path_txt}")
    print(f"Saved → {path_json}")
    print("\n── Overall table ──────────────────────────────────────────")
    print(tex_overall)
    print("\n── Stratified table ───────────────────────────────────────")
    print(tex_strat)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="ACS Income gender gap decomposition")
    p.add_argument("--flow",    default="outputs/flows/acs/flow_models.pt")
    p.add_argument("--tensors", default="outputs/data/acs_tensors.pt")
    p.add_argument("--logreg",  default="outputs/outcome/acs/logreg.joblib")
    p.add_argument("--mlp",     default="outputs/outcome/acs/mlp.joblib")
    p.add_argument("--K",       type=int, default=500,
                   help="IS samples per individual (default 500)")
    p.add_argument("--n-inst",  type=int, default=500,
                   help="Validation individuals to use (default 500)")
    p.add_argument("--n-boot",  type=int, default=2000,
                   help="Bootstrap iterations for CIs (default 2000)")
    main(p.parse_args())
