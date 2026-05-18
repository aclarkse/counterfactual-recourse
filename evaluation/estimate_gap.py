"""
evaluation/estimate_gap.py — Generic Hydra entry point for NDE/NIE decomposition.

Run: python -m evaluation.estimate_gap [dataset=acs|bar]
"""

import json
import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import joblib

import hydra
from omegaconf import OmegaConf

from flows.models import load_flow_models
from flows.diagnostics import make_sample_fns
from data.build_tensors import stack_sfm_features

X0, X1 = 0.0, 1.0   # group 0 = reference for sampling


# ── Outcome model wrapper ─────────────────────────────────────────────────────

def make_outcome_fn(pipe, scaler):
    """
    Wrap an sklearn Pipeline into a callable:
      (x, wd, wc_std, z) → Tensor of shape (N,)
    """
    def fn(x: torch.Tensor, wd: torch.Tensor,
           wc_std: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        feat  = stack_sfm_features(x, z, wd, wc_std, scaler)   # numpy
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
) -> tuple:
    """
    Estimate per-individual NDE and NIE by sampling W ~ P(W | X=0, Z=z_i)
    for every individual in Z_val.

    Returns nde, nie — both arrays of shape (N,).
    """
    N     = len(Z_val)
    X_ref = torch.zeros(N, 1)   # always group-0 reference for sampling
    nde_list, nie_list = [], []

    for start in range(0, N, batch_size):
        end  = min(start + batch_size, N)
        xi   = X_ref[start:end]         # (b, 1)
        zi   = Z_val[start:end]         # (b, dim_z)
        b    = len(xi)

        with torch.no_grad():
            out    = sample_fn(xi, zi, K=K)
            wd     = out["w_disc"]           # (b*K, dim_wd)
            wc     = out["w_cont"]           # (b*K, dim_wc)
            z_rep  = out["z_rep"]            # (b*K, dim_z)
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
                 alpha: float = 0.05) -> tuple:
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

def compute_raw_gap(pipe, scaler, data: dict) -> tuple:
    """
    E[ f(group-1 observed data) ] - E[ f(group-0 observed data) ]
    using each group's actual mediator and confounder values.
    """
    X_va  = data["X_va"]
    Z_va  = data["Z_va"]
    Wd_va = data["Wd_va"]
    Wc_va = data["Wc_va"]

    mask_0 = (X_va[:, 0] == 0).numpy()
    mask_1 = (X_va[:, 0] == 1).numpy()

    feat_0 = stack_sfm_features(X_va[mask_0], Z_va[mask_0], Wd_va[mask_0], Wc_va[mask_0], scaler)
    feat_1 = stack_sfm_features(X_va[mask_1], Z_va[mask_1], Wd_va[mask_1], Wc_va[mask_1], scaler)

    p_0 = float(pipe.predict_proba(feat_0)[:, 1].mean())
    p_1 = float(pipe.predict_proba(feat_1)[:, 1].mean())
    return p_1 - p_0, p_1, p_0


# ── Full gap decomposition for one outcome model ──────────────────────────────

def compute_gap_stats(
    pipe, scaler, data: dict,
    sample_fn, log_prob_fn,
    K: int, n_inst: int, n_boot: int,
) -> dict:
    """
    Run the full gap decomposition for a single outcome model.

    Returns a results dict with keys:
      'raw', 'p_g1', 'p_g0'
      'overall': {'nde', 'nie', 'te'}  (each a (mean, lo, hi) triple)
      'g0'     : {'nde', 'nie', 'te'}
      'g1'     : {'nde', 'nie', 'te'}
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

    mask_0 = (X_sub[:, 0] == 0).numpy().astype(bool)
    mask_1 = (X_sub[:, 0] == 1).numpy().astype(bool)

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

    for label, mask in [("g0", mask_0), ("g1", mask_1)]:
        if mask.sum() >= 10:
            nde_g, nie_g, te_g = nde[mask], nie[mask], te[mask]
            res[label] = {
                "nde": _ci_triple(nde_g),
                "nie": _ci_triple(nie_g),
                "te":  _ci_triple(te_g),
                "n":   int(mask.sum()),
            }

    raw, p_1, p_0 = compute_raw_gap(pipe, scaler, data)
    res["raw"]  = raw
    res["p_g1"] = p_1
    res["p_g0"] = p_0

    return res


# ── LaTeX table helpers ───────────────────────────────────────────────────────

def _cell(mean: float, lo: float, hi: float, d: int = 4) -> str:
    return rf"${mean:+.{d}f}\ [{lo:+.{d}f},\ {hi:+.{d}f}]$"


def _raw_cell(v: float, d: int = 4) -> str:
    return f"${v:+.{d}f}$"


def make_overall_table(
    all_results: list,
    model_names: list,
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
    all_results: list,
    model_names: list,
    caption: str,
    label: str,
    group_labels: dict = None,
) -> str:
    """Two rows per outcome model — group-0 and group-1 strata."""
    if group_labels is None:
        group_labels = {"g0": "Group 0", "g1": "Group 1"}
    strata = [("g0", group_labels.get("g0", "Group 0")),
              ("g1", group_labels.get("g1", "Group 1"))]
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
    all_results: list,
    model_names: list,
    group_labels: dict = None,
) -> str:
    if group_labels is None:
        group_labels = {"g0": "Group 0", "g1": "Group 1"}
    lines = []
    for name, res in zip(model_names, all_results):
        header = f"{'='*60}\n{name}\n{'='*60}"
        print(f"\n{header}")
        lines.append(header)

        raw_line = (
            f"Raw gap : {res['raw']:+.4f}  "
            f"(P_g1={res['p_g1']:.4f}, P_g0={res['p_g0']:.4f})"
        )
        print(f"  {raw_line}")
        lines.append(raw_line)

        strata_display = [
            ("overall", "Overall"),
            ("g0", group_labels.get("g0", "Group 0")),
            ("g1", group_labels.get("g1", "Group 1")),
        ]
        for key, label_str in strata_display:
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

@hydra.main(config_path="../conf", config_name="config", version_base="1.1")
def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    flows_path   = cfg.dataset.paths.flows
    tensors_path = cfg.dataset.paths.tensors
    outcome_dir  = cfg.dataset.paths.outcome_dir
    gaps_dir     = cfg.dataset.paths.gaps_dir
    dataset_name = cfg.dataset.name

    # Load flow models
    g_phi, f_theta, scaler, _ = load_flow_models(flows_path, device)
    sample_fn, log_prob_fn    = make_sample_fns(g_phi, f_theta, scaler, device)

    # Load tensors + outcome models
    data = torch.load(tensors_path, map_location="cpu", weights_only=False)

    model_specs = OmegaConf.to_container(cfg.dataset.outcome.models, resolve=True)
    gap_cfg     = cfg.dataset.gap

    # Build slug→filename mapping: type field gives the filename
    type_to_slug = {"logreg": "logreg", "mlp": "mlp"}

    all_results  = []
    model_names  = []

    for spec in model_specs:
        name = spec["name"]
        slug = type_to_slug.get(spec["type"], spec["type"])
        pipe = joblib.load(f"{outcome_dir}/{slug}.joblib")

        print(f"\n[{name}]")
        res = compute_gap_stats(
            pipe, scaler, data,
            sample_fn, log_prob_fn,
            K=gap_cfg.K, n_inst=gap_cfg.n_inst, n_boot=gap_cfg.n_boot,
        )
        all_results.append(res)
        model_names.append(name)

    # Build group labels for display
    sensitive_name = list(cfg.dataset.sfm.sensitive)[0]
    # Generic labels — could be parameterised further
    group_labels = {"g0": "Group 0", "g1": "Group 1"}

    # Print + collect text summary
    summary = print_and_collect_summary(all_results, model_names, group_labels)

    os.makedirs(gaps_dir, exist_ok=True)

    # Overall LaTeX table
    tex_overall = make_overall_table(
        all_results, model_names,
        caption=(
            f"Gap decomposition on {dataset_name} (validation set). "
            r"NDE = Natural Direct Effect; "
            r"NIE = Natural Indirect Effect; "
            r"TE = Total Effect $=$ NDE $+$ NIE. "
            r"Values are means with 95\% bootstrap confidence intervals "
            r"($K=" + str(gap_cfg.K) + r"$ IS samples per individual)."
        ),
        label=f"tab:{dataset_name}_gap_overall",
    )

    # Stratified LaTeX table
    tex_strat = make_stratified_table(
        all_results, model_names,
        caption=(
            f"Gap decomposition on {dataset_name} stratified by observed group. "
            r"W is always sampled from the group-0 baseline. "
            r"95\% bootstrap confidence intervals in brackets."
        ),
        label=f"tab:{dataset_name}_gap_stratified",
        group_labels=group_labels,
    )

    path_ov    = f"{gaps_dir}/{dataset_name}_gender_gap_overall.tex"
    path_st    = f"{gaps_dir}/{dataset_name}_gender_gap_stratified.tex"
    path_txt   = f"{gaps_dir}/{dataset_name}_gender_gap.txt"
    path_json  = f"{gaps_dir}/{dataset_name}_gender_gap.json"

    with open(path_ov,  "w", encoding="utf-8") as f:
        f.write(tex_overall)
    with open(path_st,  "w", encoding="utf-8") as f:
        f.write(tex_strat)
    with open(path_txt, "w", encoding="utf-8") as f:
        f.write(summary)

    # JSON snapshot used by compute_recourse.py / sweep_lambda.py to read λ_EB
    snapshot = {}
    for name, res in zip(model_names, all_results):
        snapshot[name] = {
            stratum: {
                "nde": res[stratum]["nde"][0],
                "nie": res[stratum]["nie"][0],
                "te":  res[stratum]["te"][0],
            }
            for stratum in ("overall", "g0", "g1")
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
    main()
