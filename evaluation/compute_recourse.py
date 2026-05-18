"""
evaluation/compute_recourse.py — Generic Hydra entry point for counterfactual recourse.

Run: python -m evaluation.compute_recourse [dataset=acs|bar]
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

from data.build_tensors import stack_sfm_features

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


def select_best(candidates: list, eta: float) -> dict:
    """Return the candidate minimising  cost + η · shortfall."""
    return min(candidates, key=lambda c: c["cost"] + eta * c["shortfall"])


# ── Batch recourse ────────────────────────────────────────────────────────────

def run_recourse_batch(pipe, scaler, data, disc_names, cont_names,
                       mediator_specs, vocab, weights,
                       threshold, nu, eta, n_max=None, rng_seed=0):
    """
    Identify true negatives, precompute candidate grids (vectorised),
    and select the best intervention at penalty weight η = eta.
    """
    X_va  = data["X_va"]
    Z_va  = data["Z_va"]
    Wd_va = data["Wd_va"]
    Wc_va = data["Wc_va"]
    Y_va  = data["Y_va"].numpy().astype(int)

    feat_all = stack_sfm_features(X_va, Z_va, Wd_va, Wc_va, scaler)
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
        cands = precompute_candidates_batch(
            pipe, scaler,
            X_va[i].unsqueeze(0), Z_va[i].unsqueeze(0),
            Wd_va[i].unsqueeze(0), Wc_va[i].unsqueeze(0),
            disc_names, cont_names, mediator_specs, vocab, weights,
            threshold, nu,
        )
        best        = select_best(cands, eta)
        sex         = int(X_va[i, 0].item())
        best["sex"] = sex
        # Post-recourse NIE: signed gap at recourse solution w'
        p_g1     = best["p_xcf"] if sex == 0 else best["p_xi"]
        p_g0     = best["p_xi"]  if sex == 0 else best["p_xcf"]
        best["nie_post"] = float(p_g1 - p_g0)
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
                "cost": nan3, "shortfall": nan3, "nie_post": nan3}
        for name in disc_names + cont_names:
            base[name] = nan3
        return base

    rng  = np.random.default_rng(0)
    feas = np.array([float(r["shortfall"] == 0.0) for r in results])
    d_cost = np.array([r["cost"]       for r in results], dtype=float)
    d_sf   = np.array([r["shortfall"]  for r in results], dtype=float)
    d_nie  = np.array([r["nie_post"]   for r in results], dtype=float)

    out = {
        "n":             n,
        "n_feasible":    int(feas.sum()),
        "feasible_rate": _bootstrap_ci(feas,   n_boot, rng=rng),
        "cost":          _bootstrap_ci(d_cost, n_boot, rng=rng),
        "shortfall":     _bootstrap_ci(d_sf,   n_boot, rng=rng),
        "nie_post":      _bootstrap_ci(d_nie,  n_boot, rng=rng),
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
    female = [r for r in results if r["sex"] == 0]
    male   = [r for r in results if r["sex"] == 1]
    return {
        "overall": aggregate(results, disc_names, cont_names, mediator_specs, n_boot),
        "female":  aggregate(female,  disc_names, cont_names, mediator_specs, n_boot),
        "male":    aggregate(male,    disc_names, cont_names, mediator_specs, n_boot),
    }


# ── LaTeX table ───────────────────────────────────────────────────────────────

def _ci_cell(mean, lo, hi, fmt=".2f"):
    return rf"${mean:{fmt}}\ [{lo:{fmt}},\ {hi:{fmt}}]$"


def _pct_cell(mean, lo, hi):
    return rf"${100*mean:.1f}\ [{100*lo:.1f},\ {100*hi:.1f}]\%$"


def make_recourse_table(all_stats: dict, model_names: list,
                        disc_names: list, cont_names: list,
                        mediator_specs: dict,
                        caption: str, label: str) -> str:
    strata = [("overall", "Overall"), ("female", "Female"), ("male", "Male")]

    # Build dynamic column headers from mediator config
    med_headers = []
    for name in disc_names + cont_names:
        display = mediator_specs[name].get("display", name)
        med_headers.append(display)

    n_med_cols = len(med_headers)
    col_spec = "ll" + "c" * (4 + n_med_cols)
    header_cells = " & ".join(med_headers)

    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        r"  \setlength{\tabcolsep}{4pt}",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
        f"    Model & Group & $n_{{\\mathrm{{TN}}}}$ & Feasible ($S{{=}}0$) & "
        f"{header_cells}"
        r" & Cost & Shortfall $S$ & $\mathrm{NIE}_{\mathrm{post}}$ \\",
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

            # Build mediator cells in order
            med_cells = []
            for mname in disc_names:
                spec = mediator_specs[mname]
                actionable = spec.get("actionable", "free")
                if actionable == "free":
                    # percentage format
                    med_cells.append(_pct_cell(*s[mname]))
                else:
                    med_cells.append(_ci_cell(*s[mname]))
            for mname in cont_names:
                med_cells.append(_ci_cell(*s[mname], fmt=".1f"))

            med_str = " & ".join(med_cells)
            lines.append(
                f"    {model_cell} & {label_str} & {s['n']} & "
                f"{_pct_cell(*s['feasible_rate'])} & "
                f"{med_str} & "
                f"{_ci_cell(*s['cost'], fmt='.3f')} & "
                f"{_ci_cell(*s['shortfall'], fmt='.4f')} & "
                f"{_ci_cell(*s['nie_post'], fmt='.4f')} \\\\"
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
        for key, label_str in [("overall", "Overall"), ("female", "Female"),
                                ("male", "Male")]:
            s = all_stats[name].get(key)
            if s is None:
                continue
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
                f"NIE_post={s['nie_post'][0]:+.4f}"
            )
            print(row)
            lines.append(row)
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

@hydra.main(config_path="../conf", config_name="config", version_base="1.1")
def main(cfg):
    tensors_path   = cfg.dataset.paths.tensors
    outcome_dir    = cfg.dataset.paths.outcome_dir
    gaps_dir       = cfg.dataset.paths.gaps_dir
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

    # Load per-model λ_EB from gap JSON
    lam_by_model = {}
    gap_json_path = f"{gaps_dir}/{dataset_name}_gender_gap.json"
    if os.path.exists(gap_json_path):
        with open(gap_json_path, encoding="utf-8") as f:
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
        print(f"Gap JSON not found at {gap_json_path}. Using fallback λ=10.0.")

    n_max_cfg = int(recourse_cfg.n_max)
    n_max     = n_max_cfg if n_max_cfg > 0 else None

    all_stats   = {}
    model_names = []

    for spec in model_specs:
        name = spec["name"]
        slug = type_to_slug.get(spec["type"], spec["type"])
        pipe = joblib.load(f"{outcome_dir}/{slug}.joblib")
        eta  = lam_by_model.get(name, 10.0)

        print(f"\n[{name}]  λ={eta:.4f}  τ={recourse_cfg.threshold}  "
              f"ν={recourse_cfg.nu}  soft_thr={recourse_cfg.threshold - recourse_cfg.nu:.3f}")
        results = run_recourse_batch(
            pipe, scaler, data,
            disc_names, cont_names, mediator_specs, vocab, weights,
            threshold=float(recourse_cfg.threshold),
            nu=float(recourse_cfg.nu),
            eta=eta,
            n_max=n_max,
        )
        all_stats[name] = stratify(results, disc_names, cont_names,
                                   mediator_specs, n_boot=int(recourse_cfg.n_boot))
        model_names.append(name)

    summary = print_summary(all_stats, model_names, disc_names, cont_names, mediator_specs)
    os.makedirs(recourse_dir, exist_ok=True)

    tex = make_recourse_table(
        all_stats, model_names, disc_names, cont_names, mediator_specs,
        caption=(
            f"Minimum-cost counterfactual recourse for {dataset_name} true negatives "
            r"($y{=}0$, $\hat{y}{=}0$). "
            r"$\lambda = \lambda_{\mathrm{EB}} = |\mathrm{NDE}|/|\mathrm{NIE}|$ "
            r"per model. "
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
