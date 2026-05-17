"""
inspect_bar_model.py — Diagnostic plots for the trained bar-passage flow model.

The bar dataset has no discrete mediators, so all diagnostics focus on the
continuous factor P(W_cont | X, Z) for W_cont = (LSAT, GPA).

Plots produced
--------------
  1. Training curve        (NLL over epochs, disc + flow split)
  2. Conditional marginals (empirical vs flow KDE per mediator per group)
  3. ESS by group          (IS quality for NDE/NIE)
       + fraction of White eval points with ESS/K < 0.05
  4. IS weight stats       (printed summary)
  5. K sensitivity         (NDE/NIE stability at K = 500, 1000, 2000)

Usage
-----
  python inspect_bar_model.py
  python inspect_bar_model.py --model outputs/flows/law_school/flow_models.pt
                               --tensors outputs/data/bar_tensors.pt
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression

from paper_style import (
    COL_GRP0, COL_GRP1, COL_NDE, COL_NIE,
    LW, MS, ALPHA_FILL,
    set_paper_style, save_figure,
)
from train_acs_flow import load_flow_models
from flow_diagnostics_shared import (
    plot_conditional_marginals,
    plot_ess_by_group,
    estimate_nde_nie_for_individual,
)

set_paper_style()

SAVE_DIR     = "figures/law_school"
GROUP_LABELS = {0: "Black", 1: "White"}
WC_NAMES     = ["LSAT score", "GPA"]


# ── Sampling utilities ────────────────────────────────────────────────────────

def make_sample_fns(g_phi, f_theta, scaler, device):
    """Return (sample_fn, log_prob_fn) closures compatible with flow_diagnostics_shared."""

    def _invert(w_cont: torch.Tensor) -> torch.Tensor:
        if scaler is None or w_cont.shape[1] == 0:
            return w_cont
        return torch.tensor(
            scaler.inverse_transform(w_cont.numpy()).astype(np.float32)
        )

    @torch.no_grad()
    def sample_fn(x, z, K=500):
        g_phi.eval(); f_theta.eval()
        x, z   = x.to(device), z.to(device)
        x_rep  = x.repeat_interleave(K, dim=0)
        z_rep  = z.repeat_interleave(K, dim=0)
        w_disc = g_phi.sample(x_rep, z_rep)
        w_cont = f_theta.sample(1, w_disc, x_rep, z_rep)
        ll_d   = g_phi.log_prob(w_disc, x_rep, z_rep)
        ll_c   = f_theta.log_prob_no_grad(w_cont, w_disc, x_rep, z_rep)
        return {
            "w_disc":      w_disc.cpu(),
            "w_cont":      w_cont.cpu(),
            "w_cont_orig": _invert(w_cont.cpu()),
            "log_prob":    (ll_d + ll_c).cpu(),
            "x_rep":       x_rep.cpu(),
            "z_rep":       z_rep.cpu(),
        }

    @torch.no_grad()
    def log_prob_fn(w_disc, w_cont, x, z):
        g_phi.eval(); f_theta.eval()
        w_disc, w_cont, x, z = [t.to(device) for t in [w_disc, w_cont, x, z]]
        ll_d = g_phi.log_prob(w_disc, x, z)
        ll_c = f_theta.log_prob_no_grad(w_cont, w_disc, x, z)
        return (ll_d + ll_c).cpu()

    return sample_fn, log_prob_fn


# ── Outcome model (logistic regression, trained inline) ───────────────────────

def build_outcome_model(data: dict, scaler):
    """
    Fit a logistic regression on training data: [X, W_cont_orig, Z] → Y.
    Returns a callable (x, w_cont_std, z) → (N,) predicted bar-pass probabilities.
    The callable accepts standardised w_cont and inverts the scaler internally.
    """
    X_tr  = data["X_tr"].numpy()
    Z_tr  = data["Z_tr"].numpy()
    Wc_tr = data["Wc_tr"].numpy()
    Y_tr  = data["Y_tr"].numpy().astype(int)

    Wc_tr_orig = scaler.inverse_transform(Wc_tr) if scaler is not None else Wc_tr
    feat_tr    = np.concatenate([X_tr, Wc_tr_orig, Z_tr], axis=1)

    lr = LogisticRegression(max_iter=500, random_state=42)
    lr.fit(feat_tr, Y_tr)
    print(f"  Outcome model train accuracy: {lr.score(feat_tr, Y_tr):.3f}")

    def outcome_model(x: torch.Tensor, w_cont: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x_np  = x.cpu().numpy()
        z_np  = z.cpu().numpy()
        wc_np = w_cont.cpu().numpy()
        wc_orig = scaler.inverse_transform(wc_np) if scaler is not None else wc_np
        feat  = np.concatenate([x_np, wc_orig, z_np], axis=1)
        probs = lr.predict_proba(feat)[:, 1].astype(np.float32)
        return torch.tensor(probs)

    return outcome_model


# ── Plot 1: Training curve ────────────────────────────────────────────────────

def plot_training_curve(history: dict, save: bool = True):
    fig, ax = plt.subplots(figsize=(5.5, 3.0), constrained_layout=True)
    epochs  = range(1, len(history["train"]) + 1)
    ax.plot(epochs, history["train"], label="Train NLL", color=COL_GRP0, lw=LW)
    ax.plot(epochs, history["val"],   label="Val NLL",   color=COL_GRP1, lw=LW)
    if "val_flow" in history:
        ax.plot(epochs, history["val_flow"], label="Val flow", color=COL_GRP1,
                lw=LW, ls="--", alpha=0.6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$-\log p(W \mid X, Z)$")
    ax.legend(frameon=False)
    sns.despine(ax=ax)
    path = save_figure(fig, "bar_training_curve", save=save, save_dir=SAVE_DIR)
    if path:
        print(f"  Saved → {path}")
    plt.show()
    return fig


# ── Plot 5: K sensitivity ─────────────────────────────────────────────────────

def plot_k_sensitivity(
    X_val: torch.Tensor,
    Z_val: torch.Tensor,
    outcome_model,
    sample_fn,
    log_prob_fn,
    K_grid: list[int] = [500, 1000, 2000],
    n_inst: int = 100,
    save: bool = True,
):
    """
    For each K in K_grid, estimate NDE and NIE over n_inst validation individuals,
    then plot mean ± 1 SD across individuals as a function of K.
    No bootstrap — variability comes from individual-level heterogeneity.
    """
    idx    = torch.randperm(len(X_val))[:n_inst]
    X_sub  = X_val[idx]
    Z_sub  = Z_val[idx]

    nde_stats, nie_stats = [], []

    for K in K_grid:
        nde_vals, nie_vals = [], []
        for i in range(n_inst):
            xi = X_sub[i].unsqueeze(0)
            zi = Z_sub[i].unsqueeze(0)
            nde, nie = estimate_nde_nie_for_individual(
                xi, zi, outcome_model, sample_fn, log_prob_fn, K
            )
            nde_vals.append(nde)
            nie_vals.append(nie)

        nde_stats.append((np.mean(nde_vals), np.std(nde_vals)))
        nie_stats.append((np.mean(nie_vals), np.std(nie_vals)))
        print(f"  K={K:5d}  "
              f"NDE={nde_stats[-1][0]:+.4f} ± {nde_stats[-1][1]:.4f}  "
              f"NIE={nie_stats[-1][0]:+.4f} ± {nie_stats[-1][1]:.4f}")

    K_arr = np.array(K_grid)
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.8), constrained_layout=True)

    for ax, stats, ylabel, color in zip(
        axes,
        [nde_stats, nie_stats],
        [r"$\widehat{\mathrm{NDE}}$", r"$\widehat{\mathrm{NIE}}$"],
        [COL_NDE, COL_NIE],
    ):
        means = np.array([s[0] for s in stats])
        stds  = np.array([s[1] for s in stats])

        ax.plot(K_arr, means, color=color, lw=LW, marker="o", ms=MS, zorder=3)
        ax.fill_between(K_arr, means - stds, means + stds,
                        color=color, alpha=ALPHA_FILL, zorder=2)
        ax.axhline(means[-1], color="0.55", lw=1.0, ls="--", zorder=1)
        ax.set_xscale("log")
        ax.set_xticks(K_grid)
        ax.set_xticklabels([str(k) for k in K_grid])
        ax.set_xlabel(r"$K$")
        ax.set_ylabel(ylabel)
        sns.despine(ax=ax)

    path = save_figure(fig, "bar_k_sensitivity", save=save, save_dir=SAVE_DIR)
    if path:
        print(f"  Saved → {path}")
    plt.show()
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model ──
    g_phi, f_theta, scaler, sfm_cfg = load_flow_models(args.model, device)
    ckpt    = torch.load(args.model, map_location="cpu", weights_only=False)
    history = ckpt.get("history", {})
    print(f"Loaded from epoch {ckpt.get('best_epoch', '?')}  "
          f"val NLL = {ckpt.get('best_val_nll', float('nan')):.4f}")

    # ── Load tensors ──
    data  = torch.load(args.tensors, map_location="cpu", weights_only=False)
    X_va  = data["X_va"]
    Z_va  = data["Z_va"]
    Wc_va = data["Wc_va"]

    sample_fn, log_prob_fn = make_sample_fns(g_phi, f_theta, scaler, device)

    # ── 0. Sanity check: empirical marginal stats per group ──
    Wc_orig = torch.tensor(
        scaler.inverse_transform(Wc_va.numpy()).astype(np.float32)
    ) if scaler is not None else Wc_va

    print("\nSanity check — empirical marginal statistics per group:")
    print(f"  {'Group':12s}  {'Mediator':18s}  {'Mean':>8}  {'Std':>8}")
    for g, label in GROUP_LABELS.items():
        mask = X_va[:, 0] == g
        for j, name in enumerate(WC_NAMES):
            vals = Wc_orig[mask, j]
            print(f"  {label:12s}  {name:18s}  {vals.mean().item():8.3f}  {vals.std().item():8.3f}")

    # ── 1. Training curve ──
    if history:
        plot_training_curve(history)
    else:
        print("No training history in checkpoint — skipping curve.")

    # ── 2. Conditional marginals (empirical vs flow) ──
    plot_conditional_marginals(
        X_va, Z_va, Wc_va,
        wc_names=WC_NAMES,
        sample_fn=sample_fn,
        scaler=scaler,
        group_labels=GROUP_LABELS,
        K=50,
        n_subsample=200,
        save_dir=SAVE_DIR,
    )
    print(f"  Saved → {SAVE_DIR}/conditional_marginals.png")

    # ── 3. ESS by group + ESS/K < 0.05 fraction ──
    _, ess_by_grp = plot_ess_by_group(
        X_va, Z_va,
        sample_fn=sample_fn,
        log_prob_fn=log_prob_fn,
        group_labels=GROUP_LABELS,
        K=200,
        n_inst=200,
        save_dir=SAVE_DIR,
    )
    print(f"  Saved → {SAVE_DIR}/ess_by_group.png")

    print("\nESS/K < 0.05 fractions:")
    for g, label in GROUP_LABELS.items():
        ess_vals = ess_by_grp[g]
        frac = sum(e < 0.05 for e in ess_vals) / len(ess_vals)
        print(f"  {label:12s}: {frac:.1%}  ({sum(e < 0.05 for e in ess_vals)}/{len(ess_vals)} eval points)")

    # ── 4. IS weight stats ──
    print("\nIS weight stats (K=200, first 50 val individuals):")
    with torch.no_grad():
        x0  = X_va[:50]
        z0  = Z_va[:50]
        out = sample_fn(x0, z0, K=200)
        x1_rep = (1.0 - x0).repeat_interleave(200, dim=0)
        log_p1 = log_prob_fn(out["w_disc"], out["w_cont"], x1_rep, out["z_rep"])
        log_r  = log_p1 - out["log_prob"]
    print(f"  mean={log_r.mean():.3f}  std={log_r.std():.3f}  "
          f"min={log_r.min():.3f}  max={log_r.max():.3f}")

    # ── 5. K sensitivity (NDE / NIE) ──
    print("\nFitting outcome model (logistic regression on train data)...")
    outcome_model = build_outcome_model(data, scaler)

    print(f"\nK sensitivity — NDE / NIE (n_inst={args.n_inst}, K ∈ {args.k_grid}):")
    plot_k_sensitivity(
        X_va, Z_va,
        outcome_model=outcome_model,
        sample_fn=sample_fn,
        log_prob_fn=log_prob_fn,
        K_grid=args.k_grid,
        n_inst=args.n_inst,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model",   default="outputs/flows/law_school/flow_models.pt")
    p.add_argument("--tensors", default="outputs/data/bar_tensors.pt")
    p.add_argument("--k-grid",  type=int, nargs="+", default=[500, 1000, 2000],
                   help="K values for sensitivity analysis")
    p.add_argument("--n-inst",  type=int, default=100,
                   help="Number of val individuals for K sensitivity")
    main(p.parse_args())
