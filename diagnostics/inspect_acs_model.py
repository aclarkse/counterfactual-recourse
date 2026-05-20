"""
inspect_acs_model.py — Diagnostic plots for the trained ACS discrete mediator model.

Equivalent of the bar-dataset flow diagnostics, adapted for all-discrete mediators:
  1. Training curve  (NLL over epochs)
  2. Empirical vs model distributions  (bar charts per mediator per group)
  3. Counterfactual shift  (P(W | X=Male) vs P(W | X=Female))
  4. ESS by group  (importance-sampling quality for NDE/NIE)
  5. IS weight stats  (numerical summary)

Usage
-----
  python diagnostics/inspect_acs_model.py
  python diagnostics/inspect_acs_model.py --model outputs/flows/acs/flow_models.pt
                                          --tensors outputs/data/acs_tensors.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns

# This script lives in diagnostics/ but imports the project's `flows` package,
# which sits at the repo root. Put the repo root on sys.path so it resolves when
# the script is run directly (python diagnostics/inspect_acs_model.py). The flat
# paper_style / flow_diagnostics_shared imports below resolve from this script's
# own directory, which stays on sys.path regardless.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from paper_style import (
    COL_GRP0, COL_GRP1, COL_REF,
    LW, ALPHA_FILL,
    set_paper_style, save_figure, legend_outside,
)
from flows.models import DiscreteMediator, ContinuousMediatorFlow, load_flow_models
from flow_diagnostics_shared import plot_ess_by_group, plot_conditional_marginals

set_paper_style()

GROUP_LABELS = {0: "Female", 1: "Male"}

SCHL_LABELS  = [r"$<$HS", "HS", "Some col.", "Bachelor's", "Master's", "Doctoral+"]
OCCP_LABELS  = ["Mgmt", "Biz/Fin", "STEM", "STEM sup.", "Arts", "Health",
                "Service", "Sales/Adm", "Constr./Prod.", "Transport/Other"]

CATEGORY_LABELS = {
    "SCHL_GRP": SCHL_LABELS,
    "OCCP_GRP": OCCP_LABELS,
}

MEDIATOR_DISPLAY = {
    "SCHL_GRP": "Education",
    "OCCP_GRP": "Occupation",
}

WKHP_CLIP = (1, 60)


# ── Sampling utilities (mirroring the notebook) ───────────────────────────────

def make_sample_fns(g_phi, f_theta, scaler, device):
    """Return sample_w_given_xz and log_prob_w_given_xz closures."""

    def _to_original_scale(w_cont: torch.Tensor) -> torch.Tensor:
        """Invert QuantileTransform → clip to WKHP_CLIP → round to nearest hour."""
        if scaler is None or w_cont.shape[1] == 0:
            return w_cont
        arr = scaler.inverse_transform(w_cont.numpy())
        arr = np.clip(arr, *WKHP_CLIP)
        arr = np.round(arr).astype(np.float32)
        return torch.tensor(arr)

    @torch.no_grad()
    def sample_w_given_xz(x, z, K=500):
        g_phi.eval(); f_theta.eval()
        x, z = x.to(device), z.to(device)
        x_rep = x.repeat_interleave(K, dim=0)
        z_rep = z.repeat_interleave(K, dim=0)
        w_disc_s = g_phi.sample(x_rep, z_rep)
        w_cont_s = f_theta.sample(1, w_disc_s, x_rep, z_rep)
        ll_disc  = g_phi.log_prob(w_disc_s, x_rep, z_rep)
        ll_cont  = f_theta.log_prob_no_grad(w_cont_s, w_disc_s, x_rep, z_rep)
        return {
            "w_disc":      w_disc_s.cpu(),
            "w_cont":      w_cont_s.cpu(),
            "w_cont_orig": _to_original_scale(w_cont_s.cpu()),
            "log_prob":    (ll_disc + ll_cont).cpu(),
            "x_rep":       x_rep.cpu(),
            "z_rep":       z_rep.cpu(),
        }

    @torch.no_grad()
    def log_prob_w_given_xz(w_disc, w_cont, x, z):
        g_phi.eval(); f_theta.eval()
        w_disc, w_cont, x, z = [t.to(device) for t in [w_disc, w_cont, x, z]]
        ll_disc = g_phi.log_prob(w_disc, x, z)
        ll_cont = f_theta.log_prob_no_grad(w_cont, w_disc, x, z)
        return (ll_disc + ll_cont).cpu()

    return sample_w_given_xz, log_prob_w_given_xz


# ── Plot 1: Training curve ────────────────────────────────────────────────────

def plot_training_curve(history: dict, save: bool = True):
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    epochs = range(1, len(history["train"]) + 1)
    ax.plot(epochs, history["train"], label="Train NLL", color=COL_GRP0, lw=LW)
    ax.plot(epochs, history["val"],   label="Val NLL",   color=COL_GRP1, lw=LW)
    if "val_disc" in history:
        ax.plot(epochs, history["val_disc"], label="Val disc", color=COL_GRP0,
                lw=LW, ls="--", alpha=0.6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$-\log p(W \mid X, Z)$")
    sns.despine(ax=ax)
    legend_outside(ax)
    path = save_figure(fig, "acs_training_curve", save=save, save_dir="figures/acs_income")
    if path:
        print(f"  Saved → {path}")
    plt.show()
    return fig


# ── Plot 2: Empirical vs model — bar charts per mediator per group ────────────

@torch.no_grad()
def _group_model_probs(X_val, Z_val, g_phi, device):
    """
    For each group g in {0, 1}, compute the average softmax probability
    vector for each discrete mediator column: P(W_j = k | x_i, z_i) averaged
    over all individuals i with X_i = g.

    Returns dict: {col_name: {g: np.ndarray of shape (n_cats,)}}
    """
    g_phi.eval()
    probs = {col: {0: None, 1: None} for col in g_phi.col_names}

    for g in [0, 1]:
        mask = (X_val[:, 0] == g)
        x_g  = X_val[mask].to(device)
        z_g  = Z_val[mask].to(device)
        h    = g_phi.backbone(g_phi._cond(x_g, z_g))
        for col, head in zip(g_phi.col_names, g_phi.heads):
            p = torch.softmax(head(h), dim=-1).mean(dim=0).cpu().numpy()
            probs[col][g] = p

    return probs


def plot_discrete_marginals(
    X_val: torch.Tensor,
    Z_val: torch.Tensor,
    Wd_val: torch.Tensor,
    vocab: dict,
    g_phi,
    device,
    save: bool = True,
):
    """
    1-row × n_mediators grid.
    Each subplot overlays empirical frequencies and model probabilities for
    both groups using hue (Female/Male color) × style (hatched=empirical,
    solid=model) to make all four distributions distinguishable.
    """
    col_names   = list(vocab.keys())
    n_cols      = len(col_names)
    model_probs = _group_model_probs(X_val, Z_val, g_phi, device)
    colors      = [COL_GRP0, COL_GRP1]
    width       = 0.18
    gap         = 0.04   # extra space between Female and Male bar pairs

    fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 3.4))
    if n_cols == 1:
        axes = [axes]

    for col_idx, (ax, col) in enumerate(zip(axes, col_names)):
        n_cats     = vocab[col]
        cats       = np.arange(n_cats)
        cat_labels = CATEGORY_LABELS.get(col, [str(k) for k in range(n_cats)])

        for g, color in zip([0, 1], colors):
            sign = -1 if g == 0 else 1
            mask       = (X_val[:, 0] == g)
            obs_counts = np.bincount(Wd_val[mask, col_idx].numpy(), minlength=n_cats)
            obs_freq   = obs_counts / obs_counts.sum()
            mod_prob   = model_probs[col][g]

            # empirical bar — outer position, hatched
            x_emp = cats + sign * (1.5 * width + gap / 2)
            ax.bar(x_emp, obs_freq, width=width,
                   color=color, alpha=0.35,
                   hatch="///", edgecolor=color, linewidth=0.5)

            # model bar — inner position, solid
            x_mod = cats + sign * (0.5 * width + gap / 2)
            ax.bar(x_mod, mod_prob, width=width,
                   color=color, alpha=0.80)

        ax.set_xticks(cats)
        ax.set_xticklabels(cat_labels[:n_cats], rotation=35, ha="right", fontsize=7)
        ax.set_xlabel(MEDIATOR_DISPLAY.get(col, col))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        if col_idx == 0:
            ax.set_ylabel(r"$P(W_j = k \mid \mathrm{Sex})$")
        sns.despine(ax=ax)

    legend_handles = [
        mpatches.Patch(facecolor=COL_GRP0, alpha=0.35, hatch="///",
                       edgecolor=COL_GRP0, label="Female -- empirical"),
        mpatches.Patch(facecolor=COL_GRP0, alpha=0.80,
                       label="Female -- model"),
        mpatches.Patch(facecolor=COL_GRP1, alpha=0.35, hatch="///",
                       edgecolor=COL_GRP1, label="Male -- empirical"),
        mpatches.Patch(facecolor=COL_GRP1, alpha=0.80,
                       label="Male -- model"),
    ]
    fig.subplots_adjust(right=0.82, bottom=0.22, wspace=0.25)
    legend_outside(fig, handles=legend_handles, pad=0.83)

    path = save_figure(fig, "acs_discrete_marginals", save=save, save_dir="figures/acs_income")
    if path:
        print(f"  Saved → {path}")
    plt.show()
    return fig


# ── Plot 3: Counterfactual shift ──────────────────────────────────────────────

@torch.no_grad()
def plot_counterfactual_shift(
    X_val: torch.Tensor,
    Z_val: torch.Tensor,
    vocab: dict,
    g_phi,
    device,
    save: bool = True,
):
    """
    For each discrete mediator, show the distribution shift induced by
    flipping X:  P(W | X=Female) vs P(W | X=Male), averaged over all Z_val.

    This is the core counterfactual quantity used in NDE/NIE estimation.
    """
    g_phi.eval()
    col_names = list(vocab.keys())
    n_cols    = len(col_names)

    # Average over the full val set, forcing X=0 and X=1 respectively
    N = X_val.shape[0]
    x0 = torch.zeros(N, X_val.shape[1])
    x1 = torch.ones( N, X_val.shape[1])
    Z  = Z_val

    x0_d, x1_d, Z_d = x0.to(device), x1.to(device), Z.to(device)
    h0 = g_phi.backbone(g_phi._cond(x0_d, Z_d))
    h1 = g_phi.backbone(g_phi._cond(x1_d, Z_d))

    fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 3.2))
    if n_cols == 1:
        axes = [axes]

    last_ax2 = None
    for ax, (col_idx, (col, head)) in zip(axes, enumerate(zip(col_names, g_phi.heads))):
        n_cats = vocab[col]
        cats   = np.arange(n_cats)
        width  = 0.35
        cat_labels = CATEGORY_LABELS.get(col, [str(k) for k in range(n_cats)])

        p0 = torch.softmax(head(h0), dim=-1).mean(dim=0).cpu().numpy()
        p1 = torch.softmax(head(h1), dim=-1).mean(dim=0).cpu().numpy()

        ax.bar(cats - width / 2, p0, width=width, color=COL_GRP0, alpha=0.75,
               label=GROUP_LABELS[0] if col_idx == 0 else None)
        ax.bar(cats + width / 2, p1, width=width, color=COL_GRP1, alpha=0.75,
               label=GROUP_LABELS[1] if col_idx == 0 else None)

        # Difference line (right axis)
        ax2 = ax.twinx()
        diff = p1 - p0
        ax2.plot(cats, diff, color=COL_REF, lw=LW, marker="o", ms=3,
                 label=r"$\Delta P$ (Male $-$ Female)" if col_idx == n_cols - 1 else None)
        ax2.axhline(0, color=COL_REF, lw=0.8, ls="--")
        ax2.set_ylabel(r"$\Delta P(W_j = k)$" if col_idx == n_cols - 1 else "")
        ax2.tick_params(labelsize=7)
        last_ax2 = ax2

        ax.set_xticks(cats)
        ax.set_xticklabels(cat_labels[:n_cats], rotation=35, ha="right", fontsize=7)
        ax.set_xlabel(MEDIATOR_DISPLAY.get(col, col))
        ax.set_ylabel(r"$P(W_j = k \mid \mathrm{Sex})$" if col_idx == 0 else "")
        sns.despine(ax=ax, right=False)

    h_bars,  l_bars  = axes[0].get_legend_handles_labels()
    h_lines, l_lines = last_ax2.get_legend_handles_labels()
    # Right twinx needs ~0.07 of figure width for its ylabel + ticks; place
    # the legend just past that, not at the figure's right edge.
    fig.subplots_adjust(right=0.78, bottom=0.22, wspace=0.30)
    legend_outside(fig,
                   handles=h_bars + h_lines,
                   labels=l_bars + l_lines,
                   pad=0.86)
    path = save_figure(fig, "acs_counterfactual_shift", save=save, save_dir="figures/acs_income")
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
    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    vocab   = ckpt["g_phi_cfg"]["vocab"]
    history = ckpt.get("history", {})
    print(f"Loaded from epoch {ckpt.get('best_epoch', '?')}  "
          f"val NLL = {ckpt.get('best_val_nll', float('nan')):.4f}")

    # ── Load tensors ──
    data = torch.load(args.tensors, map_location="cpu", weights_only=False)
    X_va  = data["X_va"]
    Z_va  = data["Z_va"]
    Wd_va = data["Wd_va"]
    Wc_va = data["Wc_va"]   # standardised WKHP (N(0,1) space)

    sample_fn, log_prob_fn = make_sample_fns(g_phi, f_theta, scaler, device)

    # ── 0. Sanity check: conditional distributions must sum to 1 ──
    print("\nSanity check — probability sums per mediator per group "
          "(should all be ≈ 1.0, diff ≈ 0.0):")
    model_probs = _group_model_probs(X_va, Z_va, g_phi, device)
    all_ok = True
    for col, probs in model_probs.items():
        s0, s1 = probs[0].sum(), probs[1].sum()
        diff_sum = (probs[1] - probs[0]).sum()
        ok = abs(s0 - 1.0) < 1e-4 and abs(s1 - 1.0) < 1e-4
        all_ok = all_ok and ok
        flag = "" if ok else "  ← FAIL"
        print(f"  {col:12s}  Female sum={s0:.6f}  Male sum={s1:.6f}"
              f"  diff sum={diff_sum:+.2e}{flag}")
    print("  All OK." if all_ok else "  WARNING: some distributions do not sum to 1!")

    # ── 1. Training curve ──
    if history:
        plot_training_curve(history)
    else:
        print("No training history in checkpoint — skipping curve.")

    # ── 2a. Empirical vs model — discrete mediators ──
    plot_discrete_marginals(X_va, Z_va, Wd_va, vocab, g_phi, device)

    # ── 2b. Empirical vs model — continuous mediator (WKHP) ──
    plot_conditional_marginals(
        X_va, Z_va, Wc_va,
        wc_names=["Hours worked / week"],
        sample_fn=sample_fn,
        scaler=scaler,
        group_labels=GROUP_LABELS,
        K=50,
        n_subsample=200,
        save_dir="figures/acs_income",
    )

    # ── 3. Counterfactual shift ──
    plot_counterfactual_shift(X_va, Z_va, vocab, g_phi, device)

    # ── 4. ESS by group ──
    plot_ess_by_group(
        X_va, Z_va,
        sample_fn=sample_fn,
        log_prob_fn=log_prob_fn,
        group_labels=GROUP_LABELS,
        K=200,
        n_inst=200,
        save_dir="figures/acs_income",
    )
    print("  Saved → figures/acs_income/ess_by_group.png")

    # ── 5. IS weight stats ──
    print("\nIS weight stats (K=200, first 50 val individuals):")
    with torch.no_grad():
        x0   = X_va[:50]
        z0   = Z_va[:50]
        K    = 200
        out  = sample_fn(x0, z0, K=K)
        x1_rep = (1.0 - x0).repeat_interleave(K, dim=0)
        log_p1 = log_prob_fn(out["w_disc"], out["w_cont"], x1_rep, out["z_rep"])
        log_r  = log_p1 - out["log_prob"]
    print(f"  mean={log_r.mean():.3f}  std={log_r.std():.3f}  "
          f"min={log_r.min():.3f}  max={log_r.max():.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model",   default="outputs/flows/acs/flow_models.pt")
    p.add_argument("--tensors", default="outputs/data/acs_tensors.pt")
    main(p.parse_args())
