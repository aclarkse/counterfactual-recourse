import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import torch


# ═══════════════════════════════════════════════════════════════════════════════
# Global style
# ═══════════════════════════════════════════════════════════════════════════════

sns.set_theme(
    style="white",
    context="paper",
    palette="colorblind",
    rc={
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 120,     # screen display
        "savefig.dpi": 300,    # export quality
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    },
)

PALETTE = sns.color_palette("colorblind")
COL_GRP0 = PALETTE[0]          # minority  (race = 0 / Black)
COL_GRP1 = PALETTE[1]          # majority  (race = 1 / White)
COL_NDE = PALETTE[2]
COL_NIE = PALETTE[3]
COL_REF = "0.55"

ALPHA_KDE = 0.25
ALPHA_FILL = 0.20
LW = 1.6
MS = 4.0

SAVE_DIR = "figures"
SAVE_EXT = "png"
os.makedirs(SAVE_DIR, exist_ok=True)

GROUP_LABELS = {0: "Black", 1: "White"}   # adjust per dataset


def _save_figure(fig, filename: str, save: bool = True, dpi: int = 300):
    """Save figure as high-resolution PNG (or chosen extension)."""
    if not save:
        return
    path = os.path.join(SAVE_DIR, f"{filename}.{SAVE_EXT}")
    fig.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONDITIONAL MARGINALS  P(W_cont | X=x, Z)  stratified by group
# ═══════════════════════════════════════════════════════════════════════════════

def plot_conditional_marginals(
    X_val: torch.Tensor,
    Z_val: torch.Tensor,
    Wc_val: torch.Tensor,
    wc_names: list[str],
    sample_fn,          # pass sample_w_given_xz
    scaler=None,
    K: int = 50,
    n_subsample: int = 200,
    save: bool = True,
):
    """
    For each continuous mediator, plot KDE of:
      - observed data  (both groups, dashed)
      - flow samples   (both groups, solid filled)
    stratified by X (sensitive attribute, assumed binary in column 0).
    All values are plotted on the original (un-standardized) scale.
    """

    if scaler is not None:
        Wc_val_orig = torch.tensor(
            scaler.inverse_transform(Wc_val.detach().cpu().numpy()),
            dtype=torch.float32,
        )
    else:
        Wc_val_orig = Wc_val.detach().cpu()

    X_val_cpu = X_val.detach().cpu()
    Z_val_cpu = Z_val.detach().cpu()

    groups = [0, 1]
    n_features = len(wc_names)

    fig, axes = plt.subplots(
        1,
        n_features,
        figsize=(3.0 * n_features, 2.8),
        sharey=False,
        constrained_layout=True,
    )
    if n_features == 1:
        axes = [axes]

    for j, (ax, feat) in enumerate(zip(axes, wc_names)):
        for g, color in zip(groups, [COL_GRP0, COL_GRP1]):
            mask = X_val_cpu[:, 0] == g

            # Observed data
            obs = Wc_val_orig[mask, j].numpy()
            sns.kdeplot(
                obs,
                ax=ax,
                color=color,
                lw=LW,
                linestyle="--",
                fill=False,
                label=f"{GROUP_LABELS[g]} (data)" if j == 0 else None,
                common_norm=False,
            )

            # Flow samples
            x_g = X_val[mask][:n_subsample]
            z_g = Z_val[mask][:n_subsample]

            with torch.no_grad():
                out = sample_fn(x_g, z_g, K=K)

            samp = out["w_cont_orig"][:, j].detach().cpu().numpy()
            sns.kdeplot(
                samp,
                ax=ax,
                color=color,
                lw=LW,
                linestyle="-",
                fill=True,
                alpha=ALPHA_KDE,
                label=f"{GROUP_LABELS[g]} (flow)" if j == 0 else None,
                common_norm=False,
            )

        ax.set_xlabel(feat)
        ax.set_ylabel("Density" if j == 0 else "")
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        sns.despine(ax=ax)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, frameon=False, loc="best")

    _save_figure(fig, "conditional_marginals", save=save)
    plt.show()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 2. EFFECTIVE SAMPLE SIZE  (ESS)  stratified by group
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ess(log_w: torch.Tensor) -> float:
    """ESS from log importance weights (numerically stable)."""
    log_w = log_w - log_w.max()
    w = torch.exp(log_w)
    return float((w.sum() ** 2) / (w.pow(2).sum()))


def plot_ess_by_group(
    X_val: torch.Tensor,
    Z_val: torch.Tensor,
    sample_fn,          # pass sample_w_given_xz
    log_prob_fn,        # pass log_prob_w_given_xz
    K: int = 500,
    n_inst: int = 200,
    save: bool = True,
):
    """
    For each individual in a validation subsample, compute ESS of IS weights
    r_k ∝ p(w|x',z) / p(w|x,z), where x' is the flipped sensitive attribute.
    Plot ESS/K distribution as a boxplot stratified by group.
    """
    groups = [0, 1]
    ess_by_grp = {g: [] for g in groups}

    for g in groups:
        mask = (X_val[:, 0] == g).nonzero(as_tuple=True)[0]
        subset = mask[:n_inst]

        for idx in subset:
            xi = X_val[idx].unsqueeze(0)
            zi = Z_val[idx].unsqueeze(0)

            with torch.no_grad():
                out = sample_fn(xi, zi, K=K)
                log_p0 = out["log_prob"]

                xi_cf = 1.0 - xi
                xi_rep = xi_cf.repeat_interleave(K, dim=0)
                zi_rep = out["z_rep"]

                log_p1 = log_prob_fn(
                    out["w_disc"],
                    out["w_cont"],
                    xi_rep,
                    zi_rep,
                )

                log_r = log_p1 - log_p0
                ess = compute_ess(log_r) / K

            ess_by_grp[g].append(ess)

    fig, ax = plt.subplots(figsize=(3.4, 2.9), constrained_layout=True)

    data = [ess_by_grp[g] for g in groups]
    labels = [GROUP_LABELS[g] for g in groups]
    colors = [COL_GRP0, COL_GRP1]

    bp = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.50,
        medianprops=dict(color="white", lw=2),
        whiskerprops=dict(lw=LW),
        capprops=dict(lw=LW),
        flierprops=dict(marker="o", ms=3, alpha=0.45, lw=0),
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.80)

    for flier, color in zip(bp["fliers"], colors):
        flier.set_markerfacecolor(color)
        flier.set_markeredgecolor(color)

    ax.axhline(
        0.1,
        color=COL_REF,
        lw=1.0,
        ls="--",
        label="ESS / K = 0.1",
    )
    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels)
    ax.set_ylabel("ESS / K")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="best")
    sns.despine(ax=ax)

    _save_figure(fig, "ess_by_group", save=save)
    plt.show()
    return fig, ess_by_grp


# ═══════════════════════════════════════════════════════════════════════════════
# 3. K-SENSITIVITY  —  NDE / NIE stability as a function of K
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_nde_nie_for_individual(
    xi: torch.Tensor,         # (1, dim_x)
    zi: torch.Tensor,         # (1, dim_z)
    outcome_model,           # callable (x, w_cont, z) -> (N,) predicted probs
    sample_fn,               # pass sample_w_given_xz
    log_prob_fn,             # pass log_prob_w_given_xz
    K: int,
    x0_val: float = 0.0,
    x1_val: float = 1.0,
) -> tuple[float, float]:
    """
    Monte Carlo NDE / NIE for a single individual using K samples.
    Returns (NDE, NIE).
    """
    with torch.no_grad():
        out = sample_fn(xi, zi, K=K)
        wc = out["w_cont"]      # (K, dim_wc), standardized
        z_rep = out["z_rep"]

        x_rep = xi.repeat_interleave(K, dim=0)
        x0_rep = torch.full_like(x_rep, x0_val)
        x1_rep = torch.full_like(x_rep, x1_val)

        f_x1_w = outcome_model(x1_rep, wc, z_rep)
        f_x0_w = outcome_model(x0_rep, wc, z_rep)

        nde = float((f_x1_w - f_x0_w).mean())

        log_p0 = out["log_prob"]
        log_p1 = log_prob_fn(
            out["w_disc"],
            out["w_cont"],
            x1_rep,
            z_rep,
        )

        log_r = log_p1 - log_p0
        log_r = log_r - log_r.max()
        r = torch.exp(log_r)
        r_bar = r / r.sum()

        nie = float((r_bar * f_x0_w).sum() - f_x0_w.mean())

    return nde, nie


def plot_k_sensitivity(
    X_val: torch.Tensor,
    Z_val: torch.Tensor,
    outcome_model,
    sample_fn,                # pass sample_w_given_xz
    log_prob_fn,              # pass log_prob_w_given_xz
    K_grid: list[int] = [50, 100, 200, 500, 1000],
    n_inst: int = 100,
    n_bootstrap: int = 200,
    save: bool = True,
):
    """
    For each K in K_grid, estimate NDE and NIE over n_inst individuals
    using bootstrap resampling, then plot mean ± 1 SD.
    """
    idx = torch.randperm(len(X_val))[:n_inst]
    X_sub = X_val[idx]
    Z_sub = Z_val[idx]

    nde_stats = []
    nie_stats = []

    for K in K_grid:
        nde_boot, nie_boot = [], []

        for _ in range(n_bootstrap):
            boot_idx = np.random.choice(n_inst, n_inst, replace=True)
            nde_vals, nie_vals = [], []

            for i in boot_idx:
                xi = X_sub[i].unsqueeze(0)
                zi = Z_sub[i].unsqueeze(0)
                nde, nie = estimate_nde_nie_for_individual(
                    xi, zi, outcome_model, sample_fn, log_prob_fn, K
                )
                nde_vals.append(nde)
                nie_vals.append(nie)

            nde_boot.append(np.mean(nde_vals))
            nie_boot.append(np.mean(nie_vals))

        nde_stats.append((np.mean(nde_boot), np.std(nde_boot)))
        nie_stats.append((np.mean(nie_boot), np.std(nie_boot)))

        print(
            f"K={K:5d}  "
            f"NDE={nde_stats[-1][0]:.4f}±{nde_stats[-1][1]:.4f}  "
            f"NIE={nie_stats[-1][0]:.4f}±{nie_stats[-1][1]:.4f}"
        )

    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.8), constrained_layout=True)
    K_arr = np.array(K_grid)

    for ax, stats, ylabel, color in zip(
        axes,
        [nde_stats, nie_stats],
        [r"$\widehat{\mathrm{NDE}}$", r"$\widehat{\mathrm{NIE}}$"],
        [COL_NDE, COL_NIE],
    ):
        means = np.array([s[0] for s in stats])
        stds = np.array([s[1] for s in stats])

        ax.plot(
            K_arr,
            means,
            color=color,
            lw=LW,
            marker="o",
            ms=MS,
            zorder=3,
        )
        ax.fill_between(
            K_arr,
            means - stds,
            means + stds,
            color=color,
            alpha=ALPHA_FILL,
            zorder=2,
        )
        ax.axhline(means[-1], color=COL_REF, lw=1.0, ls="--", zorder=1)

        ax.set_xscale("log")
        ax.set_xticks(K_grid)
        ax.set_xticklabels([str(k) for k in K_grid], rotation=30, ha="right")
        ax.set_xlabel(r"$K$")
        ax.set_ylabel(ylabel)
        sns.despine(ax=ax)

    _save_figure(fig, "k_sensitivity", save=save)
    plt.show()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Example usage
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    wc_names = cfg["mediators_cont"]

    # fig_marg = plot_conditional_marginals(
    #     X_va, Z_va, Wc_va, wc_names,
    #     sample_fn=sample_w_given_xz,
    #     scaler=scaler,
    #     K=50,
    # )

    # fig_ess, ess_data = plot_ess_by_group(
    #     X_va, Z_va,
    #     sample_fn=sample_w_given_xz,
    #     log_prob_fn=log_prob_w_given_xz,
    #     K=500,
    #     n_inst=200,
    # )

    # fig_k = plot_k_sensitivity(
    #     X_va, Z_va,
    #     outcome_model=your_model,
    #     sample_fn=sample_w_given_xz,
    #     log_prob_fn=log_prob_w_given_xz,
    #     K_grid=[50, 100, 200, 500, 1000],
    # )