"""
inspect_recourse_shift.py — Recourse-induced distribution shift diagnostic.

For each mediator dimension W_j, plot three densities side-by-side on the
SAME Y=0 recourse-evaluation cohort:

  1. W | X=advantaged              (natural reference)
  2. W | X=disadvantaged           (natural, pre-recourse)
  3. W'_recourse | X=disadvantaged (post-recourse, at η = λ_EB)

Continuous mediators use a KDE; discrete mediators use a grouped bar chart.
One figure is produced per outcome model (λ_EB and the selected recourse
solution W' are both model-specific). The script reuses the exact Stage-4b
machinery — the shared candidate pool and its scored cache — so the third
series is guaranteed consistent with the sweep_lambda tables.

Usage
-----
  uv run python diagnostics/inspect_recourse_shift.py              # ACS (default)
  uv run python diagnostics/inspect_recourse_shift.py dataset=bar

Run from the project root. If the Stage-4b scored cache is absent it is
rebuilt (needs outputs/outcome/<ds>/{logreg,mlp}.joblib); otherwise no model
inference is performed.
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import torch
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

import hydra
from omegaconf import OmegaConf

# This script lives in diagnostics/ but imports the project's evaluation /
# outcome packages, which sit at the repo root. Put the repo root on sys.path
# so they resolve when the script is run directly. The flat paper_style import
# below resolves from this script's own directory, which stays on sys.path.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from paper_style import (
    COL_GRP0, COL_GRP1, COL_NIE,
    LW, ALPHA_KDE,
    set_paper_style, save_figure, legend_outside,
)
from evaluation.sweep_lambda import (
    select_eval_idx,
    enumerate_pool,
    score_pool,
    load_lambda_eb,
)
from evaluation.compute_recourse import compute_weights, select_best
from outcome.models import build_torch_pipeline

set_paper_style()

# Sensitive-attribute orientation. For ACS Income SEX is coded 0=Female,
# 1=Male; the outcome gap NIE = f(Male) − f(Female) is positive, so Male is
# the advantaged group and Female (the group that receives recourse) is the
# disadvantaged one. Mirrors GROUP_LABELS in inspect_acs_model.py.
ADVANTAGED_X    = 1
DISADVANTAGED_X = 0
SEX_LABELS      = {0: "Female", 1: "Male"}

# ACS category / display labels (mirror inspect_acs_model.py). Unknown
# mediators fall back to integer codes / the raw column name, so the script
# still runs on other datasets, just without pretty labels.
CATEGORY_LABELS = {
    "SCHL_GRP": [r"$<$HS", "HS", "Some col.", "Bachelor's", "Master's", "Doctoral+"],
    "OCCP_GRP": ["Mgmt", "Biz/Fin", "STEM", "STEM sup.", "Arts", "Health",
                 "Service", "Sales/Adm", "Constr./Prod.", "Transport/Other"],
}
MEDIATOR_DISPLAY = {
    "SCHL_GRP": "Education",
    "OCCP_GRP": "Occupation",
    "WKHP":     "Weekly hours",
}


def _fuzzy_leb(leb_by_model: dict, name: str):
    """Resolve λ_EB for a model spec name, tolerating spacing/case (matches
    the lookup convention in sweep_lambda.main)."""
    leb = leb_by_model.get(name)
    if leb is not None:
        return leb
    key = name.lower().replace(" ", "")
    for k, v in leb_by_model.items():
        if k.lower().replace(" ", "") == key:
            return v
    return None


def _slug(name: str) -> str:
    """Stage-4b file slug, e.g. 'Logistic Reg.' → 'logistic_reg'."""
    return (name.lower().replace(" ", "_").replace("(", "").replace(")", "")
            .replace("-", "").replace(".", ""))


def _reconstruct(data, scaler, eval_idx, all_candidates, sex_labels, leb,
                 disc_names, cont_names):
    """
    Walk the scored cohort and assemble, per mediator, the three series.

    Returns: dict name -> {"adv": np.array, "pre": np.array, "post": np.array}
    Discrete series hold integer category codes; continuous series hold
    natural-unit values. All three are drawn from the same Y=0 eval cohort.
    """
    Wd_va, Wc_va = data["Wd_va"], data["Wc_va"]

    series = {nm: {"adv": [], "pre": [], "post": []}
              for nm in disc_names + cont_names}

    for k, idx in enumerate(eval_idx):
        sex  = int(sex_labels[k])
        best = select_best(all_candidates[k], leb)

        wd_fact = Wd_va[idx].numpy()
        wc_raw  = Wc_va[idx].numpy()
        if scaler is not None and len(wc_raw) > 0:
            wc_fact = scaler.inverse_transform(wc_raw.reshape(1, -1))[0]
        else:
            wc_fact = wc_raw.astype(np.float32)

        for i, nm in enumerate(disc_names):
            fact  = int(wd_fact[i])
            prime = int(round(fact + best[f"delta_{nm}"]))
            if sex == ADVANTAGED_X:
                series[nm]["adv"].append(fact)
            elif sex == DISADVANTAGED_X:
                series[nm]["pre"].append(fact)
                series[nm]["post"].append(prime)

        for i, nm in enumerate(cont_names):
            fact  = float(wc_fact[i])
            prime = fact + float(best[f"delta_{nm}"])
            if sex == ADVANTAGED_X:
                series[nm]["adv"].append(fact)
            elif sex == DISADVANTAGED_X:
                series[nm]["pre"].append(fact)
                series[nm]["post"].append(prime)

    return {nm: {k: np.asarray(v) for k, v in d.items()}
            for nm, d in series.items()}


def _plot_model(series, disc_names, cont_names, vocab, mediator_specs,
                n_adv, n_dis, figures_dir, slug):
    """One figure for one model: a panel per mediator dimension."""
    adv_lbl  = SEX_LABELS[ADVANTAGED_X]
    dis_lbl  = SEX_LABELS[DISADVANTAGED_X]
    leg = [
        f"{adv_lbl} (natural ref., n={n_adv})",
        f"{dis_lbl} (natural, pre-recourse, n={n_dis})",
        rf"{dis_lbl} (recourse $W'$ @ $\eta=\lambda_{{\mathrm{{EB}}}}$)",
    ]
    cols   = [COL_GRP1, COL_GRP0, COL_NIE]
    names  = disc_names + cont_names
    n      = len(names)

    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 3.4))
    if n == 1:
        axes = [axes]

    for ax, nm in zip(axes, names):
        s = series[nm]
        if nm in disc_names:
            n_cats = vocab[nm]
            cats   = np.arange(n_cats)
            width  = 0.26
            offs   = (-width, 0.0, width)
            for (key, c, off) in zip(("adv", "pre", "post"), cols, offs):
                arr  = s[key]
                freq = (np.bincount(arr, minlength=n_cats) / max(len(arr), 1)
                        if len(arr) else np.zeros(n_cats))
                ax.bar(cats + off, freq[:n_cats], width=width,
                       color=c, alpha=0.85)
            labels = CATEGORY_LABELS.get(nm, [str(k) for k in range(n_cats)])
            ax.set_xticks(cats)
            ax.set_xticklabels(labels[:n_cats], rotation=35, ha="right",
                               fontsize=7)
            ax.set_ylabel("Frequency")
        else:
            clip = mediator_specs.get(nm, {}).get("clip", [None, None])
            for key, c in zip(("adv", "pre", "post"), cols):
                arr = s[key]
                if len(arr) > 1:
                    sns.kdeplot(arr, ax=ax, color=c, lw=LW, fill=True,
                                alpha=ALPHA_KDE, clip=tuple(clip),
                                common_norm=False)
            ax.set_ylabel("Density")

        ax.set_xlabel(MEDIATOR_DISPLAY.get(nm, nm))
        sns.despine(ax=ax)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.85) for c in cols]
    fig.subplots_adjust(right=0.80, bottom=0.22, wspace=0.30)
    legend_outside(fig, handles=handles, labels=leg, pad=0.81)

    path = save_figure(fig, f"recourse_shift_{slug}", save=True,
                        save_dir=figures_dir)
    if path:
        print(f"  Saved → {path}")
    plt.close(fig)
    return path


@hydra.main(config_path="../conf", config_name="config", version_base="1.1")
def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tensors_path = cfg.dataset.paths.tensors
    outcome_dir  = cfg.dataset.paths.outcome_dir
    gaps_dir     = cfg.dataset.paths.gaps_dir
    recourse_dir = cfg.dataset.paths.recourse_dir
    figures_dir  = cfg.dataset.paths.figures_dir
    dataset_name = cfg.dataset.name

    disc_names     = list(cfg.dataset.sfm.mediators_disc)
    cont_names     = list(cfg.dataset.sfm.mediators_cont)
    mediator_specs = OmegaConf.to_container(cfg.dataset.recourse.mediators,
                                            resolve=True)
    model_specs    = OmegaConf.to_container(cfg.dataset.outcome.models,
                                            resolve=True)
    recourse_cfg   = cfg.dataset.recourse
    type_to_slug   = {"logreg": "logreg", "mlp": "mlp"}

    if dataset_name != "acs":
        print(f"Note: category/sex labels are tuned for ACS; '{dataset_name}' "
              f"will use generic fallbacks.")

    gap_json = f"{gaps_dir}/{dataset_name}_gender_gap.json"
    print("Loading λ_EB:")
    leb_by_model = load_lambda_eb(gap_json)

    data    = torch.load(tensors_path, map_location="cpu", weights_only=False)
    scaler  = data["scaler"]
    vocab   = data["vocab"]
    weights = compute_weights(data, disc_names, cont_names)

    n_max_cfg = int(recourse_cfg.sweep_n_max)
    n_max     = n_max_cfg if n_max_cfg > 0 else 10**12
    threshold = float(recourse_cfg.threshold)
    nu        = float(recourse_cfg.nu)

    # Same model-independent eval cohort + candidate pool as Stage 4b, so the
    # cached scored pool (scored_<slug>_y0_n<n_max>_...pkl) is reused verbatim.
    eval_idx = select_eval_idx(data, n_max, rng_seed=0)
    pool     = enumerate_pool(data, eval_idx, scaler, disc_names, cont_names,
                              mediator_specs, vocab, weights)

    for spec in model_specs:
        name       = spec["name"]
        slug_model = type_to_slug.get(spec["type"], spec["type"])
        slug_file  = _slug(name)

        leb = _fuzzy_leb(leb_by_model, name)
        if leb is None:
            print(f"\nWARNING: no λ_EB for '{name}' — skipping.")
            continue

        print(f"\n{'='*60}\nModel: {name}   λ_EB = {leb:.4f}\n{'='*60}")
        pipe       = joblib.load(f"{outcome_dir}/{slug_model}.joblib")
        torch_pipe = build_torch_pipeline(pipe).to(device)

        all_candidates, sex_labels, _ = score_pool(
            torch_pipe, pool, threshold=threshold, nu=nu,
            slug=slug_file, save_dir=recourse_dir, n_max=n_max,
            use_cache=True, device=device,
        )

        series = _reconstruct(data, scaler, eval_idx, all_candidates,
                              sex_labels, leb, disc_names, cont_names)

        sex_arr = np.asarray(sex_labels)
        n_adv   = int((sex_arr == ADVANTAGED_X).sum())
        n_dis   = int((sex_arr == DISADVANTAGED_X).sum())
        print(f"  Cohort: {n_adv} {SEX_LABELS[ADVANTAGED_X]} (ref), "
              f"{n_dis} {SEX_LABELS[DISADVANTAGED_X]} (pre/post)")

        _plot_model(series, disc_names, cont_names, vocab, mediator_specs,
                    n_adv, n_dis, figures_dir, slug_file)

    print(f"\nAll recourse-shift figures → {figures_dir}/")


if __name__ == "__main__":
    main()
