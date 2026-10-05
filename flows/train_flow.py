"""
flows/train_flow.py — Generic Hydra entry point for flow training.

Run: python -m flows.train_flow [dataset=acs|adult|bar|german]
     python -m flows.train_flow dataset=acs dataset.loader.kwargs.year=2019
"""

import importlib
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

import hydra
from omegaconf import OmegaConf

from data.build_tensors import build_tensors
from flows.models import (
    ConditionalGaussian,
    ContinuousMediatorFlow,
    DiscreteMediator,
    combined_loss,
    evaluate,
)
from flows.schema import MediatorSchema
from flows.posterior_predictive import mediator_posterior_predictive


_ROLE_KEYS = (
    "sensitive", "confounders", "mediators_disc", "mediators_cont", "outcome"
)


def _tensor_schema_matches(saved, sfm_cfg, split_cfg):
    saved_cfg = saved.get("cfg", {})
    legacy_split = {"test_size": 0.2, "stratify": False}
    return (
        all(saved_cfg.get(key) == sfm_cfg.get(key) for key in _ROLE_KEYS)
        and saved.get("split", legacy_split) == split_cfg
    )


def _confounder_standardization(Z_tr, sfm_cfg):
    """Fit normalization only for explicitly configured flow confounders."""
    names = list(sfm_cfg["confounders"])
    selected = set(sfm_cfg.get("standardize_confounders", []))
    unknown = selected - set(names)
    if unknown:
        raise ValueError(
            "Unknown standardize_confounders entries: {}".format(
                sorted(unknown)
            )
        )
    mean = torch.zeros(len(names), dtype=torch.float32)
    scale = torch.ones(len(names), dtype=torch.float32)
    for index, name in enumerate(names):
        if name not in selected:
            continue
        mean[index] = Z_tr[:, index].mean()
        scale[index] = Z_tr[:, index].std(unbiased=False)
        if scale[index] <= 1e-8:
            raise ValueError(f"Cannot standardize constant confounder {name}.")
    return mean.tolist(), scale.tolist()


_PPC_LIMITS = (
    ("max_overall_marginal_tv", "max_overall_marginal_tv"),
    ("max_group_marginal_tv", "max_group_marginal_tv"),
    ("max_ordered_wasserstein", "max_ordered_wasserstein"),
    ("max_overall_continuous_w1", "max_overall_continuous_w1"),
    ("max_group_continuous_w1", "max_group_continuous_w1"),
    (
        "max_continuous_correlation_error",
        "max_continuous_correlation_error",
    ),
)


def _ppc_is_eligible(diagnostics, selection_cfg):
    return all(
        float(diagnostics[metric]) <= float(selection_cfg[limit])
        for metric, limit in _PPC_LIMITS
        if limit in selection_cfg
    )


def _ppc_penalty(diagnostics, selection_cfg):
    return sum(
        float(diagnostics[metric])
        for metric, limit in _PPC_LIMITS
        if limit in selection_cfg
    )


@hydra.main(config_path="../conf", config_name="config", version_base="1.1")
def main(cfg):
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    sfm_cfg = OmegaConf.to_container(cfg.dataset.sfm, resolve=True)
    configured_layers = sfm_cfg.get("mediator_layers")
    if configured_layers is None:
        configured_layers = [[name] for name in sfm_cfg["mediators_disc"]]
        if sfm_cfg["mediators_cont"]:
            configured_layers.append(list(sfm_cfg["mediators_cont"]))
        warnings.warn(
            "No sfm.mediator_layers configured; using discrete singleton "
            "layers followed by one joint continuous block.",
            stacklevel=2,
        )
        sfm_cfg["mediator_layers"] = configured_layers
    mediator_schema = MediatorSchema.build(
        configured_layers,
        sfm_cfg["mediators_disc"],
        sfm_cfg["mediators_cont"],
    )
    wc_clip = list(cfg.dataset.wc_clip) if cfg.dataset.wc_clip is not None else None
    flow_cfg = cfg.dataset.flow
    tensors_path = cfg.dataset.paths.tensors
    raw_split_cfg = cfg.dataset.get("split")
    split_cfg = (
        OmegaConf.to_container(raw_split_cfg, resolve=True)
        if raw_split_cfg is not None else {}
    )
    split_cfg = {"test_size": split_cfg.get("test_size", 0.2),
                 "stratify": bool(split_cfg.get("stratify", False))}
    reuse_tensors = bool(flow_cfg.get("reuse_tensors", True))
    if reuse_tensors and os.path.exists(tensors_path):
        print(f"Reusing fixed data split from {tensors_path}")
        saved = torch.load(tensors_path, map_location="cpu", weights_only=False)
        if not _tensor_schema_matches(saved, sfm_cfg, split_cfg):
            raise RuntimeError(
                "Cached tensors do not match the configured SFM roles. "
                "Rerun with dataset.flow.reuse_tensors=false to rebuild "
                "them before training."
            )
        X_tr, Z_tr, Wd_tr, Wc_tr, Y_tr = (
            saved["X_tr"], saved["Z_tr"], saved["Wd_tr"], saved["Wc_tr"],
            saved["Y_tr"],
        )
        X_va, Z_va, Wd_va, Wc_va, Y_va = (
            saved["X_va"], saved["Z_va"], saved["Wd_va"], saved["Wc_va"],
            saved["Y_va"],
        )
        scaler, vocab = saved["scaler"], saved["vocab"]
    else:
        loader_module = importlib.import_module(cfg.dataset.loader.module)
        loader_fn = getattr(loader_module, cfg.dataset.loader.fn)
        kwargs = OmegaConf.to_container(cfg.dataset.loader.kwargs, resolve=True)
        df_raw = loader_fn(**kwargs)
        stratify = (
            df_raw[sfm_cfg["outcome"]] if split_cfg["stratify"] else None
        )
        idx_tr, idx_val = train_test_split(
            np.arange(len(df_raw)), test_size=split_cfg["test_size"],
            random_state=int(cfg.seed), stratify=stratify,
        )
        df_tr = df_raw.iloc[idx_tr].reset_index(drop=True)
        df_val = df_raw.iloc[idx_val].reset_index(drop=True)
        X_tr, Z_tr, Wd_tr, Wc_tr, Y_tr, scaler, vocab = build_tensors(
            df_tr, sfm_cfg, fit_scaler=True, wc_clip=wc_clip
        )
        X_va, Z_va, Wd_va, Wc_va, Y_va, _, _ = build_tensors(
            df_val, sfm_cfg, scaler=scaler, fit_scaler=False, wc_clip=wc_clip
        )

    dim_x  = X_tr.shape[1]
    dim_z  = Z_tr.shape[1]
    dim_wc = Wc_tr.shape[1]
    dim_wd = Wd_tr.shape[1]

    print(f"Train: {len(X_tr):,}  Val: {len(X_va):,}")
    print(f"dim_x={dim_x}  dim_z={dim_z}  dim_W_disc={dim_wd}  dim_W_cont={dim_wc}")
    print(f"Vocab: {vocab}")
    z_mean, z_scale = _confounder_standardization(Z_tr, sfm_cfg)
    standardized_z = list(sfm_cfg.get("standardize_confounders", []))
    if standardized_z:
        print(
            f"Standardized flow confounders: {', '.join(standardized_z)}"
        )

    # ── Models ──
    g_phi = DiscreteMediator(
        dim_x, dim_z, vocab,
        hidden_dim=flow_cfg.hidden_dim,
        n_layers=flow_cfg.n_layers,
        autoregressive=bool(flow_cfg.get("disc_autoregressive", True)),
        embed_dim=int(flow_cfg.get("disc_embed_dim", flow_cfg.embed_dim)),
        layers=mediator_schema.discrete_layers,
        z_mean=z_mean, z_scale=z_scale,
        within_block_autoregressive=bool(
            flow_cfg.get("disc_within_block_autoregressive", False)),
    ).to(device)
    continuous_family = str(flow_cfg.get("continuous_family", "spline"))
    continuous_cls = (ConditionalGaussian if continuous_family == "gaussian"
                      else ContinuousMediatorFlow)
    condition_on_disc = bool(flow_cfg.get("condition_cont_on_disc", True))
    continuous_vocab = vocab if condition_on_disc else {}
    condition_on_x = bool(flow_cfg.get("condition_cont_on_sensitive", True))
    f_theta = continuous_cls(
        dim_wc=dim_wc, dim_x=dim_x, dim_z=dim_z, vocab=continuous_vocab,
        embed_dim=flow_cfg.embed_dim,
        hidden_features=flow_cfg.hidden_features,
        n_flow_layers=flow_cfg.n_flow_layers,
        n_blocks=flow_cfg.n_blocks,
        num_bins=flow_cfg.num_bins,
        tail_bound=flow_cfg.tail_bound,
        z_mean=z_mean, z_scale=z_scale,
        condition_on_x=condition_on_x,
    ).to(device)

    # ── Loaders ──
    batch_size = flow_cfg.batch_size

    def make_loader(X, Z, Wd, Wc, Y, shuffle):
        return DataLoader(TensorDataset(X, Z, Wd, Wc, Y),
                          batch_size=batch_size, shuffle=shuffle)

    tr_loader  = make_loader(X_tr, Z_tr, Wd_tr, Wc_tr, Y_tr, shuffle=True)
    val_loader = make_loader(X_va, Z_va, Wd_va, Wc_va, Y_va, shuffle=False)

    # ── Optimiser ──
    params    = list(g_phi.parameters()) + list(f_theta.parameters())
    optimiser = Adam(params, lr=flow_cfg.lr)
    scheduler = ReduceLROnPlateau(optimiser, patience=5, factor=0.5)

    raw_selection_cfg = flow_cfg.get("checkpoint_selection")
    selection_cfg = (
        OmegaConf.to_container(raw_selection_cfg, resolve=True)
        if raw_selection_cfg is not None else {}
    )
    ppc_enabled = bool(selection_cfg.get("posterior_predictive", False))
    if ppc_enabled and not vocab and dim_wc == 0:
        raise ValueError(
            "Posterior-predictive checkpoint selection requires at least "
            "one mediator."
        )

    history = {
        "train": [], "val": [], "val_disc": [], "val_flow": [],
        "selection_score": [], "checkpoint_eligible": [],
        "posterior_predictive": [],
    }
    best_val = float("inf")
    best_score = float("inf")
    best_state = None
    last_ppc = None
    wait       = 0
    max_epochs = flow_cfg.max_epochs
    patience   = flow_cfg.patience

    print(f"\n{'Epoch':>5}  {'Train NLL':>12}  {'Val NLL':>12}  {'Val disc':>10}  {'Val flow':>10}  {'LR':>8}")
    print("-" * 68)

    for epoch in range(1, max_epochs + 1):
        g_phi.train(); f_theta.train()
        tr_sum, tr_n = 0.0, 0
        for batch in tr_loader:
            optimiser.zero_grad()
            loss = combined_loss(batch, g_phi, f_theta, device)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimiser.step()
            bs = batch[0].shape[0]
            tr_sum += loss.item() * bs
            tr_n   += bs

        tr_nll = tr_sum / tr_n
        g_phi.eval(); f_theta.eval()
        val_disc, val_flow = evaluate(val_loader, g_phi, f_theta, device)
        val_nll = val_disc + val_flow
        scheduler.step(val_nll)
        diagnostics = None
        eligible = True
        selection_score = val_nll
        if ppc_enabled:
            diagnostics = mediator_posterior_predictive(
                g_phi, f_theta, X_va, Z_va, Wd_va, Wc_va, vocab,
                sfm_cfg["mediators_cont"],
                samples_per_context=int(selection_cfg["samples_per_context"]),
                max_rows=int(selection_cfg["max_rows"]),
                seed=int(cfg.seed) + 10_000,
                ordered_names=selection_cfg.get("ordered_mediators", []),
                min_group_rows=int(selection_cfg["min_group_rows"]),
            )
            eligible = _ppc_is_eligible(diagnostics, selection_cfg)
            selection_score += (
                float(selection_cfg.get("score_weight", 1.0))
                * _ppc_penalty(diagnostics, selection_cfg)
            )
            last_ppc = diagnostics

        history["train"].append(tr_nll)
        history["val"].append(val_nll)
        history["val_disc"].append(val_disc)
        history["val_flow"].append(val_flow)
        history["selection_score"].append(selection_score)
        history["checkpoint_eligible"].append(eligible)
        history["posterior_predictive"].append(diagnostics)

        lr_now = optimiser.param_groups[0]["lr"]
        ppc_text = ""
        if diagnostics is not None:
            ppc_text = (
                "  PPC "
                f"TV={diagnostics['max_overall_marginal_tv']:.3f} "
                f"group={diagnostics['max_group_marginal_tv']:.3f} "
                f"ordW1={diagnostics['max_ordered_wasserstein']:.3f} "
                f"contW1={diagnostics['max_overall_continuous_w1']:.3f} "
                f"contGroup={diagnostics['max_group_continuous_w1']:.3f} "
                f"corr={diagnostics['max_continuous_correlation_error']:.3f} "
                f"{'pass' if eligible else 'reject'}"
            )
        print(
            f"{epoch:5d}  {tr_nll:12.4f}  {val_nll:12.4f}  "
            f"{val_disc:10.4f}  {val_flow:10.4f}  {lr_now:8.1e}"
            f"{ppc_text}"
        )

        if eligible and selection_score < best_score:
            best_val = val_nll
            best_score = selection_score
            best_state = {
                "g_phi":   {k: v.cpu().clone() for k, v in g_phi.state_dict().items()},
                "f_theta": {k: v.cpu().clone() for k, v in f_theta.state_dict().items()},
                "epoch":   epoch,
                "posterior_predictive": diagnostics,
            }
            wait = 0
        elif best_state is not None:
            wait += 1
            if wait >= patience:
                print(
                    f"\nEarly stopping at epoch {epoch}. "
                    f"Best eligible selection score: {best_score:.4f}"
                )
                break

    if best_state is None:
        raise RuntimeError(
            "No checkpoint passed the configured posterior-predictive gates. "
            f"Last diagnostics: {last_ppc}"
        )

    g_phi.load_state_dict(best_state["g_phi"])
    f_theta.load_state_dict(best_state["f_theta"])
    g_phi.to(device); f_theta.to(device)
    print(
        f"\nBest model: epoch {best_state['epoch']}  "
        f"val NLL = {best_val:.4f}  selection score = {best_score:.4f}"
    )

    # ── Sanity check: sample shapes + IS weight stats ──
    g_phi.eval(); f_theta.eval()
    with torch.no_grad():
        x0 = X_va[:50].to(device)
        z0 = Z_va[:50].to(device)
        K  = 100
        x_rep = x0.repeat_interleave(K, dim=0)
        z_rep = z0.repeat_interleave(K, dim=0)
        wd_s  = g_phi.sample(x_rep, z_rep)
        wc_s  = f_theta.sample(1, wd_s, x_rep, z_rep)
        ll_d  = g_phi.log_prob(wd_s, x_rep, z_rep)
        ll_c  = f_theta.log_prob_no_grad(wc_s, wd_s, x_rep, z_rep)
        log_p = ll_d + ll_c

        x1_rep   = (1.0 - x0).repeat_interleave(K, dim=0)
        ll_d_cf  = g_phi.log_prob(wd_s, x1_rep, z_rep)
        ll_c_cf  = f_theta.log_prob_no_grad(wc_s, wd_s, x1_rep, z_rep)
        log_p_cf = ll_d_cf + ll_c_cf
        log_r    = log_p_cf - log_p

    print(f"\nSample shapes — w_disc: {tuple(wd_s.shape)}  w_cont: {tuple(wc_s.shape)}")
    print(f"IS log-weight stats  mean={log_r.mean():.3f}  std={log_r.std():.3f}  "
          f"min={log_r.min():.3f}  max={log_r.max():.3f}")

    # ── Save ──
    flows_path = cfg.dataset.paths.flows
    os.makedirs(os.path.dirname(flows_path), exist_ok=True)
    torch.save({
        "artifact_version": str(flow_cfg.get("artifact_version", "legacy")),
        "g_phi_state":   g_phi.state_dict(),
        "f_theta_state": f_theta.state_dict(),
        "continuous_family": continuous_family,
        "g_phi_cfg":     dict(
            dim_x=dim_x, dim_z=dim_z, vocab=vocab,
            hidden_dim=flow_cfg.hidden_dim, n_layers=flow_cfg.n_layers,
            autoregressive=bool(flow_cfg.get("disc_autoregressive", True)),
            embed_dim=int(flow_cfg.get("disc_embed_dim", flow_cfg.embed_dim)),
            layers=[list(block) for block in mediator_schema.discrete_layers],
            z_mean=z_mean, z_scale=z_scale,
            within_block_autoregressive=bool(
                flow_cfg.get("disc_within_block_autoregressive", False)),
        ),
        "f_theta_cfg":   dict(
            dim_wc=dim_wc, dim_x=dim_x, dim_z=dim_z, vocab=continuous_vocab,
            embed_dim=flow_cfg.embed_dim,
            hidden_features=flow_cfg.hidden_features,
            n_flow_layers=flow_cfg.n_flow_layers,
            n_blocks=flow_cfg.n_blocks,
            num_bins=flow_cfg.num_bins, tail_bound=flow_cfg.tail_bound,
            z_mean=z_mean, z_scale=z_scale,
            condition_on_x=condition_on_x,
        ),
        "scaler":        scaler,
        "sfm_cfg":       sfm_cfg,
        "mediator_layers": [list(block) for block in mediator_schema.layers],
        "flow_context_standardization": {
            "names": standardized_z, "z_mean": z_mean, "z_scale": z_scale,
        },
        "checkpoint_selection": selection_cfg,
        "best_selection_score": best_score,
        "best_posterior_predictive": best_state["posterior_predictive"],
        "best_val_nll":  best_val,
        "best_epoch":    best_state["epoch"],
        "history":       history,
    }, flows_path)
    print(f"Models saved → {flows_path}")

    os.makedirs(os.path.dirname(tensors_path), exist_ok=True)
    torch.save({
        "X_tr": X_tr, "Z_tr": Z_tr, "Wd_tr": Wd_tr, "Wc_tr": Wc_tr, "Y_tr": Y_tr,
        "X_va": X_va, "Z_va": Z_va, "Wd_va": Wd_va, "Wc_va": Wc_va, "Y_va": Y_va,
        "scaler": scaler, "vocab": vocab, "cfg": sfm_cfg,
        "dim_x": dim_x, "dim_z": dim_z, "dim_wc": dim_wc, "dim_wd": dim_wd,
        "split": split_cfg,
    }, tensors_path)
    print(f"Tensors saved → {tensors_path}")


if __name__ == "__main__":
    main()
