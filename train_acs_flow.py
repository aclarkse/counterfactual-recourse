"""
train_acs_flow.py — Train P(W | X, Z) flow models on ACS Income data.

SFM role assignment
-------------------
  Sensitive   X : SEX          (0=Female, 1=Male)
  Confounders Z : AGEP, POBP_US
  Mediators   W_disc : SCHL_GRP, OCCP_GRP
              W_cont : WKHP
  Outcome     Y : income (PINCP > $50k)

Outputs
-------
  outputs/flows/acs/flow_models.pt   — trained g_phi + f_theta + metadata
  outputs/data/acs_tensors.pt        — train/val tensors + scaler + vocab

Usage
-----
  python train_acs_flow.py
  python train_acs_flow.py --year 2019 --states CA NY TX --max-epochs 150
  python train_acs_flow.py --max-rows 50000          # fast dev run
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from nflows import flows
from nflows.transforms import (
    CompositeTransform,
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
    RandomPermutation,
)
from nflows.distributions import StandardNormal

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer

import warnings
warnings.filterwarnings("ignore")


# ── SFM config ────────────────────────────────────────────────────────────────

ACS_CFG = {
    "sensitive":      ["SEX"],
    "confounders":    ["AGEP", "POBP_US"],
    "mediators_disc": ["SCHL_GRP", "OCCP_GRP"],
    "mediators_cont": ["WKHP"],
    "outcome":        "income",
}

# Clip WKHP to [1, 60] before QuantileTransform — removes implausibly long
# work-weeks and bounds the domain seen by the flow.
WKHP_CLIP = (1, 60)


# ── Data ──────────────────────────────────────────────────────────────────────

def build_tensors(df: pd.DataFrame, cfg: dict, scaler=None, fit_scaler: bool = True,
                  wc_clip: tuple | None = None):
    """
    Project a DataFrame onto SFM roles. Returns:
      X_t, Z_t, W_disc_t, W_cont_t, Y_t, scaler, vocab
    """
    X_cols  = cfg["sensitive"]
    Z_cols  = cfg["confounders"]
    Wd_cols = cfg["mediators_disc"]
    Wc_cols = cfg["mediators_cont"]
    y_col   = cfg["outcome"]

    vocab, Wd_arrays = {}, []
    for col in Wd_cols:
        if pd.api.types.is_string_dtype(df[col]) or str(df[col].dtype) == "category":
            cats = sorted(df[col].unique())
            mapping = {c: i for i, c in enumerate(cats)}
            Wd_arrays.append(df[col].map(mapping).values)
            vocab[col] = len(cats)
        else:
            Wd_arrays.append(df[col].values.astype(int))
            vocab[col] = int(df[col].max()) + 1

    Wc_np = df[Wc_cols].values.astype(np.float32) if Wc_cols else np.empty((len(df), 0), np.float32)
    if Wc_cols and wc_clip is not None:
        Wc_np = Wc_np.clip(*wc_clip)
    if fit_scaler and Wc_cols:
        scaler = QuantileTransformer(output_distribution="normal", random_state=42)
        Wc_np = scaler.fit_transform(Wc_np).astype(np.float32)
    elif scaler is not None and Wc_cols:
        Wc_np = scaler.transform(Wc_np).astype(np.float32)

    X_np = df[X_cols].values.astype(np.float32) if X_cols else np.empty((len(df), 0), np.float32)

    if Z_cols:
        Z_parts = []
        for col in Z_cols:
            if pd.api.types.is_string_dtype(df[col]) or str(df[col].dtype) == "category":
                cats = sorted(df[col].unique())
                mapping = {c: i for i, c in enumerate(cats)}
                Z_parts.append(df[col].map(mapping).values.astype(np.float32))
            else:
                Z_parts.append(df[col].values.astype(np.float32))
        Z_np = np.column_stack(Z_parts)
    else:
        Z_np = np.empty((len(df), 0), np.float32)

    Y_np = df[y_col].values.astype(np.float32)

    to_t = lambda a: torch.tensor(a, dtype=torch.float32)
    W_disc_t = (
        torch.tensor(np.column_stack(Wd_arrays), dtype=torch.long)
        if Wd_arrays
        else torch.empty(len(df), 0, dtype=torch.long)
    )
    return to_t(X_np), to_t(Z_np), W_disc_t, to_t(Wc_np), to_t(Y_np), scaler, vocab


# ── Models ────────────────────────────────────────────────────────────────────

class DiscreteMediator(nn.Module):
    """P(W_disc | X, Z) as a product of independent categoricals via a shared MLP."""

    def __init__(self, dim_x: int, dim_z: int, vocab: dict,
                 hidden_dim: int = 64, n_layers: int = 2):
        super().__init__()
        self.vocab     = vocab
        self.col_names = list(vocab.keys())
        n_cats = list(vocab.values())
        in_dim = dim_x + dim_z

        if in_dim == 0 or not vocab:
            self.backbone = self.heads = None
            return

        layers, d = [], in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden_dim), nn.ReLU()]
            d = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, k) for k in n_cats])

    def _cond(self, x, z):
        parts = [p for p in [x, z] if p.shape[1] > 0]
        return torch.cat(parts, dim=1) if parts else None

    def log_prob(self, w_disc: torch.Tensor, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if not self.vocab:
            return torch.zeros(x.shape[0], device=x.device)
        h = self.backbone(self._cond(x, z))
        total = torch.zeros(x.shape[0], device=x.device)
        for i, head in enumerate(self.heads):
            total += F.cross_entropy(head(h), w_disc[:, i], reduction="none") * -1.0
        return total

    @torch.no_grad()
    def sample(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if not self.vocab:
            return torch.empty(x.shape[0], 0, dtype=torch.long, device=x.device)
        h = self.backbone(self._cond(x, z))
        cols = [torch.multinomial(torch.softmax(head(h), dim=-1), 1) for head in self.heads]
        return torch.cat(cols, dim=1)


class ContinuousMediatorFlow(nn.Module):
    """P(W_cont | W_disc, X, Z) via a conditional Neural Spline Flow (NSF).

    NSF replaces the affine coupling of MAF with a piecewise rational-quadratic
    spline, which is essential when dim_wc=1: stacked affine transforms on a
    1-D variable compose to a single affine map (= conditional Gaussian), giving
    a flat NLL after epoch 1. NSF remains expressive for any dim_wc >= 1.
    """

    def __init__(self, dim_wc: int, dim_x: int, dim_z: int, vocab: dict,
                 embed_dim: int = 4, hidden_features: int = 64,
                 n_flow_layers: int = 4, n_blocks: int = 2,
                 num_bins: int = 8, tail_bound: float = 3.0):
        super().__init__()
        self.dim_wc = dim_wc
        self.embeddings = nn.ModuleList([
            nn.Embedding(n_cats, embed_dim) for n_cats in vocab.values()
        ])
        self.dim_context = len(vocab) * embed_dim + dim_x + dim_z

        if dim_wc == 0:
            self.flow = None
            return

        transforms_ = []
        for _ in range(n_flow_layers):
            transforms_.append(
                MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                    features=dim_wc,
                    hidden_features=hidden_features,
                    context_features=self.dim_context if self.dim_context > 0 else None,
                    num_bins=num_bins,
                    num_blocks=n_blocks,
                    use_residual_blocks=True,
                    activation=F.relu,
                    tails="linear",
                    tail_bound=tail_bound,
                )
            )
            if dim_wc > 1:
                transforms_.append(RandomPermutation(features=dim_wc))

        self.flow = flows.Flow(
            transform=CompositeTransform(transforms_),
            distribution=StandardNormal(shape=[dim_wc]),
        )

    def _context(self, w_disc, x, z):
        parts = [emb(w_disc[:, i]) for i, emb in enumerate(self.embeddings)]
        parts += [p for p in [x, z] if p.shape[1] > 0]
        return torch.cat(parts, dim=1) if parts else None

    def log_prob(self, w_cont, w_disc, x, z):
        if self.dim_wc == 0 or self.flow is None:
            return torch.zeros(x.shape[0], device=x.device)
        return self.flow.log_prob(w_cont, context=self._context(w_disc, x, z))

    @torch.no_grad()
    def sample(self, n: int, w_disc, x, z):
        if self.dim_wc == 0 or self.flow is None:
            return torch.empty(x.shape[0], 0, device=x.device)
        return self.flow.sample(1, context=self._context(w_disc, x, z)).squeeze(1)

    @torch.no_grad()
    def log_prob_no_grad(self, w_cont, w_disc, x, z):
        return self.log_prob(w_cont, w_disc, x, z)


def load_flow_models(checkpoint_path: str, device: torch.device):
    """Reload trained g_phi and f_theta from a checkpoint file."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    g_phi_ = DiscreteMediator(**ckpt["g_phi_cfg"]).to(device)
    f_theta_ = ContinuousMediatorFlow(**ckpt["f_theta_cfg"]).to(device)
    g_phi_.load_state_dict(ckpt["g_phi_state"])
    f_theta_.load_state_dict(ckpt["f_theta_state"])
    g_phi_.eval(); f_theta_.eval()
    return g_phi_, f_theta_, ckpt["scaler"], ckpt["sfm_cfg"]


# ── Training ──────────────────────────────────────────────────────────────────

def combined_loss(batch, g_phi, f_theta, device):
    x, z, w_disc, w_cont = [t.to(device) for t in batch[:4]]
    ll_disc = g_phi.log_prob(w_disc, x, z)
    ll_cont = f_theta.log_prob(w_cont, w_disc, x, z)
    return -(ll_disc + ll_cont).mean()


@torch.no_grad()
def evaluate(loader, g_phi, f_theta, device):
    total_disc, total_cont, n = 0.0, 0.0, 0
    for batch in loader:
        x, z, w_disc, w_cont = [t.to(device) for t in batch[:4]]
        ll_disc = g_phi.log_prob(w_disc, x, z)
        ll_cont = f_theta.log_prob(w_cont, w_disc, x, z)
        bs = x.shape[0]
        total_disc += (-ll_disc.mean().item()) * bs
        total_cont += (-ll_cont.mean().item()) * bs
        n += bs
    return total_disc / n, total_cont / n


def train(args):
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load data ──
    from data_acs import load_acs_income
    states = args.states if args.states else None
    df_raw = load_acs_income(year=args.year, states=states, max_rows=args.max_rows)

    cfg = ACS_CFG
    idx_tr, idx_val = train_test_split(np.arange(len(df_raw)), test_size=0.2, random_state=42)
    df_tr  = df_raw.iloc[idx_tr].reset_index(drop=True)
    df_val = df_raw.iloc[idx_val].reset_index(drop=True)

    X_tr, Z_tr, Wd_tr, Wc_tr, Y_tr, scaler, vocab = build_tensors(df_tr, cfg, fit_scaler=True, wc_clip=WKHP_CLIP)
    X_va, Z_va, Wd_va, Wc_va, Y_va, _, _          = build_tensors(df_val, cfg, scaler=scaler, fit_scaler=False, wc_clip=WKHP_CLIP)

    dim_x  = X_tr.shape[1]
    dim_z  = Z_tr.shape[1]
    dim_wc = Wc_tr.shape[1]
    dim_wd = Wd_tr.shape[1]

    print(f"Train: {len(df_tr):,}  Val: {len(df_val):,}")
    print(f"dim_x={dim_x}  dim_z={dim_z}  dim_W_disc={dim_wd}  dim_W_cont={dim_wc}")
    print(f"Vocab: {vocab}")

    # ── Models ──
    g_phi   = DiscreteMediator(dim_x, dim_z, vocab, hidden_dim=64).to(device)
    # vocab={} → flow conditions only on (X, Z), not on discrete mediator
    # embeddings. This avoids the chicken-and-egg deadlock that occurs when
    # randomly-initialized embeddings prevent the flow from learning, which
    # in turn prevents the embeddings from receiving useful gradients.
    f_theta = ContinuousMediatorFlow(
        dim_wc=dim_wc, dim_x=dim_x, dim_z=dim_z, vocab={},
        embed_dim=4, hidden_features=64, n_flow_layers=4,
    ).to(device)

    # ── Loaders ──
    def make_loader(X, Z, Wd, Wc, Y, shuffle):
        return DataLoader(TensorDataset(X, Z, Wd, Wc, Y),
                          batch_size=args.batch_size, shuffle=shuffle)

    tr_loader  = make_loader(X_tr, Z_tr, Wd_tr, Wc_tr, Y_tr, shuffle=True)
    val_loader = make_loader(X_va, Z_va, Wd_va, Wc_va, Y_va, shuffle=False)

    # ── Optimiser ──
    params    = list(g_phi.parameters()) + list(f_theta.parameters())
    optimiser = Adam(params, lr=args.lr)
    scheduler = ReduceLROnPlateau(optimiser, patience=5, factor=0.5)

    history    = {"train": [], "val": [], "val_disc": [], "val_flow": []}
    best_val   = float("inf")
    best_state = None
    wait       = 0

    print(f"\n{'Epoch':>5}  {'Train NLL':>12}  {'Val NLL':>12}  {'Val disc':>10}  {'Val flow':>10}  {'LR':>8}")
    print("-" * 68)

    for epoch in range(1, args.max_epochs + 1):
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

        history["train"].append(tr_nll)
        history["val"].append(val_nll)
        history["val_disc"].append(val_disc)
        history["val_flow"].append(val_flow)

        lr_now = optimiser.param_groups[0]["lr"]
        print(f"{epoch:5d}  {tr_nll:12.4f}  {val_nll:12.4f}  {val_disc:10.4f}  {val_flow:10.4f}  {lr_now:8.1e}")

        if val_nll < best_val:
            best_val = val_nll
            best_state = {
                "g_phi":   {k: v.cpu().clone() for k, v in g_phi.state_dict().items()},
                "f_theta": {k: v.cpu().clone() for k, v in f_theta.state_dict().items()},
                "epoch":   epoch,
            }
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}. Best val NLL: {best_val:.4f}")
                break

    g_phi.load_state_dict(best_state["g_phi"])
    f_theta.load_state_dict(best_state["f_theta"])
    g_phi.to(device); f_theta.to(device)
    print(f"\nBest model: epoch {best_state['epoch']}  val NLL = {best_val:.4f}")

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
    save_dir = "outputs/flows/acs"
    os.makedirs(save_dir, exist_ok=True)
    torch.save({
        "g_phi_state":   g_phi.state_dict(),
        "f_theta_state": f_theta.state_dict(),
        "g_phi_cfg":     dict(dim_x=dim_x, dim_z=dim_z, vocab=vocab),
        "f_theta_cfg":   dict(dim_wc=dim_wc, dim_x=dim_x, dim_z=dim_z, vocab={}, num_bins=8, tail_bound=3.0),
        "scaler":        scaler,
        "sfm_cfg":       cfg,
        "best_val_nll":  best_val,
        "best_epoch":    best_state["epoch"],
        "history":       history,
    }, f"{save_dir}/flow_models.pt")
    print(f"Models saved → {save_dir}/flow_models.pt")

    os.makedirs("outputs/data", exist_ok=True)
    torch.save({
        "X_tr": X_tr, "Z_tr": Z_tr, "Wd_tr": Wd_tr, "Wc_tr": Wc_tr, "Y_tr": Y_tr,
        "X_va": X_va, "Z_va": Z_va, "Wd_va": Wd_va, "Wc_va": Wc_va, "Y_va": Y_va,
        "scaler": scaler, "vocab": vocab, "cfg": cfg,
        "dim_x": dim_x, "dim_z": dim_z, "dim_wc": dim_wc, "dim_wd": dim_wd,
    }, "outputs/data/acs_tensors.pt")
    print("Tensors saved → outputs/data/acs_tensors.pt")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train ACS Income flow models")
    p.add_argument("--year",       type=int,   default=2018)
    p.add_argument("--states",     nargs="+",  default=None,
                   help="State abbreviations (default: CA only). Pass multiple: --states CA NY TX")
    p.add_argument("--max-rows",   type=int,   default=None,
                   help="Subsample to this many rows (useful for quick dev runs)")
    p.add_argument("--batch-size", type=int,   default=256)
    p.add_argument("--max-epochs", type=int,   default=100)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--patience",   type=int,   default=10)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
