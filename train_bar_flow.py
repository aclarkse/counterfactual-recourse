"""
train_bar_flow.py — Train P(W | X, Z) flow models on the Law School bar passage dataset.

SFM role assignment
-------------------
  Sensitive   X : race           (0=Black, 1=White)
  Confounders Z : gender, fam_inc
  Mediators   W_disc : (none)
  Mediators   W_cont : lsat, gpa  (QuantileTransform → N(0,1))
  Outcome     Y : bar_pass

No discrete mediators means the flow conditions only on (X, Z), so there is
no embedding deadlock. dim_wc=2 keeps the NSF expressive (no affine-collapse).

Outputs
-------
  outputs/flows/law_school/flow_models.pt
  outputs/data/bar_tensors.pt

Usage
-----
  python train_bar_flow.py
  python train_bar_flow.py --data data/bar.csv --max-epochs 150
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

from train_acs_flow import (
    DiscreteMediator,
    ContinuousMediatorFlow,
    load_flow_models,          # noqa: F401 — re-exported for downstream use
    build_tensors,
    combined_loss,
    evaluate,
)

import warnings
warnings.filterwarnings("ignore")


# ── SFM config ────────────────────────────────────────────────────────────────

BAR_CFG = {
    "sensitive":      ["race"],
    "confounders":    ["gender", "fam_inc"],
    "mediators_disc": [],
    "mediators_cont": ["lsat", "gpa"],
    "outcome":        "bar_pass",
}

GROUP_LABELS = {0: "Black", 1: "White"}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_bar_data(path: str = "data/bar.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = ["gender", "race1", "lsat", "gpa", "fam_inc", "pass_bar"]
    df = df[cols].dropna(subset=["gender", "fam_inc"]).copy()

    df["gender"] = df["gender"].map({"male": 1, "female": 0})
    df["race"]   = df["race1"].map({"white": 1, "black": 0})
    df = df[df["race"].isin([0, 1])].drop(columns="race1")

    df["fam_inc"] = df["fam_inc"].astype(int)
    df["lsat"]    = df["lsat"].round(0).astype(float)
    df = df.rename(columns={"pass_bar": "bar_pass"})

    print(f"Rows: {len(df):,}")
    print(f"Bar pass rate: {df['bar_pass'].mean():.3f}")
    print(f"Black share  : {(df['race'] == 0).mean():.3f}")
    print(f"White share  : {(df['race'] == 1).mean():.3f}")
    return df.reset_index(drop=True)


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df_raw = load_bar_data(args.data)
    cfg    = BAR_CFG

    idx_tr, idx_val = train_test_split(np.arange(len(df_raw)), test_size=0.2, random_state=42)
    df_tr  = df_raw.iloc[idx_tr].reset_index(drop=True)
    df_val = df_raw.iloc[idx_val].reset_index(drop=True)

    # No wc_clip: lsat and gpa are naturally bounded and don't need hard clipping
    X_tr, Z_tr, Wd_tr, Wc_tr, Y_tr, scaler, vocab = build_tensors(df_tr,  cfg, fit_scaler=True)
    X_va, Z_va, Wd_va, Wc_va, Y_va, _,      _     = build_tensors(df_val, cfg, scaler=scaler, fit_scaler=False)

    dim_x  = X_tr.shape[1]
    dim_z  = Z_tr.shape[1]
    dim_wc = Wc_tr.shape[1]
    dim_wd = Wd_tr.shape[1]

    print(f"\nTrain: {len(df_tr):,}  Val: {len(df_val):,}")
    print(f"dim_x={dim_x}  dim_z={dim_z}  dim_W_disc={dim_wd}  dim_W_cont={dim_wc}")
    print(f"Vocab: {vocab}")

    # DiscreteMediator with empty vocab is a no-op (returns 0 log-prob, empty samples)
    g_phi   = DiscreteMediator(dim_x, dim_z, vocab, hidden_dim=64).to(device)
    # vocab={} → no discrete embeddings; flow conditions on (X, Z) only
    f_theta = ContinuousMediatorFlow(
        dim_wc=dim_wc, dim_x=dim_x, dim_z=dim_z, vocab={},
        embed_dim=4, hidden_features=64, n_flow_layers=4,
    ).to(device)

    def make_loader(X, Z, Wd, Wc, Y, shuffle):
        return DataLoader(TensorDataset(X, Z, Wd, Wc, Y),
                          batch_size=args.batch_size, shuffle=shuffle)

    tr_loader  = make_loader(X_tr, Z_tr, Wd_tr, Wc_tr, Y_tr, shuffle=True)
    val_loader = make_loader(X_va, Z_va, Wd_va, Wc_va, Y_va, shuffle=False)

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
            bs      = batch[0].shape[0]
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

    # ── Save ──
    save_dir = "outputs/flows/law_school"
    os.makedirs(save_dir, exist_ok=True)
    torch.save({
        "g_phi_state":   g_phi.state_dict(),
        "f_theta_state": f_theta.state_dict(),
        "g_phi_cfg":     dict(dim_x=dim_x, dim_z=dim_z, vocab=vocab),
        "f_theta_cfg":   dict(dim_wc=dim_wc, dim_x=dim_x, dim_z=dim_z, vocab={},
                              num_bins=8, tail_bound=3.0),
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
    }, "outputs/data/bar_tensors.pt")
    print("Tensors saved → outputs/data/bar_tensors.pt")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train Law School bar passage flow models")
    p.add_argument("--data",       default="data/bar.csv")
    p.add_argument("--batch-size", type=int,   default=256)
    p.add_argument("--max-epochs", type=int,   default=100)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--patience",   type=int,   default=10)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
