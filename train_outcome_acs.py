"""
train_outcome_acs.py — Fit outcome models P(Y | X, W, Z) on ACS Income data.

Two models are trained and compared:
  - Logistic Regression  (interpretable baseline)
  - MLP Classifier       (64 → 32 hidden units, sklearn)

Features
--------
  SEX      (binary,       passthrough)
  AGEP     (continuous,   StandardScaler)
  POBP_US  (binary,       passthrough)
  SCHL_GRP (categorical,  OneHotEncoder)
  OCCP_GRP (categorical,  OneHotEncoder)
  WKHP     (continuous,   StandardScaler, original scale via inverse QT)

Outputs
-------
  outputs/outcome/acs/logreg.joblib   — fitted LogisticRegression pipeline
  outputs/outcome/acs/mlp.joblib      — fitted MLPClassifier pipeline

Usage
-----
  python train_outcome_acs.py
  python train_outcome_acs.py --tensors outputs/data/acs_tensors.pt
"""

import argparse
import os

import numpy as np
import torch
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ── Feature construction ──────────────────────────────────────────────────────

# Column indices in the raw feature matrix built by build_features()
_COL_SEX     = 0   # binary — passthrough
_COL_AGEP    = 1   # continuous — scale
_COL_POBP_US = 2   # binary — passthrough
_COL_SCHL    = 3   # categorical — OHE
_COL_OCCP    = 4   # categorical — OHE
_COL_WKHP    = 5   # continuous — scale


def build_features(X, Z, Wd, Wc, scaler) -> np.ndarray:
    """
    Stack SFM tensors into a single numpy feature matrix.

    Column order: [SEX, AGEP, POBP_US, SCHL_GRP, OCCP_GRP, WKHP_orig]

    Wc is in the QuantileTransformer's output space; scaler.inverse_transform
    maps it back to hours/week before feeding to the outcome model.
    """
    X_np  = X.numpy().astype(np.float32)           # (N, 1)
    Z_np  = Z.numpy().astype(np.float32)           # (N, 2): AGEP, POBP_US
    Wd_np = Wd.numpy().astype(np.float32)          # (N, 2): SCHL_GRP, OCCP_GRP
    Wc_np = (scaler.inverse_transform(Wc.numpy())
             if scaler is not None
             else Wc.numpy()).astype(np.float32)    # (N, 1): WKHP in hours

    return np.concatenate([X_np, Z_np, Wd_np, Wc_np], axis=1)


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("pass",      "passthrough",
             [_COL_SEX, _COL_POBP_US]),
            ("scale",     StandardScaler(),
             [_COL_AGEP, _COL_WKHP]),
            ("ohe_schl",  OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
             [_COL_SCHL]),
            ("ohe_occp",  OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
             [_COL_OCCP]),
        ],
        remainder="drop",
    )


# ── Evaluation helpers ────────────────────────────────────────────────────────

def evaluate(pipe, feat, y, X_raw, label: str):
    """Print accuracy, AUC, and group-stratified metrics."""
    pred   = pipe.predict(feat)
    prob   = pipe.predict_proba(feat)[:, 1]
    acc    = accuracy_score(y, pred)
    auc    = roc_auc_score(y, prob)
    pos_rt = prob.mean()

    print(f"\n  [{label}]")
    print(f"    Overall   — acc={acc:.4f}  AUC={auc:.4f}  pred-pos={pos_rt:.4f}")

    sex = X_raw[:, 0].astype(int)
    for g, name in [(0, "Female"), (1, "Male")]:
        m = sex == g
        g_acc = accuracy_score(y[m], pred[m])
        g_auc = roc_auc_score(y[m], prob[m])
        g_pos = prob[m].mean()
        print(f"    {name:8s}  — acc={g_acc:.4f}  AUC={g_auc:.4f}  pred-pos={g_pos:.4f}"
              f"  (n={m.sum():,})")


# ── Main ──────────────────────────────────────────────────────────────────────

def train(args):
    # ── Load tensors ──
    data   = torch.load(args.tensors, map_location="cpu", weights_only=False)
    scaler = data["scaler"]

    X_tr, Z_tr, Wd_tr, Wc_tr = data["X_tr"], data["Z_tr"], data["Wd_tr"], data["Wc_tr"]
    X_va, Z_va, Wd_va, Wc_va = data["X_va"], data["Z_va"], data["Wd_va"], data["Wc_va"]
    Y_tr = data["Y_tr"].numpy().astype(int)
    Y_va = data["Y_va"].numpy().astype(int)

    feat_tr = build_features(X_tr, Z_tr, Wd_tr, Wc_tr, scaler)
    feat_va = build_features(X_va, Z_va, Wd_va, Wc_va, scaler)

    print(f"Train: {len(feat_tr):,}  Val: {len(feat_va):,}")
    print(f"Feature matrix shape: {feat_tr.shape}")
    print(f"Positive rate — train: {Y_tr.mean():.3f}  val: {Y_va.mean():.3f}")

    prep = make_preprocessor()

    # ── Logistic Regression ──
    print("\nFitting Logistic Regression...")
    lr_pipe = Pipeline([
        ("prep", prep),
        ("clf",  LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs",
                                    random_state=42)),
    ])
    lr_pipe.fit(feat_tr, Y_tr)
    evaluate(lr_pipe, feat_tr, Y_tr, X_tr.numpy(), "LogReg — train")
    evaluate(lr_pipe, feat_va, Y_va, X_va.numpy(), "LogReg — val")

    # ── MLP ──
    print("\nFitting MLP (64 → 32)...")
    mlp_pipe = Pipeline([
        ("prep", make_preprocessor()),   # fresh preprocessor fitted on same data
        ("clf",  MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=42,
            verbose=False,
        )),
    ])
    mlp_pipe.fit(feat_tr, Y_tr)
    evaluate(mlp_pipe, feat_tr, Y_tr, X_tr.numpy(), "MLP — train")
    evaluate(mlp_pipe, feat_va, Y_va, X_va.numpy(), "MLP — val")

    # ── Save ──
    save_dir = "outputs/outcome/acs"
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(lr_pipe,  f"{save_dir}/logreg.joblib")
    joblib.dump(mlp_pipe, f"{save_dir}/mlp.joblib")
    print(f"\nModels saved → {save_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tensors", default="outputs/data/acs_tensors.pt")
    train(p.parse_args())
