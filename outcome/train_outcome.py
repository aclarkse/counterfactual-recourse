"""
outcome/train_outcome.py — Generic Hydra entry point for outcome model training.

Run: python -m outcome.train_outcome [dataset=acs|bar]
"""

import json
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

import hydra
from omegaconf import OmegaConf

from data.build_tensors import stack_sfm_features
from outcome.models import make_preprocessor


def _evaluate(pipe, feat, y, X_raw, label: str, sensitive_col: int = 0,
              group_labels: dict = None):
    """Print accuracy, AUC, and group-stratified metrics."""
    pred   = pipe.predict(feat)
    prob   = pipe.predict_proba(feat)[:, 1]
    acc    = accuracy_score(y, pred)
    auc    = roc_auc_score(y, prob)
    pos_rt = prob.mean()

    print(f"\n  [{label}]")
    print(f"    Overall   — acc={acc:.4f}  AUC={auc:.4f}  pred-pos={pos_rt:.4f}")

    grp = X_raw[:, sensitive_col].astype(int)
    if group_labels is None:
        group_labels = {0: "Group 0", 1: "Group 1"}
    groups = {}
    for g, name in group_labels.items():
        m = grp == g
        if m.sum() < 2:
            continue
        g_acc = accuracy_score(y[m], pred[m])
        g_auc = roc_auc_score(y[m], prob[m])
        g_pos = prob[m].mean()
        print(f"    {name:8s}  — acc={g_acc:.4f}  AUC={g_auc:.4f}  pred-pos={g_pos:.4f}"
              f"  (n={m.sum():,})")
        groups[str(g)] = {
            "label": str(name), "n": int(m.sum()), "accuracy": float(g_acc),
            "auc": float(g_auc), "mean_predicted_probability": float(g_pos),
        }
    return {
        "n": int(len(y)), "accuracy": float(acc), "auc": float(auc),
        "mean_predicted_probability": float(pos_rt), "groups": groups,
    }


@hydra.main(config_path="../conf", config_name="config", version_base="1.1")
def main(cfg):
    # ── Load tensors ──
    tensors_path = cfg.dataset.paths.tensors
    data   = torch.load(tensors_path, map_location="cpu", weights_only=False)
    scaler = data["scaler"]

    X_tr, Z_tr, Wd_tr, Wc_tr = data["X_tr"], data["Z_tr"], data["Wd_tr"], data["Wc_tr"]
    X_va, Z_va, Wd_va, Wc_va = data["X_va"], data["Z_va"], data["Wd_va"], data["Wc_va"]
    Y_tr = data["Y_tr"].numpy().astype(int)
    Y_va = data["Y_va"].numpy().astype(int)

    feat_tr = stack_sfm_features(X_tr, Z_tr, Wd_tr, Wc_tr, scaler)
    feat_va = stack_sfm_features(X_va, Z_va, Wd_va, Wc_va, scaler)

    print(f"Train: {len(feat_tr):,}  Val: {len(feat_va):,}")
    print(f"Feature matrix shape: {feat_tr.shape}")
    print(f"Positive rate — train: {Y_tr.mean():.3f}  val: {Y_va.mean():.3f}")

    col_types  = OmegaConf.to_container(cfg.dataset.outcome.col_types, resolve=True)
    disc_names = list(cfg.dataset.sfm.mediators_disc)

    # ── Sensitive attribute for group-stratified evaluation ──
    sensitive_name = list(cfg.dataset.sfm.sensitive)[0]
    # Column index of sensitive in feat: always 0 (X comes first)
    sensitive_col = 0

    # ── Fit models ──
    model_specs = OmegaConf.to_container(cfg.dataset.outcome.models, resolve=True)
    save_dir = cfg.dataset.paths.outcome_dir
    os.makedirs(save_dir, exist_ok=True)

    metrics = {"seed": int(cfg.seed), "dataset": str(cfg.dataset.name),
               "models": {}}
    for spec in model_specs:
        name      = spec["name"]
        mtype     = spec["type"]
        prep      = make_preprocessor(col_types, disc_names)

        if mtype == "logreg":
            clf = LogisticRegression(
                C=spec.get("C", 1.0),
                max_iter=spec.get("max_iter", 1000),
                solver="lbfgs",
                random_state=int(cfg.seed),
            )
            slug = "logreg"
        elif mtype == "mlp":
            hl = tuple(spec.get("hidden_layer_sizes", [64, 32]))
            clf = MLPClassifier(
                hidden_layer_sizes=hl,
                activation="relu",
                solver="adam",
                max_iter=spec.get("max_iter", 500),
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                random_state=int(cfg.seed),
                verbose=False,
            )
            slug = "mlp"
        elif mtype == "random_forest":
            clf = RandomForestClassifier(
                n_estimators=int(spec.get("n_estimators", 300)),
                max_depth=spec.get("max_depth"),
                min_samples_leaf=int(spec.get("min_samples_leaf", 1)),
                max_features=spec.get("max_features", "sqrt"),
                class_weight=spec.get("class_weight"),
                n_jobs=int(spec.get("n_jobs", -1)),
                random_state=int(cfg.seed),
            )
            slug = "random_forest"
        else:
            raise ValueError(f"Unknown model type: {mtype}")

        print(f"\nFitting {name}...")
        pipe = Pipeline([("prep", prep), ("clf", clf)])
        pipe.fit(feat_tr, Y_tr)

        train_metrics = _evaluate(
            pipe, feat_tr, Y_tr, X_tr.numpy(), f"{name} — train",
            sensitive_col=sensitive_col,
        )
        val_metrics = _evaluate(
            pipe, feat_va, Y_va, X_va.numpy(), f"{name} — val",
            sensitive_col=sensitive_col,
        )

        out_path = f"{save_dir}/{slug}.joblib"
        joblib.dump(pipe, out_path)
        print(f"  Saved → {out_path}")
        metrics["models"][name] = {
            "type": mtype, "slug": slug, "train": train_metrics,
            "validation": val_metrics,
        }

    metrics_path = f"{save_dir}/metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nAll models and metrics saved → {save_dir}/")


if __name__ == "__main__":
    main()
