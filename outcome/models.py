"""
outcome/models.py — Generic outcome model utilities.

Provides: make_preprocessor, _SFMPreprocess, TorchLogReg, TorchMLP,
          build_torch_pipeline, stack_sfm_features (re-export).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Re-export for backward compat
from data.build_tensors import stack_sfm_features  # noqa: F401


def make_preprocessor(col_types: dict, disc_names: list) -> ColumnTransformer:
    """
    Build a ColumnTransformer from a col_types specification dict.

    col_types keys: 'X', 'Z', 'Wd', 'Wc' — each maps to a list of transform
    names ('passthrough', 'scale', 'ohe') in column order.
    disc_names: ordered list of discrete mediator names (matches Wd order).
    """
    pass_cols, scale_cols = [], []
    ohe_dict = {}   # name → col_idx
    col = 0
    for t in col_types.get("X", []):
        (pass_cols if t == "passthrough" else scale_cols).append(col); col += 1
    for t in col_types.get("Z", []):
        (pass_cols if t == "passthrough" else scale_cols).append(col); col += 1
    for name, t in zip(disc_names, col_types.get("Wd", [])):
        if t == "ohe":
            ohe_dict[name] = col
        elif t == "scale":
            scale_cols.append(col)
        else:
            pass_cols.append(col)
        col += 1
    for t in col_types.get("Wc", []):
        (pass_cols if t == "passthrough" else scale_cols).append(col); col += 1

    transformers = []
    if pass_cols:
        transformers.append(("pass", "passthrough", pass_cols))
    if scale_cols:
        transformers.append(("scale", StandardScaler(), scale_cols))
    for name, c in ohe_dict.items():
        transformers.append((f"ohe_{name.lower()}",
                             OneHotEncoder(sparse_output=False, handle_unknown="ignore"), [c]))
    return ColumnTransformer(transformers=transformers, remainder="drop")


class _SFMPreprocess(nn.Module):
    """
    Replicate a fitted ColumnTransformer's operations in PyTorch.
    Stores self.output_dim.
    """

    def __init__(self, prep):
        super().__init__()
        scale_tr = prep.named_transformers_.get("scale")
        if scale_tr is not None and hasattr(scale_tr, "mean_"):
            self.register_buffer("mean_", torch.tensor(scale_tr.mean_, dtype=torch.float32))
            self.register_buffer("scale_", torch.tensor(scale_tr.scale_, dtype=torch.float32))
        else:
            self.mean_ = self.scale_ = None

        self.plan = []   # list of ("pass"|"scale"|"ohe", col_list, [vocab_size])
        self.output_dim = 0

        for tr_name, transformer, cols in prep.transformers_:
            if tr_name == "remainder":
                continue
            cols = list(cols)
            if isinstance(transformer, str) and transformer == "passthrough":
                self.plan.append(("pass", cols))
                self.output_dim += len(cols)
            elif hasattr(transformer, "mean_"):   # StandardScaler
                self.plan.append(("scale", cols))
                self.output_dim += len(cols)
            elif hasattr(transformer, "categories_"):  # OHE (single column per transformer)
                vsz = len(transformer.categories_[0])
                self.plan.append(("ohe", cols, vsz))
                self.output_dim += vsz

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        parts = []
        for entry in self.plan:
            t = entry[0]
            cols = entry[1]
            if t == "pass":
                parts.append(feat[:, cols])
            elif t == "scale":
                x = feat[:, cols]
                parts.append((x - self.mean_) / self.scale_)
            else:  # ohe
                parts.append(F.one_hot(feat[:, cols[0]].long(), entry[2]).float())
        return torch.cat(parts, dim=1)


class TorchLogReg(nn.Module):
    """LogisticRegression pipeline: frozen preprocessing + frozen linear."""

    def __init__(self, prep, clf):
        super().__init__()
        self.preprocess = _SFMPreprocess(prep)
        n_in = self.preprocess.output_dim
        self.linear = nn.Linear(n_in, 1)
        self.linear.weight.data = torch.tensor(clf.coef_, dtype=torch.float32)
        self.linear.bias.data   = torch.tensor(clf.intercept_, dtype=torch.float32)
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, feat):
        return torch.sigmoid(self.linear(self.preprocess(feat))).squeeze(1)


class TorchMLP(nn.Module):
    """MLPClassifier pipeline: frozen preprocessing + frozen MLP."""

    def __init__(self, prep, clf):
        super().__init__()
        self.preprocess = _SFMPreprocess(prep)
        n_in = self.preprocess.output_dim
        layers = []
        in_dim = n_in
        for i, (W, b) in enumerate(zip(clf.coefs_, clf.intercepts_)):
            out_dim = W.shape[1]
            lin = nn.Linear(in_dim, out_dim)
            lin.weight.data = torch.tensor(W.T, dtype=torch.float32)
            lin.bias.data   = torch.tensor(b, dtype=torch.float32)
            for p in lin.parameters():
                p.requires_grad_(False)
            layers.append(lin)
            if i < len(clf.coefs_) - 1:
                layers.append(nn.ReLU())
            in_dim = out_dim
        self.net = nn.Sequential(*layers)
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, feat):
        return torch.sigmoid(self.net(self.preprocess(feat))).squeeze(1)


def build_torch_pipeline(pipe) -> nn.Module:
    """
    Convert a fitted sklearn Pipeline (prep + clf) into a frozen PyTorch Module.
    Supports LogisticRegression and MLPClassifier.
    """
    prep = pipe.named_steps["prep"]
    clf  = pipe.named_steps["clf"]
    return TorchMLP(prep, clf) if hasattr(clf, "coefs_") else TorchLogReg(prep, clf)
