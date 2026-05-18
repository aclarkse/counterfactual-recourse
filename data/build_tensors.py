"""
data/build_tensors.py — Generic SFM tensor construction utilities.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import QuantileTransformer


def build_tensors(df: pd.DataFrame, cfg: dict, scaler=None, fit_scaler: bool = True,
                  wc_clip=None):
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
        lo, hi = wc_clip[0], wc_clip[1]
        Wc_np = Wc_np.clip(lo, hi)
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


def stack_sfm_features(X, Z, Wd, Wc, scaler) -> np.ndarray:
    """
    Assemble SFM tensors into a (N, n_x+n_z+n_wd+n_wc) numpy array.
    Column order: [X | Z | Wd | Wc_natural]
    Wc is inverse-transformed from scaler space to natural units before stacking.
    """
    parts = []
    for t in [X, Z]:
        if t.shape[1] > 0:
            parts.append(t.numpy().astype(np.float32))
    if Wd.shape[1] > 0:
        parts.append(Wd.numpy().astype(np.float32))
    if Wc.shape[1] > 0:
        wc = scaler.inverse_transform(Wc.numpy()) if scaler is not None else Wc.numpy()
        parts.append(wc.astype(np.float32))
    return np.concatenate(parts, axis=1) if parts else np.empty((len(X), 0), np.float32)
