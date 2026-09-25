"""Fast, model-independent candidate enumeration and model scoring."""

import os
import pickle
import time

import numpy as np
import torch

from data.build_tensors import stack_sfm_features
from evaluation.compute_recourse import _build_candidate_arrays

_BATCH = 500_000


def sanity_check_proba(pipe, torch_pipe, data, scaler, n_check=500,
                       tol=1e-5, device=None):
    feat = stack_sfm_features(
        data["X_va"][:n_check], data["Z_va"][:n_check],
        data["Wd_va"][:n_check], data["Wc_va"][:n_check], scaler,
    )
    sklearn_prob = pipe.predict_proba(feat)[:, 1]
    with torch.no_grad():
        pytorch_prob = torch_pipe(torch.as_tensor(
            feat, dtype=torch.float32, device=device
        )).cpu().numpy()
    error = np.abs(sklearn_prob - pytorch_prob)
    print(f"    predict_proba agreement: max={error.max():.2e} "
          f"mean={error.mean():.2e} (n={len(feat)})")
    if error.max() > tol:
        raise RuntimeError(f"PyTorch conversion error {error.max():.2e} > {tol:.2e}")
    return float(error.max())


def select_eval_idx(data, n_max, rng_seed=0):
    """Legacy helper: fixed Y=0 cohort. New recourse uses model-specific TNs."""
    idx = np.flatnonzero(data["Y_va"].numpy().astype(int) == 0)
    if len(idx) > n_max:
        idx = np.sort(np.random.default_rng(rng_seed).choice(
            idx, int(n_max), replace=False
        ))
    return idx


def enumerate_pool(data, eval_idx, scaler, disc_names, cont_names,
                   mediator_specs, vocab, weights):
    all_x, all_xcf, metadata, sizes, groups, factual_positions = [], [], [], [], [], []
    print(f"  Enumerating candidate pool for {len(eval_idx)} individuals...")
    started = time.time()
    for idx in eval_idx:
        x = float(data["X_va"][idx, 0])
        z = data["Z_va"][idx].numpy().astype(np.float32)
        wd = data["Wd_va"][idx].numpy()
        wc_std = data["Wc_va"][idx].numpy()
        wc = (scaler.inverse_transform(wc_std.reshape(1, -1))[0]
              if scaler is not None and len(wc_std) else wc_std)
        feat_x, feat_xcf, cost, changes = _build_candidate_arrays(
            x, z, wd, wc, disc_names, cont_names, mediator_specs, vocab, weights
        )
        all_x.append(feat_x); all_xcf.append(feat_xcf)
        metadata.append((cost, changes)); sizes.append(len(cost)); groups.append(int(x))
        factual_positions.append(int(np.argmin(cost)))
    total = sum(sizes)
    print(f"  Enumeration done in {time.time()-started:.1f}s "
          f"(total candidates: {total:,})")
    return {"global_xi": np.vstack(all_x), "global_xcf": np.vstack(all_xcf),
            "per_ind_meta": metadata, "ind_sizes": sizes,
            "sex_labels": groups, "factual_pos": factual_positions,
            "total_cands": total}


def _predict(torch_pipe, features, device):
    result = np.empty(len(features), dtype=np.float32)
    torch_pipe.eval()
    with torch.no_grad():
        for start in range(0, len(features), _BATCH):
            end = min(start + _BATCH, len(features))
            tensor = torch.as_tensor(features[start:end], dtype=torch.float32,
                                     device=device)
            result[start:end] = torch_pipe(tensor).cpu().numpy()
    return result


def score_pool(torch_pipe, pool, threshold, nu, slug, save_dir, n_max,
               use_cache=True, device=None):
    """Score candidates; third return value is the factual direct gap.

    It is intentionally not labelled NIE: it compares predictions at two X
    values while holding a factual mediator fixed.
    """
    os.makedirs(save_dir, exist_ok=True)
    cache = (f"{save_dir}/scored_v2_{slug}_n{n_max}_tau{threshold:.2f}"
             f"_nu{nu:.2f}.pkl")
    if use_cache and os.path.exists(cache):
        with open(cache, "rb") as f:
            saved = pickle.load(f)
        if saved.get("format") == "direct_gap_v2":
            return (saved["all_candidates"], saved["groups"],
                    saved["factual_direct_gap"])

    print(f"  Running global PyTorch inference on {pool['total_cands']:,} "
          "candidates x 2 sensitive values...")
    p_x = _predict(torch_pipe, pool["global_xi"], device)
    p_xcf = _predict(torch_pipe, pool["global_xcf"], device)
    shortfall = (np.maximum(0.0, threshold - nu - p_x)
                 + np.maximum(0.0, threshold - nu - p_xcf))
    offsets = np.concatenate([[0], np.cumsum(pool["ind_sizes"])])
    all_candidates, factual_direct_gap = [], []
    for i, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        costs, changes = pool["per_ind_meta"][i]
        records = []
        for j in range(end - start):
            records.append({
                "cost": float(costs[j]),
                "shortfall": float(shortfall[start + j]),
                "p_xi": float(p_x[start + j]),
                "p_xcf": float(p_xcf[start + j]),
                **{key: (int(value[j]) if value.dtype == np.int32
                         else float(value[j])) for key, value in changes.items()},
            })
        all_candidates.append(records)
        factual = records[pool["factual_pos"][i]]
        factual_direct_gap.append(float(factual["p_xcf"] - factual["p_xi"]))
    if use_cache:
        with open(cache, "wb") as f:
            pickle.dump({"format": "direct_gap_v2",
                         "all_candidates": all_candidates,
                         "groups": pool["sex_labels"],
                         "factual_direct_gap": factual_direct_gap}, f, protocol=4)
    return all_candidates, pool["sex_labels"], factual_direct_gap
