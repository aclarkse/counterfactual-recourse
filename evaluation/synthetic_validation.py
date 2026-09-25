"""Synthetic ground-truth check for the mediation estimator.

The linear SCM includes an X-W interaction specifically to verify that TE is
estimated directly rather than incorrectly set to pure NDE + pure NIE.
"""

import argparse
import json

import numpy as np
import torch

from evaluation.estimate_gap import estimate_mediation_effects


def run(n=1000, K=2000, seed=42):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    z = torch.as_tensor(rng.normal(size=(n, 1)).astype(np.float32))
    alpha, gamma, sigma = 0.8, 0.5, 1.0
    beta0, beta_x, beta_w, beta_xw = 0.1, 0.3, 0.4, 0.25

    def sample_fn(x, z_value, K):
        x_rep = x.repeat_interleave(K, 0)
        z_rep = z_value.repeat_interleave(K, 0)
        noise = torch.randn(len(x_rep), 1) * sigma
        w = alpha * x_rep + gamma * z_rep + noise
        return {"w_disc": torch.empty(len(w), 0, dtype=torch.long),
                "w_cont": w, "z_rep": z_rep}

    def outcome_fn(x, wd, w, z_value):
        del wd, z_value
        return (beta0 + beta_x * x[:, 0] + beta_w * w[:, 0]
                + beta_xw * x[:, 0] * w[:, 0])

    estimated = estimate_mediation_effects(z, sample_fn, outcome_fn, K=K)
    mean_z = float(z.mean())
    exact = {
        "nde": beta_x + beta_xw * gamma * mean_z,
        "nie": beta_w * alpha,
        "te": (beta_x + beta_w * alpha + beta_xw * alpha
               + beta_xw * gamma * mean_z),
        "interaction": beta_xw * (alpha + gamma * mean_z),
    }
    result = {
        "n": n, "K": K, "seed": seed,
        "estimand": "classifier/structural-score scale",
        "parameters": {"alpha": alpha, "gamma": gamma, "sigma": sigma,
                       "beta_x": beta_x, "beta_w": beta_w,
                       "beta_xw": beta_xw},
        "effects": {
            name: {"truth": truth, "estimate": float(estimated[name].mean()),
                   "absolute_error": abs(float(estimated[name].mean()) - truth)}
            for name, truth in exact.items()
        },
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--K", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="outputs/synthetic_validation.json")
    args = parser.parse_args()
    result = run(args.n, args.K, args.seed)
    print(json.dumps(result, indent=2))
    import os
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
