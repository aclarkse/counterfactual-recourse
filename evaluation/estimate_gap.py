"""Estimate classifier-scale mediation effects by direct flow sampling.

The estimands concern the fixed classifier ``f_hat`` rather than the observed
outcome Y. For baseline x0 and contrast x1 we estimate
``mu_ab(z) = E[f_hat(x=a, W_x=b, z) | Z=z]``. This makes interaction explicit:
the total effect is estimated directly and is never constructed as NDE + NIE.
"""

import json
import os
import time
import warnings

warnings.filterwarnings("ignore")

import hydra
import joblib
import numpy as np
import torch
from omegaconf import OmegaConf

from data.build_tensors import stack_sfm_features
from flows.diagnostics import make_sample_fns
from flows.models import load_flow_models


def make_outcome_fn(pipe, scaler):
    """Wrap an sklearn pipeline as ``f(x, wd, wc_standardized, z)``."""
    def fn(x, wd, wc_std, z):
        feat = stack_sfm_features(x, z, wd, wc_std, scaler)
        return torch.as_tensor(pipe.predict_proba(feat)[:, 1].astype(np.float32))
    return fn


def estimate_mediation_effects(Z_val, sample_fn, outcome_fn, K=500,
                               batch_size=50, x0=0.0, x1=1.0):
    """Return individual-level pure/total effects on the classifier scale.

    Direct draws from both mediator distributions avoid importance-weight reuse
    and provide an independently checkable plug-in estimator.
    """
    names = ("mu00", "mu10", "mu01", "mu11", "nde", "nie", "tnde",
             "tnie", "te", "interaction")
    components = {name: [] for name in names}

    for start in range(0, len(Z_val), batch_size):
        z = Z_val[start:start + batch_size]
        b = len(z)
        x0_batch = torch.full((b, 1), float(x0))
        x1_batch = torch.full((b, 1), float(x1))
        with torch.no_grad():
            w0 = sample_fn(x0_batch, z, K=K)
            w1 = sample_fn(x1_batch, z, K=K)
            x0_rep = torch.full((b * K, 1), float(x0))
            x1_rep = torch.full((b * K, 1), float(x1))
            mu00 = outcome_fn(x0_rep, w0["w_disc"], w0["w_cont"],
                              w0["z_rep"]).view(b, K).mean(1)
            mu10 = outcome_fn(x1_rep, w0["w_disc"], w0["w_cont"],
                              w0["z_rep"]).view(b, K).mean(1)
            mu01 = outcome_fn(x0_rep, w1["w_disc"], w1["w_cont"],
                              w1["z_rep"]).view(b, K).mean(1)
            mu11 = outcome_fn(x1_rep, w1["w_disc"], w1["w_cont"],
                              w1["z_rep"]).view(b, K).mean(1)

        values = {
            "mu00": mu00, "mu10": mu10, "mu01": mu01, "mu11": mu11,
            "nde": mu10 - mu00,             # pure NDE
            "nie": mu01 - mu00,             # pure NIE
            "tnde": mu11 - mu01,            # total NDE
            "tnie": mu11 - mu10,            # total NIE
            "te": mu11 - mu00,               # direct estimate
            "interaction": mu11 - mu10 - mu01 + mu00,
        }
        for name, value in values.items():
            components[name].extend(value.cpu().numpy().tolist())
    return {name: np.asarray(value) for name, value in components.items()}


def bootstrap_ci(arr, n_boot=2000, alpha=0.05, rng=None):
    arr = np.asarray(arr, dtype=float)
    if len(arr) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(0) if rng is None else rng
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    return (float(arr.mean()),
            float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def compute_raw_gap(pipe, scaler, data):
    """Observed classifier prediction gap, group 1 minus group 0."""
    mask0 = (data["X_va"][:, 0] == 0).numpy()
    mask1 = (data["X_va"][:, 0] == 1).numpy()

    def predict(mask):
        feat = stack_sfm_features(data["X_va"][mask], data["Z_va"][mask],
                                  data["Wd_va"][mask], data["Wc_va"][mask],
                                  scaler)
        return float(pipe.predict_proba(feat)[:, 1].mean())

    p0, p1 = predict(mask0), predict(mask1)
    return p1 - p0, p1, p0


def compute_gap_stats(pipe, scaler, data, sample_fn, K, n_inst, n_boot,
                      rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    n = min(int(n_inst), len(data["X_va"]))
    idx = np.sort(rng.choice(len(data["X_va"]), n, replace=False))
    x_sub, z_sub = data["X_va"][idx], data["Z_va"][idx]

    print(f"    Directly sampling W|x,z (K={K}, n={n})...")
    started = time.time()
    effects = estimate_mediation_effects(
        z_sub, sample_fn, make_outcome_fn(pipe, scaler), K=int(K)
    )
    print(f"    Done in {time.time() - started:.1f}s")
    effect_names = ("nde", "nie", "tnde", "tnie", "te", "interaction")

    def summarize(mask):
        result = {name: bootstrap_ci(effects[name][mask], n_boot)
                  for name in effect_names}
        te = result["te"][0]
        result["addressability"] = (float(abs(result["nie"][0] / te))
                                      if abs(te) > 1e-12 else float("nan"))
        result["n"] = int(np.asarray(mask).sum())
        return result

    res = {"overall": summarize(np.ones(n, dtype=bool))}
    for value, label in ((0, "g0"), (1, "g1")):
        mask = x_sub[:, 0].numpy() == value
        if mask.sum() >= 10:
            res[label] = summarize(mask)
    raw, p1, p0 = compute_raw_gap(pipe, scaler, data)
    res.update(raw=raw, p_g1=p1, p_g0=p0)
    nie = res["overall"]["nie"][0]
    res["effect_ratio_heuristic"] = (
        float(abs(res["overall"]["nde"][0] / nie))
        if abs(nie) > 1e-12 else float("nan")
    )
    return res


def _cell(ci, digits=4):
    return rf"${ci[0]:+.{digits}f}\ [{ci[1]:+.{digits}f},\ {ci[2]:+.{digits}f}]$"


def make_table(results, model_names, caption, label):
    lines = [r"\begin{table}[htbp]", r"  \centering",
             f"  \\caption{{{caption}}}", f"  \\label{{{label}}}",
             r"  \begin{tabular}{lccccc}", r"    \toprule",
             r"    Model & pure NDE & pure NIE & TE & interaction & $|\mathrm{NIE}/\mathrm{TE}|$ \\",
             r"    \midrule"]
    for name, res in zip(model_names, results):
        row = res["overall"]
        line = (f"    {name} & {_cell(row['nde'])} & {_cell(row['nie'])} & "
                f"{_cell(row['te'])} & {_cell(row['interaction'])} & "
                f"{row['addressability']:.3f} ")
        lines.append(line + r"\\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def make_summary(results, model_names):
    lines = []
    for name, res in zip(model_names, results):
        row = res["overall"]
        lines.extend([
            f"{name} (n={row['n']})",
            f"  observed prediction gap: {res['raw']:+.4f}",
            f"  pure NDE: {row['nde'][0]:+.4f} [{row['nde'][1]:+.4f}, {row['nde'][2]:+.4f}]",
            f"  pure NIE: {row['nie'][0]:+.4f} [{row['nie'][1]:+.4f}, {row['nie'][2]:+.4f}]",
            f"  total effect (direct): {row['te'][0]:+.4f} [{row['te'][1]:+.4f}, {row['te'][2]:+.4f}]",
            f"  X-W interaction residual: {row['interaction'][0]:+.4f}",
            f"  addressability |NIE/TE|: {row['addressability']:.3f}",
            f"  effect-ratio heuristic |NDE/NIE|: {res['effect_ratio_heuristic']:.3f}",
            "",
        ])
    text = "\n".join(lines).rstrip()
    print("\n" + text)
    return text


@hydra.main(config_path="../conf", config_name="config", version_base="1.1")
def main(cfg):
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    g_phi, f_theta, scaler, _ = load_flow_models(cfg.dataset.paths.flows, device)
    sample_fn, _ = make_sample_fns(g_phi, f_theta, scaler, device)
    data = torch.load(cfg.dataset.paths.tensors, map_location="cpu", weights_only=False)
    specs = OmegaConf.to_container(cfg.dataset.outcome.models, resolve=True)
    gap_cfg = cfg.dataset.gap

    results, names = [], []
    for spec in specs:
        name = spec["name"]
        slug = {"logreg": "logreg", "mlp": "mlp"}.get(spec["type"], spec["type"])
        pipe = joblib.load(f"{cfg.dataset.paths.outcome_dir}/{slug}.joblib")
        print(f"\n[{name}]")
        results.append(compute_gap_stats(pipe, scaler, data, sample_fn,
                                         gap_cfg.K, gap_cfg.n_inst,
                                         gap_cfg.n_boot,
                                         rng_seed=int(cfg.seed)))
        names.append(name)

    out_dir = cfg.dataset.paths.gaps_dir
    os.makedirs(out_dir, exist_ok=True)
    stem = f"{out_dir}/{cfg.dataset.name}_gender_gap"
    summary = make_summary(results, names)
    table = make_table(
        results, names,
        "Classifier-scale mediation effects estimated by direct conditional sampling. "
        "TE is estimated directly; interaction is TE minus pure NDE minus pure NIE. "
        "Intervals hold the fitted flow and classifier fixed.",
        f"tab:{cfg.dataset.name}_gap_overall",
    )
    snapshot = {}
    for name, res in zip(names, results):
        snapshot[name] = {
            stratum: ({effect: res[stratum][effect][0]
                       for effect in ("nde", "nie", "tnde", "tnie", "te",
                                      "interaction")} |
                      {"addressability": res[stratum]["addressability"],
                       "n": res[stratum]["n"]})
            for stratum in ("overall", "g0", "g1") if stratum in res
        }
        snapshot[name]["effect_ratio_heuristic"] = res["effect_ratio_heuristic"]
        snapshot[name]["raw_prediction_gap"] = res["raw"]
    snapshot["_metadata"] = {
        "seed": int(cfg.seed), "K": int(gap_cfg.K),
        "n_inst": int(gap_cfg.n_inst), "n_boot": int(gap_cfg.n_boot),
        "estimand_scale": "fixed classifier prediction",
        "sampling": "direct draws from both conditional mediator models",
        "interval_scope": "individual bootstrap; fitted models held fixed",
    }

    with open(stem + ".json", "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
    with open(stem + ".txt", "w", encoding="utf-8") as f:
        f.write(summary)
    with open(stem + "_overall.tex", "w", encoding="utf-8") as f:
        f.write(table)
    print(f"\nSaved → {stem}.json/.txt and {stem}_overall.tex")


if __name__ == "__main__":
    main()
