"""Sensitivity of effects and recourse to latent X--W confounding.

Uses ``data.german_synth_confounded``: a latent U shifts group membership and
every ordinal mediator with strength delta, which violates W_s _||_ X | Z.
Each configuration (delta, proxy) is refitted end to end per seed: mediator
model and classifiers are trained on observed data only, with or without the
proxy PROXY = U + sigma * noise in Z. Selection uses the fitted model, as in
practice. Evaluation compares, against the true SCM,

* the fitted vs. true interventional NIE (and NDE) on the classifier scale;
* recourse closure measured with the fitted model (as reported in the paper)
  vs. with the true reference P(W_{x1} | z) and the true interventional law of
  each selected plan.

Usage
-----
  python -m evaluation.confounding_sensitivity --stage fit --workers 4
  python -m evaluation.confounding_sensitivity --stage eval --workers 4
  python -m evaluation.confounding_sensitivity --stage summary
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

DATASET = "german_synth_conf"
MODELS = [("Logistic Reg.", "logreg"), ("MLP (64--32)", "mlp"),
          ("Random Forest", "random_forest")]
GAMMA, EPSILON, THRESHOLD, NU = 0.5, 0.1, 0.5, 0.05
N_RECIPIENTS, K_REF, K_INT, K_TRUE = 250, 200, 32, 64


def _tag(delta, proxy):
    return f"delta_{delta:g}_proxy_{proxy}"


def _paths(root, delta, proxy, seed):
    base = root / _tag(delta, proxy) / f"seed_{seed}"
    return {"base": base, "tensors": base / "data" / "tensors.pt",
            "flows": base / "flows" / "flow_models.pt",
            "outcome": base / "outcome", "logs": base / "logs",
            "eval": base / "eval.json"}


def _overrides(delta, proxy, seed, paths):
    uses_proxy = proxy != "none"
    sigma = float(proxy) if uses_proxy else 0.5
    out = [f"dataset={DATASET}", f"seed={seed}",
           f"dataset.loader.kwargs.delta={delta}",
           f"dataset.loader.kwargs.proxy_sigma={sigma}",
           f"dataset.paths.tensors={paths['tensors']}",
           f"dataset.paths.flows={paths['flows']}",
           f"dataset.paths.outcome_dir={paths['outcome']}"]
    if uses_proxy:
        out += ["dataset.sfm.confounders=[AGE,PROXY]",
                "dataset.sfm.standardize_confounders=[AGE,PROXY]",
                "dataset.outcome.col_types.Z=[scale,scale]"]
    return out


def fit_one(job):
    root, delta, proxy, seed = job
    paths = _paths(Path(root), delta, proxy, seed)
    if paths["flows"].exists() and (paths["outcome"] / "metrics.json").exists():
        return f"skip fit {_tag(delta, proxy)} seed {seed}"
    paths["logs"].mkdir(parents=True, exist_ok=True)
    ov = _overrides(delta, proxy, seed, paths)
    for stage, cmd in (
            ("flow", ["-m", "flows.train_flow", *ov,
                      "dataset.flow.reuse_tensors=false"]),
            ("outcome", ["-m", "outcome.train_outcome", *ov])):
        with (paths["logs"] / f"{stage}.log").open("w") as log:
            subprocess.run([sys.executable, *cmd], stdout=log,
                           stderr=subprocess.STDOUT, check=True)
    return f"fit {_tag(delta, proxy)} seed {seed}"


def _closure(pre, post):
    return 1.0 - np.abs(post).mean() / np.abs(pre).mean()


def eval_one(job):
    root, delta, proxy, seed = job
    import joblib
    import torch
    from omegaconf import OmegaConf

    from data.build_tensors import stack_sfm_features
    from data.german_synth_confounded import (
        make_oracle_sample_fn, sample_true_plan)
    from evaluation.compute_recourse import (
        compute_weights, precompute_interventional_candidates_batch)
    from evaluation.estimate_gap import compute_gap_stats
    from evaluation.recourse_metrics import (
        estimate_reference_terms, select_distribution_constrained,
        select_recourse_indices, select_validity_constrained)
    from flows.diagnostics import make_sample_fns
    from flows.models import load_flow_models
    from flows.schema import MediatorSchema

    paths = _paths(Path(root), delta, proxy, seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 20_000)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    uses_proxy = proxy != "none"
    sigma = float(proxy) if uses_proxy else 0.5
    cfg = OmegaConf.load(f"conf/dataset/{DATASET}.yaml")
    rec = cfg.recourse
    specs = OmegaConf.to_container(rec.mediators, resolve=True)
    data = torch.load(paths["tensors"], map_location="cpu", weights_only=False)
    scaler, vocab, sfm = data["scaler"], data["vocab"], data["cfg"]
    disc, cont = list(sfm["mediators_disc"]), list(sfm["mediators_cont"])
    weights = compute_weights(data, disc, cont)
    g_phi, f_theta, _, flow_sfm = load_flow_models(str(paths["flows"]), device)
    schema = MediatorSchema.from_sfm_config(flow_sfm)
    fitted_fn, _ = make_sample_fns(g_phi, f_theta, scaler, device)
    oracle_fn = make_oracle_sample_fn(scaler, seed=seed + 1, delta=delta,
                                      proxy_sigma=sigma, uses_proxy=uses_proxy)

    def score(pipe, x, z, wd, wc_natural):
        wc_std = scaler.transform(wc_natural).astype(np.float32)
        feat = stack_sfm_features(
            torch.as_tensor(x, dtype=torch.float32),
            torch.as_tensor(z, dtype=torch.float32),
            torch.as_tensor(wd, dtype=torch.long),
            torch.as_tensor(wc_std), scaler)
        return pipe.predict_proba(feat)[:, 1]

    result = {"delta": delta, "proxy": proxy, "seed": seed, "models": {}}
    for name, slug in MODELS:
        pipe = joblib.load(paths["outcome"] / f"{slug}.joblib")
        fit_gap = compute_gap_stats(pipe, scaler, data, fitted_fn, 500, 500,
                                    200, rng_seed=seed)
        true_gap = compute_gap_stats(pipe, scaler, data, oracle_fn, 500, 500,
                                     200, rng_seed=seed)
        idx = select_recourse_indices(data, pipe, scaler, 0, N_RECIPIENTS,
                                      rng_seed=seed)
        Z = data["Z_va"][idx]
        ref_fit = estimate_reference_terms(pipe, scaler, Z, fitted_fn, K=K_REF)
        ref_true = estimate_reference_terms(pipe, scaler, Z, oracle_fn,
                                            K=K_REF)
        per = {m: {"pre_fit": [], "post_fit": [], "pre_true": [],
                   "post_true": [], "abstain": []}
               for m in ("ordinary", "ours")}
        for i, j in enumerate(idx):
            x, z = data["X_va"][j:j + 1], data["Z_va"][j:j + 1]
            wd, wc = data["Wd_va"][j:j + 1], data["Wc_va"][j:j + 1]
            wd_np = wd[0].numpy()
            wc_nat = scaler.inverse_transform(wc.numpy())[0]
            cands = precompute_interventional_candidates_batch(
                pipe, scaler, x, z, wd, wc, disc, cont, specs, vocab, weights,
                THRESHOLD, NU, g_phi, f_theta, schema, device,
                ref_fit["wd_reference"][i], ref_fit["wc_reference"][i],
                ref_fit["prediction_reference"][i], n_samples=K_INT,
                same_level="preserve_factual")
            factual = min(cands, key=lambda c: c["cost"])["p_xi"]
            for method, plan in (
                    ("ordinary", select_validity_constrained(cands, GAMMA)),
                    ("ours", select_distribution_constrained(
                        cands, GAMMA, EPSILON))):
                t_disc = np.array([int(round(wd_np[k] + plan[f"delta_{n}"]))
                                   for k, n in enumerate(disc)])
                t_cont = np.array([wc_nat[k] + plan[f"delta_{n}"]
                                   for k, n in enumerate(cont)])
                draws_d, draws_c = sample_true_plan(
                    int(x[0, 0]), z[0].numpy(), wd_np, wc_nat, t_disc, t_cont,
                    t_disc != wd_np, ~np.isclose(t_cont, wc_nat, atol=1e-7),
                    K_TRUE, rng, delta=delta, proxy_sigma=sigma,
                    uses_proxy=uses_proxy)
                p_true = score(pipe, np.repeat(x.numpy(), K_TRUE, 0),
                               np.repeat(z.numpy(), K_TRUE, 0), draws_d,
                               draws_c).mean()
                rec_ = per[method]
                rec_["pre_fit"].append(ref_fit["reference"][i] - factual)
                rec_["post_fit"].append(ref_fit["reference"][i] - plan["p_xi"])
                rec_["pre_true"].append(ref_true["reference"][i] - factual)
                rec_["post_true"].append(ref_true["reference"][i] - p_true)
                rec_["abstain"].append(not plan["constraint_feasible"])
        model_out = {
            "nie_fit": fit_gap["overall"]["nie"][0],
            "nie_true": true_gap["overall"]["nie"][0],
            "nde_fit": fit_gap["overall"]["nde"][0],
            "nde_true": true_gap["overall"]["nde"][0],
            "n_recipients": int(len(idx)),
            "reference_validity_fit": float(np.mean(
                (ref_fit["prediction_reference"] >= THRESHOLD).mean(1)
                >= GAMMA)),
            "reference_validity_true": float(np.mean(
                (ref_true["prediction_reference"] >= THRESHOLD).mean(1)
                >= GAMMA)),
            "reference_mean_bias": float(np.mean(
                ref_fit["reference"] - ref_true["reference"])),
        }
        for method, r in per.items():
            a = {k: np.asarray(v, dtype=float) for k, v in r.items()}
            model_out[method] = {
                "closure_fit": float(_closure(a["pre_fit"], a["post_fit"])),
                "closure_true": float(_closure(a["pre_true"], a["post_true"])),
                "abstention": float(a["abstain"].mean()),
                "overshoot_true": float(
                    (a["pre_true"] * a["post_true"] < 0).mean()),
                "abs_post_true": a["post_true"].__abs__().tolist(),
            }
        o, u = model_out["ordinary"], model_out["ours"]
        gain = np.asarray(o.pop("abs_post_true")) - np.asarray(
            u.pop("abs_post_true"))
        model_out["paired_gain_true"] = float(gain.mean())
        result["models"][name] = model_out
    paths["eval"].write_text(json.dumps(result, indent=2))
    return f"eval {_tag(delta, proxy)} seed {seed}"


def summarize(root, deltas, proxies, seeds):
    lines = ["# Latent-confounding sensitivity (German-Synth, kappa = 1)", "",
             "Mean over seeds and the three classifiers (SD across seeds of "
             "the classifier average). Closure in %. 'fit' is measured with "
             "the fitted mediator model, 'true' against the SCM.", "",
             "| delta | proxy | NIE true | NIE bias | Ref. valid fit/true | "
             "Ord. closure true | Ours closure fit | Ours closure true | "
             "Ours abst. | Gain true |",
             "|---:|---|---:|---:|---|---:|---:|---:|---:|---:|"]
    table = []
    for delta in deltas:
        for proxy in proxies:
            runs = [json.loads(_paths(root, delta, proxy, s)["eval"]
                               .read_text())
                    for s in seeds if _paths(root, delta, proxy, s)["eval"]
                    .exists()]
            if not runs:
                continue

            def stat(fn):
                per_seed = np.array([np.mean([fn(m) for m in r["models"]
                                              .values()]) for r in runs])
                return per_seed.mean(), (per_seed.std(ddof=1)
                                         if len(per_seed) > 1 else 0.0)
            row = {
                "delta": delta, "proxy": proxy, "n_seeds": len(runs),
                "nie_true": stat(lambda m: m["nie_true"]),
                "nie_bias": stat(lambda m: m["nie_fit"] - m["nie_true"]),
                "ref_valid_fit": stat(lambda m: m["reference_validity_fit"]),
                "ref_valid_true": stat(lambda m: m["reference_validity_true"]),
                "ord_true": stat(lambda m: m["ordinary"]["closure_true"]),
                "ours_fit": stat(lambda m: m["ours"]["closure_fit"]),
                "ours_true": stat(lambda m: m["ours"]["closure_true"]),
                "ours_abst": stat(lambda m: m["ours"]["abstention"]),
                "gain_true": stat(lambda m: m["paired_gain_true"]),
            }
            table.append(row)
            f = lambda v, s=100, d=1: f"{s * v[0]:.{d}f} ± {s * v[1]:.{d}f}"
            lines.append(
                f"| {delta:g} | {proxy} | {f(row['nie_true'], 1, 3)} | "
                f"{f(row['nie_bias'], 1, 3)} | "
                f"{100 * row['ref_valid_fit'][0]:.0f} / "
                f"{100 * row['ref_valid_true'][0]:.0f} | "
                f"{f(row['ord_true'])} | {f(row['ours_fit'])} | "
                f"{f(row['ours_true'])} | {f(row['ours_abst'])} | "
                f"{f(row['gain_true'], 1, 3)} |")
    (root / "summary.md").write_text("\n".join(lines) + "\n")
    (root / "summary.json").write_text(json.dumps(table, indent=2))
    Path("drafts/appendix_confounding_table.tex").write_text(latex(table))
    print("\n".join(lines))


def _corr_x_u(delta):
    from data.german_synth_confounded import sample_population
    df = sample_population(10_000, 0, delta)
    return float(np.corrcoef(df["SEX"], df["U"])[0, 1])


def latex(table):
    def ms(v, scale=100.0, digits=1, lead=True, sign=False):
        m, s = scale * v[0], scale * v[1]
        if round(m, digits) == 0:
            m = 0.0
        text = f"{m:+.{digits}f}" if sign else f"{m:.{digits}f}"
        sd = f"{s:.{digits}f}"
        if not lead:
            text, sd = text.replace("0.", ".", 1), sd.replace("0.", ".", 1)
        return (text + r" $\pm$ " + sd).replace("-", "$-$")

    lines = [r"""% Generated by evaluation/confounding_sensitivity.py; do not edit by hand.
\begin{table*}[t]
\centering
\footnotesize
\caption{Sensitivity to latent confounding of $X$ and $W$ (German-Synth,
$\kappa=1$, $\gamma=0.5$, $\varepsilon=0.1$). A latent $U\sim\mathcal N(0,1)$
sets $\Pr[X=1\mid U]=\sigma(\delta U)$ and adds $\delta U$ to the latent mean
of every ordinal mediator, violating $W_s\perp X\mid Z$;
$\rho$ is the resulting correlation of $X$ and $U$. The mediator model and
classifiers are refitted on observed data, with $Z$ either age alone
(``none'') or age and a proxy $V=U+s\,\xi$, $\xi\sim\mathcal N(0,1)$. Plans
are selected with the fitted model. ``True'' quantities are computed from the
SCM: the interventional $\NIE$, the reference $P(W_{x^{(1)}}\mid z)$, and the
interventional law of each selected plan, with $U$ drawn from its posterior
given the recipient's group, $z$, and non-descendant mediators. ``Model''
closure is what the fitted model reports. Ref.\ valid is the true share of
$\gamma$-valid references. Gain is the true paired reduction in
$|D_{\mathrm{post}}|$ relative to ordinary recourse. Closure and abstention in
\%; mean $\pm$ SD over five seeds of the average over three classifiers.}
\label{tab:app_confounding}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{rrl cc c c cc c c}
\hline
& & & \multicolumn{2}{c}{$\NIE$} & & Ord. & \multicolumn{2}{c}{Ours} & & \\
$\delta$ & $\rho$ & Proxy & True & Bias & Ref.\ valid & True closure &
Model closure & True closure & Abst. & Gain \\
\hline"""]
    corr = {}
    for k, row in enumerate(table):
        d = row["delta"]
        if d not in corr:
            corr[d] = _corr_x_u(d)
        first = k == 0 or table[k - 1]["delta"] != d
        proxy = "none" if row["proxy"] == "none" else f"$s={row['proxy']}$"
        lines.append(" & ".join([
            f"{d:g}" if first else "",
            f"{max(corr[d], 0.0):.2f}".replace("0.", ".", 1) if first else "",
            proxy,
            ms(row["nie_true"], 1, 3, lead=False),
            ms(row["nie_bias"], 1, 3, lead=False, sign=True),
            f"{100 * row['ref_valid_true'][0]:.0f}",
            ms(row["ord_true"]), ms(row["ours_fit"]), ms(row["ours_true"]),
            ms(row["ours_abst"]),
            ms(row["gain_true"], 1, 3, lead=False),
        ]) + r" \\")
        last = k == len(table) - 1 or table[k + 1]["delta"] != d
        if last:
            lines.append(r"\hline")
    lines += [r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["fit", "eval", "summary"],
                   required=True)
    p.add_argument("--root", type=Path, default=Path("outputs/confounding"))
    p.add_argument("--deltas", type=float, nargs="+",
                   default=[0.0, 0.5, 1.0, 2.0])
    p.add_argument("--proxies", nargs="+",
                   default=["none", "1.0", "0.5", "0.1"])
    p.add_argument("--seeds", type=int, nargs="+",
                   default=[42, 43, 44, 45, 46])
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    if args.stage == "summary":
        summarize(args.root, args.deltas, args.proxies, args.seeds)
        return
    jobs = [(str(args.root), d, px, s) for d in args.deltas
            for px in args.proxies for s in args.seeds]
    fn = fit_one if args.stage == "fit" else eval_one
    with ProcessPoolExecutor(args.workers) as pool:
        for message in pool.map(fn, jobs):
            print(message, flush=True)


if __name__ == "__main__":
    main()
