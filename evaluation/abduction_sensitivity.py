"""Fresh-noise vs abducted-noise evaluation of selected plans (German-Synth).

The post-action law Q_a in the paper is interventional: mediator blocks after
the earliest acted block are redrawn with fresh noise. This script asks how
the selected plans a* fare if the recipient instead keeps their own latent
noise, drawn from the posterior p(U | x0, w0, z).

For each seed of the published multi-seed German-Synth run (kappa = 1), the
recourse step is rerun with the same RNG order as ``evaluation.ablate_recourse``
so the selected plans are exactly the published ones (checked against the
saved ablation JSONs). Each plan of ordinary recourse and of our method is
then evaluated under four post-action laws:

  fit_fresh    fitted mediator model, fresh noise (as in the paper, but with
               draws independent of those used to select the plan)
  fit_abduct   fitted model, noise abducted from the fitted model's posterior
  true_fresh   true SCM, fresh noise
  true_abduct  true SCM, noise abducted from the true posterior

Abduction:
  * ordinal mediators use inverse-CDF noise in category order:
    U | w0, pa0 ~ Uniform(F(w0 - 1 | pa0), F(w0 | pa0)), new value
    F^{-1}(U | pa'). This coupling is not identified from data; for the true
    ordered-logit SCM it is exactly the retained logistic noise.
  * the continuous block inverts the fitted flow at the factual context and
    pushes the noise forward at the new context. The true SCM recovers the
    bivariate normal noise from the recipient's raw values, sampling it by
    rejection when a value sits at a clip bound.

No global torch or numpy RNG is consumed by the evaluation, so later
classifiers in the same seed still reproduce the published plans.

Usage
-----
  python -m evaluation.abduction_sensitivity --stage eval --workers 5
  python -m evaluation.abduction_sensitivity --stage summary
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

DATASET = "german_synth"
MODELS = [("logistic_reg", "LR"), ("mlp_6432", "MLP"), ("random_forest", "RF")]
METHODS = ("ordinary", "ours")
MODES = ("fit_fresh", "fit_abduct", "true_fresh", "true_abduct")
GAMMA, EPSILON, THRESHOLD = 0.5, 0.1, 0.5
K_EVAL, K_REF_TRUE = 128, 200
DISC = ["EDUCATION", "JOB", "SAVINGS_GRP"]
CONT = ["CREDIT_AMOUNT", "DURATION"]
DISC_LAYER = [0, 1, 2]     # the continuous block is layer 3
CONT_LAYER = 3
NO_ACTION = 4
TABLE_PATH = Path("drafts/appendix_abduction_table.tex")


# --------------------------------------------------------------------------
# Fitted mediator model
# --------------------------------------------------------------------------
def _head_probs(g_phi, j, context, w_disc):
    import torch
    parts = [context] + [g_phi.embeddings[k](w_disc[:, k])
                         for k in g_phi.parent_indices[j]]
    logits = g_phi.heads[j](torch.cat(parts, 1))
    return torch.softmax(logits, -1).double().cpu().numpy()


def _inverse_cdf(cdf, u):
    return np.minimum((u[:, None] > cdf).sum(1), cdf.shape[1] - 1)


def fitted_draws(g_phi, f_theta, x, z, wd, wc, target_disc, target_cont_flow,
                 mask_disc, mask_cont, first, mode, rng):
    """Post-action mediators under the fitted model; wc in flow space."""
    import torch
    with torch.no_grad():
        dev = next(g_phi.parameters()).device
        X, Z = x.repeat(K_EVAL, 1).to(dev), z.repeat(K_EVAL, 1).to(dev)
        factual = wd.repeat(K_EVAL, 1).to(dev)
        cur = factual.clone()
        context = g_phi._cond(X, Z)
        for j in range(len(DISC)):
            if mask_disc[j]:
                cur[:, j] = int(target_disc[j])
                continue
            if DISC_LAYER[j] <= first:
                continue  # non-descendant or unacted same-block: factual
            cdf_new = np.cumsum(_head_probs(g_phi, j, context, cur), 1)
            if mode == "fresh":
                u = rng.random(K_EVAL)
            else:
                p_f = _head_probs(g_phi, j, context, factual)
                w0 = int(wd[0, j])
                lo = np.cumsum(p_f, 1)[:, w0 - 1] if w0 > 0 else np.zeros(K_EVAL)
                u = lo + rng.random(K_EVAL) * p_f[:, w0]
            cur[:, j] = torch.as_tensor(_inverse_cdf(cdf_new, u), device=dev)
        wc_rep = wc.repeat(K_EVAL, 1).to(dev)
        if first < CONT_LAYER:
            flow = f_theta.flow
            if mode == "fresh":
                noise = torch.as_tensor(rng.standard_normal((K_EVAL, len(CONT))),
                                        dtype=torch.float32, device=dev)
            else:
                noise = flow.transform_to_noise(
                    wc_rep, context=f_theta._context(factual, X, Z))
            wc_rep = flow._transform.inverse(
                noise, context=flow._embedding_net(f_theta._context(cur, X, Z)))[0]
        wc_rep = torch.where(torch.as_tensor(mask_cont, device=dev)[None],
                             torch.as_tensor(target_cont_flow, device=dev)[None],
                             wc_rep)
    return cur.cpu(), wc_rep.cpu()


# --------------------------------------------------------------------------
# True SCM (data.german_synth, kappa = 1)
# --------------------------------------------------------------------------
def _sigmoid(v):
    return 1.0 / (1.0 + np.exp(-v))


def _ordinal_mean(gs, j, sex, a, edu, job):
    if j == 0:
        return gs._EDU["sex"] * sex + gs._EDU["age"] * a
    if j == 1:
        return gs._JOB["sex"] * sex + gs._JOB["age"] * a + gs._JOB["edu"] * edu
    return gs._SAV["sex"] * sex + gs._SAV["age"] * a + gs._SAV["job"] * job


def _log_cont_means(gs, a, job, sav):
    am, du = gs._AMOUNT, gs._DURATION
    return (am["intercept"] + am["age"] * a + am["job"] * job + am["sav"] * sav,
            du["intercept"] + du["age"] * a + du["job"] * job + du["sav"] * sav)


def _abduct_cont_noise(gs, a, job, sav, amount, duration, rng):
    """Posterior draws of the bivariate amount/duration noise."""
    means = _log_cont_means(gs, a, job, sav)
    sds = (gs._AMOUNT["noise"], gs._DURATION["noise"])
    logs = (np.log(amount), np.log(duration))
    bounds = [(np.log(r[0]), np.log(r[1]))
              for r in (gs.AMOUNT_RANGE, gs.DURATION_RANGE)]
    u = np.array([(logs[k] - means[k]) / sds[k] for k in range(2)])
    clip = [("lo" if abs(logs[k] - bounds[k][0]) < 1e-6 else
             "hi" if abs(logs[k] - bounds[k][1]) < 1e-6 else None)
            for k in range(2)]
    if clip == [None, None]:
        return np.repeat(u[None], K_EVAL, 0)
    # A clipped value only bounds its noise. Rejection-sample from the prior,
    # conditioning softly (|du| < 0.02) on an unclipped coordinate.
    cov = np.array([[1.0, gs._RHO_LD], [gs._RHO_LD, 1.0]])
    kept, n_kept = [], 0
    while n_kept < K_EVAL:
        d = rng.multivariate_normal([0.0, 0.0], cov, 200_000)
        ok = np.ones(len(d), bool)
        for k, c in enumerate(clip):
            level = means[k] + sds[k] * d[:, k]
            if c is None:
                ok &= np.abs(d[:, k] - u[k]) < 0.02
            elif c == "lo":
                ok &= level <= bounds[k][0]
            else:
                ok &= level >= bounds[k][1]
        d = d[ok]
        for k, c in enumerate(clip):
            if c is None:
                d[:, k] = u[k]
        kept.append(d)
        n_kept += len(d)
    return np.concatenate(kept)[:K_EVAL]


def true_draws(gs, x, age, wd, raw_cont, target_disc, target_cont, mask_disc,
               mask_cont, first, mode, rng):
    """Post-action mediators under the true SCM; continuous in natural units."""
    cuts = [gs._EDU_CUTS, gs._JOB_CUTS, gs._SAV_CUTS]
    sex, a = float(x), float(gs._age_std(np.array([age]))[0])
    cur = np.repeat(np.asarray(wd)[None], K_EVAL, 0).astype(int)
    for j in range(len(DISC)):
        if mask_disc[j]:
            cur[:, j] = int(target_disc[j])
            continue
        if DISC_LAYER[j] <= first:
            continue
        mean_new = _ordinal_mean(gs, j, sex, a, cur[:, 0], cur[:, 1])
        if mode == "fresh":
            eps = rng.logistic(size=K_EVAL)
        else:
            mean_f = _ordinal_mean(gs, j, sex, a, wd[0], wd[1])
            c = np.concatenate([[-np.inf], cuts[j], [np.inf]])
            w0 = int(wd[j])
            lo, hi = _sigmoid(c[w0] - mean_f), _sigmoid(c[w0 + 1] - mean_f)
            p = np.clip(lo + rng.random(K_EVAL) * (hi - lo), 1e-12, 1 - 1e-12)
            eps = np.log(p) - np.log1p(-p)
        cur[:, j] = np.searchsorted(cuts[j], mean_new + eps)
    cont = np.repeat(np.asarray(raw_cont, float)[None], K_EVAL, 0)
    if first < CONT_LAYER:
        if mode == "fresh":
            cov = [[1.0, gs._RHO_LD], [gs._RHO_LD, 1.0]]
            u = rng.multivariate_normal([0.0, 0.0], cov, K_EVAL)
        else:
            u = _abduct_cont_noise(gs, a, wd[1], wd[2], raw_cont[0],
                                   raw_cont[1], rng)
        la, ld = _log_cont_means(gs, a, cur[:, 1], cur[:, 2])
        cont[:, 0] = np.clip(np.exp(la + gs._AMOUNT["noise"] * u[:, 0]),
                             *gs.AMOUNT_RANGE)
        cont[:, 1] = np.clip(np.exp(ld + gs._DURATION["noise"] * u[:, 1]),
                             *gs.DURATION_RANGE)
    for k in range(len(CONT)):
        if mask_cont[k]:
            cont[:, k] = target_cont[k]
    return cur, cont


class RawLookup:
    """Recover exact raw continuous values (the tensors hold them only after a
    lossy quantile transform) by matching age and the ordinal mediators."""

    def __init__(self, gs):
        self.df = gs.sample_german_synthetic(10_000, seed=0, mediation_scale=1.0)
        self.age = self.df["AGE"].to_numpy(np.float32)
        self.disc = self.df[DISC].to_numpy()
        self.cont = self.df[CONT].to_numpy(float)

    def __call__(self, age, wd, wc_natural_approx):
        hit = np.flatnonzero((np.abs(self.age - np.float32(age)) < 1e-4)
                             & (self.disc == wd).all(1))
        if not len(hit):
            raise RuntimeError(f"no raw row for age={age}, mediators={wd}")
        err = np.abs(np.log(self.cont[hit]) - np.log(wc_natural_approx)).sum(1)
        return self.cont[hit[np.argmin(err)]]


# --------------------------------------------------------------------------
# Per-seed evaluation
# --------------------------------------------------------------------------
def eval_seed(job):
    root, seed = job
    import torch
    from hydra import compose, initialize_config_dir
    from scipy.stats import wasserstein_distance

    import evaluation.ablate_recourse as ar
    from data import german_synth as gs
    from data.build_tensors import stack_sfm_features
    from evaluation.recourse_metrics import (
        estimate_reference_terms, select_distribution_constrained,
        select_validity_constrained)

    repo = Path(__file__).resolve().parents[1]
    base = repo / "outputs" / "multiseed" / DATASET / f"seed_{seed}"
    rng = np.random.default_rng(10_000 + seed)
    raw_lookup = RawLookup(gs)
    captured, results = {}, {}
    model_order = iter(slug for slug, _ in MODELS)

    def score(pipe, x, z, wd_t, wc_flow, scaler):
        n = len(wd_t)
        feat = stack_sfm_features(x.repeat(n, 1), z.repeat(n, 1), wd_t,
                                  wc_flow, scaler)
        return pipe.predict_proba(feat)[:, 1]

    def score_natural(pipe, x, z, wd_np, wc_nat, scaler):
        wc_flow = torch.as_tensor(scaler.transform(wc_nat).astype(np.float32))
        return score(pipe, x, z, torch.as_tensor(wd_np, dtype=torch.long),
                     wc_flow, scaler)

    def summary(p, reference_draws):
        return {"p": float(p.mean()), "s": float((p >= THRESHOLD).mean()),
                "w1": float(wasserstein_distance(p, reference_draws))}

    def evaluate(all_candidates, refs, factual_pred, slug):
        data, pipe, idx = captured["data"], captured["pipe"], captured["idx"]
        g_phi, f_theta, scaler = (captured["g_phi"], captured["f_theta"],
                                  data["scaler"])
        oracle = gs.make_oracle_sample_fn(scaler, seed=seed + 1,
                                          mediation_scale=1.0)
        with torch.no_grad():
            ref_true = estimate_reference_terms(
                pipe, scaler, data["Z_va"][idx], oracle, K=K_REF_TRUE)
        out = {m: [] for m in METHODS}
        for i, j in enumerate(idx):
            x, z = data["X_va"][j:j + 1], data["Z_va"][j:j + 1]
            wd, wc = data["Wd_va"][j:j + 1], data["Wc_va"][j:j + 1]
            wd_np = wd[0].numpy()
            wc_nat = scaler.inverse_transform(wc.numpy())[0]
            raw = raw_lookup(float(z[0, 0]), wd_np, wc_nat)
            fact_true = float(score_natural(pipe, x, z, wd_np[None], raw[None],
                                            scaler)[0])
            plans = {
                "ordinary": select_validity_constrained(all_candidates[i], GAMMA),
                "ours": select_distribution_constrained(
                    all_candidates[i], GAMMA, EPSILON),
            }
            for method, plan in plans.items():
                delta_d = np.array([plan[f"delta_{n}"] for n in DISC])
                delta_c = np.array([plan[f"delta_{n}"] for n in CONT])
                t_disc = np.rint(wd_np + delta_d).astype(int)
                t_cont_nat = (wc_nat + delta_c).astype(np.float32)
                m_disc = t_disc != wd_np
                m_cont = ~np.isclose(t_cont_nat, wc_nat, atol=1e-7)
                acted = ([DISC_LAYER[k] for k in range(len(DISC)) if m_disc[k]]
                         + [CONT_LAYER for k in range(len(CONT)) if m_cont[k]])
                first = min(acted) if acted else NO_ACTION
                rec = {
                    "served": bool(plan["constraint_feasible"]),
                    "first_layer": int(first),
                    "ref_fit": float(refs["reference"][i]),
                    "ref_true": float(ref_true["reference"][i]),
                    "fact_fit": float(factual_pred[i]),
                    "fact_true": fact_true,
                    "fit_selection_draws": {
                        "p": plan["p_xi"], "s": plan["success_probability_xi"],
                        "w1": plan["outcome_wasserstein"]},
                }
                t_cont_flow = torch.as_tensor(
                    scaler.transform(t_cont_nat[None]).astype(np.float32)[0])
                t_cont_true = raw + delta_c
                for mode in ("fresh", "abduct"):
                    if first == NO_ACTION:
                        p_fit = np.full(K_EVAL, float(factual_pred[i]))
                        p_true = np.full(K_EVAL, fact_true)
                    else:
                        dd, cc = fitted_draws(
                            g_phi, f_theta, x, z, wd, wc, t_disc, t_cont_flow,
                            m_disc, m_cont, first, mode, rng)
                        p_fit = score(pipe, x, z, dd, cc, scaler)
                        dd, cc = true_draws(
                            gs, float(x[0, 0]), float(z[0, 0]), wd_np, raw,
                            t_disc, t_cont_true, m_disc, m_cont, first, mode,
                            rng)
                        p_true = score_natural(pipe, x, z, dd, cc, scaler)
                    rec[f"fit_{mode}"] = summary(
                        p_fit, refs["prediction_reference"][i])
                    rec[f"true_{mode}"] = summary(
                        p_true, ref_true["prediction_reference"][i])
                out[method].append(rec)
        results[slug] = out

    # Hooks: capture what ablate_recourse.run builds, evaluate after the
    # constrained selection, and restore the torch RNG so later classifiers
    # see the same random stream as the published run.
    originals = (ar.select_recourse_indices, ar.load_flow_models, ar.run_grid)

    def select_hook(data, pipe, *args, **kwargs):
        idx = originals[0](data, pipe, *args, **kwargs)
        captured.update(idx=idx, pipe=pipe, data=data)
        return idx

    def load_hook(*args, **kwargs):
        loaded = originals[1](*args, **kwargs)
        captured.update(g_phi=loaded[0], f_theta=loaded[1])
        return loaded

    rerun_rows = {}

    def grid_hook(all_c, refs, fact, eta, lam, rho, dv, n_boot, cfg_c=None):
        rows = originals[2](all_c, refs, fact, eta, lam, rho, dv, n_boot, cfg_c)
        if cfg_c is not None and eta:
            slug = next(model_order)
            rerun_rows[slug] = rows
            state = torch.get_rng_state()
            evaluate(all_c, refs, fact, slug)
            torch.set_rng_state(state)
        return rows

    ar.select_recourse_indices, ar.load_flow_models, ar.run_grid = (
        select_hook, load_hook, grid_hook)
    try:
        with tempfile.TemporaryDirectory() as tmp, initialize_config_dir(
                config_dir=str(repo / "conf"), version_base="1.1"):
            cfg = compose("config", overrides=[
                f"dataset={DATASET}", f"seed={seed}",
                f"dataset.paths.tensors={base}/data/tensors.pt",
                f"dataset.paths.flows={base}/flows/flow_models.pt",
                f"dataset.paths.outcome_dir={base}/outcome",
                f"dataset.paths.gaps_dir={base}/gaps",
                f"dataset.paths.recourse_dir={tmp}/recourse",
                f"dataset.paths.figures_dir={tmp}/figures",
                "dataset.recourse.constrained_recourse.gamma_sensitivity_grid=[]",
            ])
            ar.run(cfg)
    finally:
        ar.select_recourse_indices, ar.load_flow_models, ar.run_grid = originals

    # The selected plans must be the published ones.
    for slug, rows in rerun_rows.items():
        published = json.loads((base / "recourse" /
                                f"ablation_{DATASET}_{slug}.json").read_text())
        for method in ("ordinary_actionable_recourse",
                       "distribution_constrained_recourse"):
            a = next(r for r in published["rows"] if r["method"] == method)
            b = next(r for r in rows if r["method"] == method)
            for key in ("recipient_level_closure", "constraint_feasible_rate"):
                if abs(a[key][0] - b[key][0]) > 1e-9:
                    raise RuntimeError(
                        f"seed {seed} {slug} {method}: rerun does not reproduce "
                        f"published {key} ({a[key][0]} vs {b[key][0]})")

    out_path = Path(root) / f"seed_{seed}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "seed": seed, "dataset": DATASET, "gamma": GAMMA, "epsilon": EPSILON,
        "k_eval": K_EVAL, "k_ref_true": K_REF_TRUE,
        "reproduces_published_plans": True, "models": results}))
    return f"seed {seed} done"


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
def _metrics(recs, mode, ordinary_recs):
    world = "true" if mode.startswith("true") else "fit"
    ref = np.array([r[f"ref_{world}"] for r in recs])
    fact = np.array([r[f"fact_{world}"] for r in recs])
    served = np.array([r["served"] for r in recs])
    p = np.array([r[mode]["p"] if r["served"] else r[f"fact_{world}"]
                  for r in recs])
    s = np.array([r[mode]["s"] for r in recs])
    w1 = np.array([r[mode]["w1"] for r in recs])
    pre, post = ref - fact, ref - p
    p_ord = np.array([r[mode]["p"] for r in ordinary_recs])
    propagating = served & (np.array([r["first_layer"] for r in recs])
                            < CONT_LAYER)
    other = "fresh" if mode.endswith("abduct") else "abduct"
    p_other = np.array([r[f"{world}_{other}"]["p"] for r in recs])
    return {
        "served": float(served.mean()),
        "valid": float((s[served] >= GAMMA).mean()),
        "score": float(p[served].mean()),
        "w1_within_eps": float((w1[served] <= EPSILON + 1e-9).mean()),
        "closure": float(1 - np.abs(post).mean() / np.abs(pre).mean()),
        "mean_d_served": float(post[served].mean()),
        "overshoot": float(((pre * post) < 0)[served].mean()),
        "gain": float((np.abs(ref - p_ord) - np.abs(post)).mean()),
        "share_propagating": float(propagating.sum() / served.sum()),
        "abduct_minus_fresh": (float(np.mean(
            (p - p_other if mode.endswith("abduct") else p_other - p)[propagating]))
            if propagating.any() else 0.0),
    }


def summarize(root, seeds):
    root = Path(root)
    per_seed = [json.loads((root / f"seed_{s}.json").read_text())
                for s in seeds]
    table = {}
    for slug, _ in MODELS:
        for method in METHODS:
            for mode in MODES + ("fit_selection_draws",):
                vals = [_metrics(d["models"][slug][method], mode,
                                 d["models"][slug]["ordinary"])
                        for d in per_seed]
                table[f"{slug}/{method}/{mode}"] = {
                    k: [float(np.mean([v[k] for v in vals])),
                        float(np.std([v[k] for v in vals], ddof=1))]
                    for k in vals[0]}
    (root / "summary.json").write_text(json.dumps(table, indent=2))

    lines = ["# Fresh-noise vs abducted-noise evaluation (German-Synth, kappa = 1)",
             "",
             f"gamma = {GAMMA}, epsilon = {EPSILON}; seeds {seeds}. Mean ± SD "
             "over seeds. 'fit_selection_draws' re-reads the draws used to "
             "select the plan (the paper's numbers); the other modes use "
             f"{K_EVAL} independent draws. Validity, score, W1, overshoot, "
             "mean D and the abducted-minus-fresh shift are over served "
             "recipients; closure and gain over all recipients.", ""]
    cols = ["valid", "score", "w1_within_eps", "closure", "mean_d_served",
            "overshoot", "gain", "abduct_minus_fresh"]
    pct = {"valid", "w1_within_eps", "closure", "overshoot"}
    for slug, label in MODELS:
        lines += [f"## {label}", "",
                  "| Method | Mode | " + " | ".join(cols) + " |",
                  "|---|---|" + "---:|" * len(cols)]
        for method in METHODS:
            for mode in ("fit_selection_draws",) + MODES:
                row = table[f"{slug}/{method}/{mode}"]
                cells = [(f"{100 * row[c][0]:.1f} ± {100 * row[c][1]:.1f}"
                          if c in pct else
                          f"{row[c][0]:+.3f} ± {row[c][1]:.3f}") for c in cols]
                lines.append(f"| {method} | {mode} | " + " | ".join(cells) + " |")
        lines.append("")
    (root / "summary.md").write_text("\n".join(lines))
    TABLE_PATH.write_text(latex(table))
    print("\n".join(lines))
    print(f"Wrote {root / 'summary.json'}, {root / 'summary.md'}, {TABLE_PATH}")


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

    lines = [r"""% Generated by evaluation/abduction_sensitivity.py; do not edit by hand.
\begin{table*}[t]
\centering
\footnotesize
\caption{Fresh versus abducted noise in the post-action law (German-Synth,
$\kappa=1$, $\gamma=0.5$, $\varepsilon=0.1$). The selected plans are those of
Table~\ref{tab:app_recourse}. Each plan is re-evaluated with $128$ draws,
independent of those used to select it, under the true SCM. Under ``Fresh''
(F), blocks after the earliest acted block are redrawn with fresh noise, as in
$Q_a$; under ``Abducted'' (A), the recipient keeps exogenous noise drawn from
the posterior $p(U\mid x^{(0)},w_0,z)$. Valid is the share of served
recipients whose plan has success probability at least $\gamma$; Shift is
the mean abducted-minus-fresh score of served plans that act on a discrete
mediator (abduction does not change the others). Model A is the closure
under abduction in the fitted mediator model, with ordinal noise coupled by
the inverse CDF in category order. Closure and Gain over all 250 recipients,
with abstentions at their pre-recourse gap; Valid, Overshoot, and Shift over
served recipients. Shift and Gain on the probability scale, other columns in
\%; mean $\pm$ SD over five seeds.}
\label{tab:app_abduction}
\setlength{\tabcolsep}{1.5pt}
\begin{tabular}{ll cc ccc cc c cc}
\hline
& & \multicolumn{2}{c}{Valid} & \multicolumn{3}{c}{Closure} &
\multicolumn{2}{c}{Overshoot} & & \multicolumn{2}{c}{Gain} \\
Clf. & Method & F & A & F & A & Model A & F & A & Shift & F & A \\
\hline"""]
    for slug, label in MODELS:
        for method in METHODS:
            key = f"{slug}/{method}"
            tf, ta = table[f"{key}/true_fresh"], table[f"{key}/true_abduct"]
            fa = table[f"{key}/fit_abduct"]
            gains = ([ms(tf["gain"], 1, 3, lead=False, sign=True),
                      ms(ta["gain"], 1, 3, lead=False, sign=True)]
                     if method == "ours" else ["", ""])
            lines.append(" & ".join([
                label if method == "ordinary" else "",
                "Ord." if method == "ordinary" else "Ours",
                ms(tf["valid"]), ms(ta["valid"]),
                ms(tf["closure"]), ms(ta["closure"]), ms(fa["closure"]),
                ms(tf["overshoot"]), ms(ta["overshoot"]),
                ms(ta["abduct_minus_fresh"], 1, 3, lead=False, sign=True),
                *gains,
            ]) + r" \\")
        lines.append(r"\hline")
    lines += [r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["eval", "summary"], required=True)
    p.add_argument("--root", type=Path, default=Path("outputs/abduction"))
    p.add_argument("--seeds", type=int, nargs="+",
                   default=[42, 43, 44, 45, 46])
    p.add_argument("--workers", type=int, default=5)
    args = p.parse_args()
    if args.stage == "summary":
        summarize(args.root, args.seeds)
        return
    jobs = [(str(args.root), s) for s in args.seeds]
    with ProcessPoolExecutor(args.workers,
                             mp_context=mp.get_context("spawn")) as pool:
        for message in pool.map(eval_seed, jobs):
            print(message, flush=True)


if __name__ == "__main__":
    main()
