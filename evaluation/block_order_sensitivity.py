"""Sensitivity of recourse to the order of the mediator blocks.

For each dataset, seed, and alternative block order, the mediator model is
refitted with the new order on the seed's existing data split; the seed's
classifiers are reused unchanged, so only the mediator model differs from the
main results. Recourse is then rerun with ``evaluation.ablate_recourse``.

Mechanics
---------
The discrete mediator columns must follow the block order (flows.schema), so
each variant stores a column-permuted copy of the seed's tensors and wraps each
saved classifier in a pipeline whose first step undoes the permutation. The
classifier, data split, and recipients are therefore identical to the main
results. The ``canonical`` variant reruns the main configuration on the
original artifacts and reproduces outputs/multiseed exactly.

By the chain rule every order represents the same joint law, so the reference
and the NIE should not move (up to refitting noise); only the intervention law
Q_a depends on the order. On German-Synth the true SCM is known, so each
selected plan is also evaluated under the true interventional law (true order
EDUCATION -> JOB -> SAVINGS_GRP -> {CREDIT_AMOUNT, DURATION}).

Usage
-----
  python -m evaluation.block_order_sensitivity --stage fit --datasets german_synth acs adult
  python -m evaluation.block_order_sensitivity --stage truth
  python -m evaluation.block_order_sensitivity --stage summary
  # internal: --stage recourse --dataset D --variant V --seed S
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path("outputs/block_order")
MULTISEED = Path("outputs/multiseed")
SEEDS = [42, 43, 44, 45, 46]
MODELS = [("Logistic Reg.", "logreg", "logistic_reg"),
          ("MLP (64--32)", "mlp", "mlp_6432"),
          ("Random Forest", "random_forest", "random_forest")]
GAMMA, EPSILON = 0.5, 0.1
K_TRUE = 64

# Main-result orders (must match conf/dataset/*.yaml).
CANONICAL = {
    "german_synth": [["EDUCATION"], ["JOB"], ["SAVINGS_GRP"],
                     ["CREDIT_AMOUNT", "DURATION"]],
    "acs": [["SCHL_GRP"], ["OCCP_GRP"], ["WKHP", "WKWN"]],
    "adult": [["EDUCATION_GRP"], ["OCCUPATION_GRP"], ["HOURS_GRP"]],
}
# Alternative orders. The continuous block must stay last (flows.schema).
VARIANTS = {
    "german_synth": {
        "swap_job_savings": [["EDUCATION"], ["SAVINGS_GRP"], ["JOB"],
                             ["CREDIT_AMOUNT", "DURATION"]],
        "reversed": [["SAVINGS_GRP"], ["JOB"], ["EDUCATION"],
                     ["CREDIT_AMOUNT", "DURATION"]],
        "joint": [["EDUCATION", "JOB", "SAVINGS_GRP"],
                  ["CREDIT_AMOUNT", "DURATION"]],
    },
    "acs": {
        "swap_edu_occ": [["OCCP_GRP"], ["SCHL_GRP"], ["WKHP", "WKWN"]],
        "joint": [["SCHL_GRP", "OCCP_GRP"], ["WKHP", "WKWN"]],
    },
    "adult": {
        "swap_edu_occ": [["OCCUPATION_GRP"], ["EDUCATION_GRP"],
                         ["HOURS_GRP"]],
        "reversed": [["HOURS_GRP"], ["OCCUPATION_GRP"], ["EDUCATION_GRP"]],
        "joint": [["EDUCATION_GRP", "OCCUPATION_GRP", "HOURS_GRP"]],
    },
}


def _layers(dataset, variant):
    return CANONICAL[dataset] if variant == "canonical" else \
        VARIANTS[dataset][variant]


def _paths(dataset, variant, seed):
    src = MULTISEED / dataset / f"seed_{seed}"
    base = ROOT / dataset / variant / f"seed_{seed}"
    if variant == "canonical":
        data = {"tensors": src / "data" / "tensors.pt",
                "flows": src / "flows" / "flow_models.pt",
                "outcome": src / "outcome", "gaps": src / "gaps"}
    else:
        data = {"tensors": base / "data" / "tensors.pt",
                "flows": base / "flows" / "flow_models.pt",
                "outcome": base / "outcome", "gaps": base / "gaps"}
    return {"src": src, "base": base, **data,
            "recourse": base / "recourse", "logs": base / "logs",
            "records": base / "records.json", "truth": base / "truth.json"}


def _hydra_list(layers):
    return "[" + ",".join("[" + ",".join(b) + "]" for b in layers) + "]"


def _overrides(dataset, variant, seed, p):
    layers = _layers(dataset, variant)
    canonical_disc = [n for b in CANONICAL[dataset] for n in b
                      if n in _disc_names(dataset)]
    disc = [n for b in layers for n in b if n in canonical_disc]
    out = [f"dataset={dataset}", f"seed={seed}",
           f"dataset.paths.tensors={p['tensors']}",
           f"dataset.paths.flows={p['flows']}",
           f"dataset.paths.outcome_dir={p['outcome']}",
           f"dataset.paths.gaps_dir={p['gaps']}",
           f"dataset.paths.recourse_dir={p['recourse']}",
           f"dataset.paths.figures_dir={p['base'] / 'figures'}"]
    if variant != "canonical":
        out += [f"dataset.sfm.mediators_disc=[{','.join(disc)}]",
                f"dataset.sfm.mediator_layers={_hydra_list(layers)}",
                "dataset.flow.reuse_tensors=true"]
        if any(len([n for n in b if n in disc]) > 1 for b in layers):
            # "+" adds the key when the dataset config does not define it.
            key = "dataset.flow.disc_within_block_autoregressive=true"
            from omegaconf import OmegaConf
            flow = OmegaConf.load(f"conf/dataset/{dataset}.yaml").flow
            out.append(key if "disc_within_block_autoregressive" in flow
                       else "+" + key)
    return out


def _disc_names(dataset):
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(f"conf/dataset/{dataset}.yaml")
    return list(cfg.sfm.mediators_disc)


def _permutation(dataset, variant):
    """perm[k] = canonical column index of the variant's k-th disc column."""
    canonical = _disc_names(dataset)
    layers = _layers(dataset, variant)
    variant_disc = [n for b in layers for n in b if n in canonical]
    return [canonical.index(n) for n in variant_disc], variant_disc


def prepare(dataset, variant, seed):
    """Write permuted tensors and wrapped classifiers for one variant."""
    import joblib
    import torch
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer

    p = _paths(dataset, variant, seed)
    perm, variant_disc = _permutation(dataset, variant)
    for key in ("tensors", "flows"):
        p[key].parent.mkdir(parents=True, exist_ok=True)
    p["outcome"].mkdir(parents=True, exist_ok=True)
    p["gaps"].mkdir(parents=True, exist_ok=True)

    data = torch.load(p["src"] / "data" / "tensors.pt", map_location="cpu",
                      weights_only=False)
    for split in ("tr", "va"):
        data[f"Wd_{split}"] = data[f"Wd_{split}"][:, perm].clone()
    data["vocab"] = {name: data["vocab"][name] for name in variant_disc}
    cfg = dict(data["cfg"])
    cfg["mediators_disc"] = variant_disc
    cfg["mediator_layers"] = [list(b) for b in _layers(dataset, variant)]
    data["cfg"] = cfg
    torch.save(data, p["tensors"])

    # Feature layout is [X | Z | Wd | Wc]; undo the Wd permutation in front of
    # the unchanged classifier.
    n_xz = data["X_va"].shape[1] + data["Z_va"].shape[1]
    n_wd = len(perm)
    n_wc = data["Wc_va"].shape[1]
    inverse = np.argsort(perm)
    columns = (list(range(n_xz)) + [n_xz + int(i) for i in inverse]
               + list(range(n_xz + n_wd, n_xz + n_wd + n_wc)))
    for _, slug, _ in MODELS:
        pipe = joblib.load(p["src"] / "outcome" / f"{slug}.joblib")
        wrapped = Pipeline([
            ("unpermute_mediators", FunctionTransformer(
                np.take, kw_args={"indices": columns, "axis": 1})),
            ("model", pipe)])
        joblib.dump(wrapped, p["outcome"] / f"{slug}.joblib")
    shutil.copy(p["src"] / "outcome" / "metrics.json",
                p["outcome"] / "metrics.json")


def fit_one(dataset, variant, seed):
    p = _paths(dataset, variant, seed)
    p["logs"].mkdir(parents=True, exist_ok=True)
    ov = _overrides(dataset, variant, seed, p)
    stages = []
    if variant != "canonical":
        prepare(dataset, variant, seed)
        stages += [("flow", ["-m", "flows.train_flow", *ov]),
                   ("gap", ["-m", "evaluation.estimate_gap", *ov])]
    stages.append(("recourse", ["-m", "evaluation.block_order_sensitivity",
                                "--stage", "recourse", "--dataset", dataset,
                                "--variant", variant, "--seed", str(seed)]))
    for stage, cmd in stages:
        with (p["logs"] / f"{stage}.log").open("w") as log:
            subprocess.run([sys.executable, *cmd], stdout=log,
                           stderr=subprocess.STDOUT, check=True)
    print(f"done {dataset} {variant} seed {seed}", flush=True)


def recourse_one(dataset, variant, seed):
    """Run ablate_recourse and keep per-recipient records of the main rows."""
    from hydra import compose, initialize_config_dir

    import evaluation.ablate_recourse as ar

    p = _paths(dataset, variant, seed)
    store = {"idx": {}, "models": {}}
    current = {}

    orig_select = ar.select_recourse_indices

    def select_recourse_indices(*args, **kwargs):
        idx = orig_select(*args, **kwargs)
        current["idx"] = [int(i) for i in idx]
        return idx
    ar.select_recourse_indices = select_recourse_indices

    captured = []
    orig_summarize = ar.summarize

    def summarize(records, *args, **kwargs):
        captured.append(records)
        return orig_summarize(records, *args, **kwargs)
    ar.summarize = summarize

    orig_run_grid = ar.run_grid

    def run_grid(*args, **kwargs):
        captured.clear()
        rows = orig_run_grid(*args, **kwargs)
        if args[3]:  # main grid call (not a gamma-sensitivity rerun)
            keep = {}
            for row, recs in zip(rows, captured):
                if row["method"] in ("ordinary_actionable_recourse",
                                     "distribution_constrained_recourse"):
                    keep[row["method"]] = [{
                        "cost": float(r["cost"]),
                        "served": bool(r.get("constraint_feasible", False)),
                        "success_prob": float(r.get("success_probability_xi",
                                                    np.nan)),
                        "p_xi": float(r["p_xi"]),
                        "pre": float(r["pre_recourse_factual_mediated_"
                                       "prediction_disparity"]),
                        "post": float(r["post_recourse_mediated_prediction_"
                                        "disparity"]),
                        "deltas": {k[len("delta_"):]: float(v)
                                   for k, v in r.items()
                                   if k.startswith("delta_")},
                    } for r in recs]
            current["records"] = keep
        return rows
    ar.run_grid = run_grid

    orig_dump = ar.json.dump

    def dump(obj, f, *args, **kwargs):
        if isinstance(obj, dict) and "model" in obj and "rows" in obj:
            store["models"][obj["model"]] = {"idx": current["idx"],
                                             **current["records"]}
        return orig_dump(obj, f, *args, **kwargs)
    ar.json.dump = dump

    with initialize_config_dir(config_dir=str(Path("conf").resolve()),
                               version_base="1.1"):
        cfg = compose(config_name="config",
                      overrides=_overrides(dataset, variant, seed, p))
    ar.run(cfg)
    ar.json.dump = orig_dump
    p["records"].write_text(json.dumps(store))


def truth_one(dataset, variant, seed):
    """German-Synth: evaluate selected plans under the true SCM."""
    import joblib
    import torch
    from omegaconf import OmegaConf

    from data.build_tensors import stack_sfm_features
    from data.german_synth import make_oracle_sample_fn
    from data.german_synth_confounded import sample_true_plan
    from evaluation.recourse_metrics import estimate_reference_terms

    p = _paths(dataset, variant, seed)
    rec = OmegaConf.load(f"conf/dataset/{dataset}.yaml").recourse
    data = torch.load(p["tensors"], map_location="cpu", weights_only=False)
    scaler = data["scaler"]
    perm, variant_disc = _permutation(dataset, variant)
    canonical = _disc_names(dataset)
    cont = list(data["cfg"]["mediators_cont"])
    records = json.loads(p["records"].read_text())
    base_oracle = make_oracle_sample_fn(scaler, seed=seed + 1)

    def oracle_fn(x, z, K=500):  # canonical -> variant column order
        out = base_oracle(x, z, K)
        out["w_disc"] = out["w_disc"][:, perm]
        return out

    def score(pipe, x, z, wd_variant, wc_natural):
        wc_std = scaler.transform(wc_natural).astype(np.float32)
        feat = stack_sfm_features(
            torch.as_tensor(x, dtype=torch.float32),
            torch.as_tensor(z, dtype=torch.float32),
            torch.as_tensor(wd_variant, dtype=torch.long),
            torch.as_tensor(wc_std), scaler)
        return pipe.predict_proba(feat)[:, 1]

    rng = np.random.default_rng(seed + 30_000)
    inverse = np.argsort(perm)  # variant column of each canonical mediator
    out = {}
    for name, slug, _ in MODELS:
        pipe = joblib.load(p["outcome"] / f"{slug}.joblib")
        model = records["models"][name]
        idx = model["idx"]
        ref_true = estimate_reference_terms(
            pipe, scaler, data["Z_va"][idx], oracle_fn, K=int(rec.reference_K),
            disadvantaged_value=int(rec.disadvantaged_value),
            advantaged_value=int(rec.advantaged_value))["reference"]
        res = {}
        for method in ("ordinary_actionable_recourse",
                       "distribution_constrained_recourse"):
            post_true = []
            for i, j in enumerate(idx):
                r = model[method][i]
                x, z = data["X_va"][j:j + 1], data["Z_va"][j:j + 1]
                wd_var = data["Wd_va"][j].numpy()
                wd_can = wd_var[inverse]
                wc_nat = scaler.inverse_transform(
                    data["Wc_va"][j:j + 1].numpy())[0]
                t_disc = np.array([int(round(wd_can[k] + r["deltas"][n]))
                                   for k, n in enumerate(canonical)])
                t_cont = np.array([wc_nat[k] + r["deltas"][n]
                                   for k, n in enumerate(cont)])
                d_can, c_nat = sample_true_plan(
                    int(x[0, 0]), z[0].numpy(), wd_can, wc_nat, t_disc,
                    t_cont, t_disc != wd_can,
                    ~np.isclose(t_cont, wc_nat, atol=1e-7), K_TRUE, rng)
                p_true = score(pipe, np.repeat(x.numpy(), K_TRUE, 0),
                               np.repeat(z.numpy(), K_TRUE, 0),
                               d_can[:, perm], c_nat).mean()
                post_true.append(float(ref_true[i] - p_true))
            # pre is reference minus the factual score; the factual score does
            # not depend on the mediator model, so shift by the reference bias.
            pre_fit = np.array([r["pre"] for r in model[method]])
            post_fit = np.array([r["post"] for r in model[method]])
            ref_fit = post_fit + np.array([r["p_xi"] for r in model[method]])
            pre_true = pre_fit - ref_fit + np.asarray(ref_true)
            res[method] = {"pre_true": pre_true.tolist(),
                           "post_true": post_true,
                           "reference_bias": float(np.mean(ref_fit - ref_true))}
        out[name] = res
    p["truth"].write_text(json.dumps(out))
    print(f"truth {dataset} {variant} seed {seed}", flush=True)


# ── Summary ─────────────────────────────────────────────────────────────────

def _closure(pre, post, mask=None):
    pre, post = np.abs(np.asarray(pre)), np.abs(np.asarray(post))
    if mask is not None:
        if not np.any(mask):
            return np.nan
        pre, post = pre[mask], post[mask]
    return 1.0 - post.sum() / pre.sum()


def _gap_nie(p, dataset, name):
    gap = next(p["gaps"].glob(f"{dataset}_*_gap.json"))
    data = json.loads(gap.read_text())[name]
    overall = data.get("overall", data)
    nie = overall["nie"]
    return float(nie[0] if isinstance(nie, list) else nie)


def seed_metrics(dataset, variant, seed):
    p = _paths(dataset, variant, seed)
    records = json.loads(p["records"].read_text())
    truth = (json.loads(p["truth"].read_text())
             if p["truth"].exists() else None)
    out = {}
    for name, _, _ in MODELS:
        model = records["models"][name]
        m = {"nie": _gap_nie(p, dataset, name)}
        ordinary = model["ordinary_actionable_recourse"]
        for key, method in (("ord", "ordinary_actionable_recourse"),
                            ("ours", "distribution_constrained_recourse")):
            r = model[method]
            pre = np.array([x["pre"] for x in r])
            post = np.array([x["post"] for x in r])
            served = np.array([x["served"] for x in r])
            cost = np.array([x["cost"] for x in r])
            flip = np.array([x["success_prob"] for x in r])
            m[f"{key}_closure"] = _closure(pre, post)
            m[f"{key}_closure_served"] = _closure(pre, post, served)
            m[f"{key}_served"] = served.mean()
            m[f"{key}_cost"] = cost.mean()
            m[f"{key}_cost_served"] = (cost[served].mean() if served.any()
                                       else np.nan)
            m[f"{key}_flip_served"] = (flip[served].mean() if served.any()
                                       else np.nan)
            if truth is not None:
                t = truth[name][method]
                m[f"{key}_closure_true"] = _closure(t["pre_true"],
                                                    t["post_true"])
                m["reference_bias"] = t["reference_bias"]
        m["gain"] = float(np.mean(
            np.abs([x["post"] for x in ordinary])
            - np.abs([x["post"] for x in
                      model["distribution_constrained_recourse"]])))
        if truth is not None:
            m["gain_true"] = float(np.mean(
                np.abs(truth[name]["ordinary_actionable_recourse"]
                       ["post_true"])
                - np.abs(truth[name]["distribution_constrained_recourse"]
                         ["post_true"])))
        out[name] = m
    return out


def summary(datasets):
    names = {"german_synth": "German-Synth", "acs": "ACS Income",
             "adult": "Adult"}
    lines = ["# Mediator block-order sensitivity (gamma = 0.5, "
             "epsilon = 0.1)", "",
             "Mediator model refitted with each block order on the seed's "
             "data split; classifiers, recipients, and references' "
             "conditioning are unchanged. Mean ± SD over seeds of the "
             "three-classifier average; closure, served, and flip in %. "
             "'| S' = among served recipients. 'True' columns evaluate the "
             "selected plans under the true SCM (German-Synth only).", ""]
    table = {}
    for ds in datasets:
        variants = ["canonical"] + list(VARIANTS[ds])
        has_truth = ds == "german_synth"
        head = ("| Order | NIE | Ord. closure | Ours closure | Ours served | "
                "Ours closure \\| S | Ours cost \\| S | Gain |")
        sep = "|---|---:|---:|---:|---:|---:|---:|---:|"
        if has_truth:
            head += " Ord. closure true | Ours closure true | Gain true |"
            sep += "---:|---:|---:|"
        lines += [f"## {names[ds]}", "",
                  "Orders: " + "; ".join(
                      f"{v}: " + " → ".join(
                          "{" + ", ".join(b) + "}" for b in _layers(ds, v))
                      for v in variants), "", head, sep]
        for v in variants:
            seeds = [s for s in SEEDS if _paths(ds, v, s)["records"].exists()]
            if not seeds:
                continue
            per = [seed_metrics(ds, v, s) for s in seeds]
            keys = per[0][MODELS[0][0]].keys()
            agg = {}
            for k in keys:
                vals = np.array([np.nanmean([sm[n][k] for n, _, _ in MODELS])
                                 if not all(np.isnan(sm[n][k])
                                            for n, _, _ in MODELS)
                                 else np.nan for sm in per])
                agg[k] = (float(np.nanmean(vals)) if not np.all(np.isnan(vals))
                          else np.nan,
                          float(np.nanstd(vals, ddof=1))
                          if np.sum(~np.isnan(vals)) > 1 else np.nan)
            agg["n_seeds"] = len(seeds)
            table[f"{ds}|{v}"] = agg

            def f(k, s=100, d=1):
                mu, sd = agg[k]
                if np.isnan(mu):
                    return "--"
                return f"{s * mu:.{d}f} ± {s * sd:.{d}f}"
            row = (f"| {v} | {f('nie', 1, 3)} | {f('ord_closure')} | "
                   f"{f('ours_closure')} | {f('ours_served', 100, 0)} | "
                   f"{f('ours_closure_served')} | "
                   f"{f('ours_cost_served', 1, 2)} | {f('gain', 1, 3)} |")
            if has_truth:
                row += (f" {f('ord_closure_true')} | "
                        f"{f('ours_closure_true')} | {f('gain_true', 1, 3)} |")
            lines.append(row)
        lines.append("")
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "summary.md").write_text("\n".join(lines))
    (ROOT / "summary.json").write_text(json.dumps(table, indent=1))
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["fit", "recourse", "truth", "summary"])
    ap.add_argument("--datasets", nargs="+",
                    default=["german_synth", "acs", "adult"])
    ap.add_argument("--dataset")
    ap.add_argument("--variant")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--variants", nargs="+")
    args = ap.parse_args()
    if args.stage == "recourse":
        recourse_one(args.dataset, args.variant, args.seed)
    elif args.stage == "fit":
        for ds in args.datasets:
            for v in args.variants or ["canonical", *VARIANTS[ds]]:
                for s in args.seeds:
                    if _paths(ds, v, s)["records"].exists():
                        continue
                    fit_one(ds, v, s)
    elif args.stage == "truth":
        for v in args.variants or ["canonical",
                                   *VARIANTS["german_synth"]]:
            for s in args.seeds:
                if _paths("german_synth", v, s)["records"].exists():
                    truth_one("german_synth", v, s)
    else:
        summary(args.datasets)


if __name__ == "__main__":
    main()
