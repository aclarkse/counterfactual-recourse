"""Shared estimands and mixed-type geometry for recourse evaluation."""

import numpy as np
import torch
from scipy.stats import wasserstein_distance

from data.build_tensors import stack_sfm_features


def select_recourse_indices(data, pipe, scaler, disadvantaged_value=0,
                            n_max=None, rng_seed=0):
    """Select only disadvantaged-group true negatives for this classifier."""
    feat = stack_sfm_features(data["X_va"], data["Z_va"], data["Wd_va"],
                              data["Wc_va"], scaler)
    pred = pipe.predict(feat).astype(int)
    y = data["Y_va"].numpy().astype(int)
    x = data["X_va"][:, 0].numpy().astype(int)
    idx = np.flatnonzero((x == disadvantaged_value) & (y == 0) & (pred == 0))
    if n_max is not None and len(idx) > n_max:
        rng = np.random.default_rng(rng_seed)
        idx = np.sort(rng.choice(idx, int(n_max), replace=False))
    return idx


def make_outcome_fn(pipe, scaler):
    def outcome(x, wd, wc, z):
        feat = stack_sfm_features(x, z, wd, wc, scaler)
        return torch.as_tensor(pipe.predict_proba(feat)[:, 1].astype(np.float32))
    return outcome


def estimate_reference_terms(pipe, scaler, Z, sample_fn, K=200,
                             disadvantaged_value=0, advantaged_value=1,
                             batch_size=50):
    """Direct plug-in terms on the recourse-recipient Z population.

    Returns per-recipient E[f(x_dis, W_xadv, z)] and
    E[f(x_dis, W_xdis, z)], plus the x_adv mediator draws used to evaluate the
    mixed-type conditional transport distance. No importance weights are used.
    """
    outcome = make_outcome_fn(pipe, scaler)
    ref_adv, natural_dis = [], []
    wd_reference, wc_reference = [], []
    for start in range(0, len(Z), batch_size):
        z = Z[start:start + batch_size]
        b = len(z)
        xa = torch.full((b, 1), float(advantaged_value))
        xd = torch.full((b, 1), float(disadvantaged_value))
        with torch.no_grad():
            wa = sample_fn(xa, z, K=K)
            wd = sample_fn(xd, z, K=K)
            xd_rep = torch.full((b * K, 1), float(disadvantaged_value))
            fa = outcome(xd_rep, wa["w_disc"], wa["w_cont"],
                         wa["z_rep"]).view(b, K).mean(1)
            fd = outcome(xd_rep, wd["w_disc"], wd["w_cont"],
                         wd["z_rep"]).view(b, K).mean(1)
        ref_adv.extend(fa.numpy().tolist())
        natural_dis.extend(fd.numpy().tolist())
        wd_reference.append(wa["w_disc"].view(b, K, -1).numpy())
        wc_std = wa["w_cont"].numpy()
        wc_nat = (scaler.inverse_transform(wc_std)
                  if scaler is not None and wc_std.shape[1] else wc_std)
        wc_reference.append(wc_nat.reshape(b, K, -1))
    return {
        "reference": np.asarray(ref_adv),
        "natural_disadvantaged": np.asarray(natural_dis),
        "pre_recourse_mediated_prediction_disparity": (
            np.asarray(ref_adv) - np.asarray(natural_dis)
        ),
        "wd_reference": np.concatenate(wd_reference, axis=0),
        "wc_reference": np.concatenate(wc_reference, axis=0),
    }


def add_candidate_geometry(candidates, factual_wd, factual_wc,
                           reference_wd, reference_wc,
                           disc_names, cont_names, mediator_specs):
    """Attach invariance and conditional mixed-metric W1 to candidates.

    The ground metric averages categorical Hamming distance and normalized
    continuous L1 distance. Since a chosen action is a point mass conditional
    on z, its W1 distance to the sampled natural reference is simply expected
    ground distance. This avoids applying a Euclidean metric to category codes.
    """
    n_dims = len(disc_names) + len(cont_names)
    if n_dims == 0:
        n_dims = 1
    disc_distance = []
    for j, name in enumerate(disc_names):
        values = {int(round(float(factual_wd[j]) + c[f"delta_{name}"]))
                  for c in candidates}
        disc_distance.append({
            value: float(np.mean(reference_wd[:, j] != value))
            for value in values
        })
    cont_distance = []
    for j, name in enumerate(cont_names):
        values = {float(factual_wc[j]) + c[f"delta_{name}"]
                  for c in candidates}
        clip = mediator_specs[name].get("clip", [None, None])
        if clip[0] is not None and clip[1] is not None:
            scale = max(float(clip[1]) - float(clip[0]), 1e-12)
        else:
            scale = max(float(np.ptp(reference_wc[:, j])), 1.0)
        cont_distance.append({
            value: float(np.mean(np.abs(reference_wc[:, j] - value))) / scale
            for value in values
        })
    for candidate in candidates:
        distance = 0.0
        for j, name in enumerate(disc_names):
            value = int(round(float(factual_wd[j]) + candidate[f"delta_{name}"]))
            distance += disc_distance[j][value]
        for j, name in enumerate(cont_names):
            value = float(factual_wc[j]) + candidate[f"delta_{name}"]
            distance += cont_distance[j][value]
        candidate["direct_effect"] = abs(candidate["p_xcf"] - candidate["p_xi"])
        candidate["transport"] = distance / n_dims


def add_sampled_candidate_geometry(candidates, sampled_wd, sampled_wc,
                                   reference_wd, reference_wc,
                                   disc_names, cont_names, mediator_specs):
    """Attach transport diagnostics to stochastic intervention plans.

    For each mediator, this is the exact empirical one-dimensional Wasserstein
    distance: total variation under categorical Hamming cost, or normalized
    W1 under continuous absolute-distance cost.  We also compute the expected
    mixed ground cost under the independent empirical coupling.  That cost is
    an upper bound on the joint W1 (rather than being mislabelled as its
    optimum), and is the conservative quantity stored in ``transport`` for the
    rho objective.  Both quantities reduce to the point-mass W1 used by
    ``add_candidate_geometry`` for deterministic plans.
    """
    sampled_wd = np.asarray(sampled_wd)
    sampled_wc = np.asarray(sampled_wc)
    reference_wd = np.asarray(reference_wd)
    reference_wc = np.asarray(reference_wc)
    n_candidates = len(candidates)
    if sampled_wd.shape[0] != n_candidates or sampled_wc.shape[0] != n_candidates:
        raise ValueError("Sample arrays must have one leading row per candidate.")
    n_dims = max(len(disc_names) + len(cont_names), 1)
    marginal_distances = np.zeros(n_candidates, dtype=float)
    coupling_upper = np.zeros(n_candidates, dtype=float)

    for j, _name in enumerate(disc_names):
        max_code = int(max(sampled_wd[:, :, j].max(initial=0),
                           reference_wd[:, j].max(initial=0)))
        ref_prob = np.bincount(
            reference_wd[:, j].astype(int), minlength=max_code + 1
        ).astype(float)
        ref_prob /= max(ref_prob.sum(), 1.0)
        for i in range(n_candidates):
            plan_prob = np.bincount(
                sampled_wd[i, :, j].astype(int), minlength=max_code + 1
            ).astype(float)
            plan_prob /= max(plan_prob.sum(), 1.0)
            marginal_distances[i] += 0.5 * np.abs(plan_prob - ref_prob).sum()
            coupling_upper[i] += 1.0 - float(np.dot(plan_prob, ref_prob))

    for j, name in enumerate(cont_names):
        clip = mediator_specs[name].get("clip", [None, None])
        if clip[0] is not None and clip[1] is not None:
            scale = max(float(clip[1]) - float(clip[0]), 1e-12)
        else:
            scale = max(float(np.ptp(reference_wc[:, j])), 1.0)
        for i in range(n_candidates):
            marginal_distances[i] += wasserstein_distance(
                sampled_wc[i, :, j], reference_wc[:, j]
            ) / scale
            coupling_upper[i] += np.abs(
                sampled_wc[i, :, j, None] - reference_wc[None, :, j]
            ).mean() / scale

    for i, candidate in enumerate(candidates):
        candidate["transport_marginal_w1"] = float(
            marginal_distances[i] / n_dims
        )
        candidate["transport_upper_bound"] = float(coupling_upper[i] / n_dims)
        candidate["transport"] = candidate["transport_upper_bound"]
        candidate["transport_estimand"] = "independent_coupling_upper_bound"


def select_best(candidates, eta, lambda_invariance=0.0, rho_anchor=0.0):
    """Minimize cost + eta*shortfall + lambda*direct effect + rho*transport."""
    return min(
        candidates,
        key=lambda c: (
            c["cost"] + eta * c["shortfall"]
            + lambda_invariance * c.get("direct_effect",
                                        abs(c["p_xcf"] - c["p_xi"]))
            + rho_anchor * c.get("transport", 0.0)
        ),
    )


def attach_mediated_disparities(record, reference, natural_disadvantaged,
                                factual_prediction, disadvantaged_value=0):
    """Attach pre/post mediated prediction-disparity plug-in estimates."""
    # p_xi is defined at the recipient's observed X, and recipients have
    # already been restricted to the configured disadvantaged group.  Its
    # meaning therefore does not depend on whether that group is coded 0 or 1.
    del disadvantaged_value
    p_post = record["p_xi"]
    pre = float(reference - natural_disadvantaged)
    pre_factual = float(reference - factual_prediction)
    post = float(reference - p_post)
    record["pre_recourse_mediated_prediction_disparity"] = pre
    record["pre_recourse_factual_mediated_prediction_disparity"] = pre_factual
    record["post_recourse_mediated_prediction_disparity"] = post
    return record
