"""German-Synth with latent X--W confounding and an optional proxy.

Sensitivity benchmark for the identification assumption W_s _||_ X | Z.
A latent background factor U ~ N(0, 1) shifts both group membership and
every ordinal mediator:

    P(SEX = 1 | U)  = sigmoid(delta * U)
    EDUCATION, JOB, SAVINGS_GRP latent means  += delta * U

All other mechanisms are those of ``data.german_synth`` (kappa = 1). U does
not enter the outcome: mediator--outcome confounding does not affect
classifier-scale effects, so it is deliberately left out. With delta > 0,
P(W | X = b, Z) differs from the interventional law P(W_b | Z), so the
fitted model's reference and NIE are biased.

``PROXY = U + sigma * xi`` is always generated; an experiment decides whether
to place it in Z. Conditioning on a precise proxy (small sigma) removes most
of the confounding.

Oracle helpers return the true interventional quantities:

* ``make_oracle_sample_fn``: draws from P(W_b | z) by integrating U over
  p(U | z), which is N(0, 1) without the proxy and the Gaussian posterior
  given PROXY with it. This is the true reference law and the true input to
  the NIE.
* ``sample_true_plan``: the true interventional law Q_a for one recipient,
  with U drawn from p(U | x, z, factual non-descendant mediators) by
  importance resampling, acted mediators clamped, and descendants redrawn
  from the true mechanisms.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.german_synth import (
    AMOUNT_RANGE, DURATION_RANGE, _AMOUNT, _DURATION, _EDU, _EDU_CUTS, _JOB,
    _JOB_CUTS, _RHO_LD, _SAV, _SAV_CUTS, _age_std, _ordered_logit,
    good_credit_probability, sample_age,
)

_CUTS = (_EDU_CUTS, _JOB_CUTS, _SAV_CUTS)


def _sigmoid(v):
    return 1.0 / (1.0 + np.exp(-v))


def _latent_means(index, sex, a, u, delta, edu=None, job=None):
    """Latent mean of ordinal mediator ``index`` (0 edu, 1 job, 2 savings)."""
    if index == 0:
        base = _EDU["sex"] * sex + _EDU["age"] * a
    elif index == 1:
        base = _JOB["sex"] * sex + _JOB["age"] * a + _JOB["edu"] * edu
    else:
        base = _SAV["sex"] * sex + _SAV["age"] * a + _SAV["job"] * job
    return base + delta * u


def _ordinal_prob(value, mean, cuts):
    """P(ordered-logit = value | latent mean)."""
    upper = np.append(cuts, np.inf)[value]
    lower = np.insert(cuts, 0, -np.inf)[value]
    return _sigmoid(upper - mean) - _sigmoid(lower - mean)


def sample_continuous(age, job, sav, rng):
    a = _age_std(age)
    cov = np.array([[1.0, _RHO_LD], [_RHO_LD, 1.0]])
    noise = rng.multivariate_normal(np.zeros(2), cov, size=len(age))
    log_amount = (_AMOUNT["intercept"] + _AMOUNT["age"] * a
                  + _AMOUNT["job"] * job + _AMOUNT["sav"] * sav
                  + _AMOUNT["noise"] * noise[:, 0])
    log_duration = (_DURATION["intercept"] + _DURATION["age"] * a
                    + _DURATION["job"] * job + _DURATION["sav"] * sav
                    + _DURATION["noise"] * noise[:, 1])
    return (np.clip(np.exp(log_amount), *AMOUNT_RANGE),
            np.clip(np.exp(log_duration), *DURATION_RANGE))


def sample_mediators(sex, age, u, rng, delta, clamp=None):
    """Draw all mediators; ``clamp`` maps index (0-4) to fixed value arrays.

    Indices 0-2 are the ordinal mediators, 3-4 amount and duration. A clamped
    coordinate is held fixed and its descendants are drawn from it.
    """
    clamp = clamp or {}
    sex = np.asarray(sex, dtype=float)
    a = _age_std(age)
    values = []
    for index in range(3):
        if index in clamp:
            values.append(np.asarray(clamp[index]).astype(int))
            continue
        mean = _latent_means(index, sex, a, u, delta,
                             *(values + [None, None])[:2])
        values.append(_ordered_logit(mean, _CUTS[index], rng))
    edu, job, sav = values
    amount, duration = sample_continuous(age, job, sav, rng)
    if 3 in clamp:
        amount = np.asarray(clamp[3], dtype=float) * np.ones_like(amount)
    if 4 in clamp:
        duration = np.asarray(clamp[4], dtype=float) * np.ones_like(duration)
    return edu, job, sav, amount, duration


def sample_population(n_samples, seed=0, delta=0.0, proxy_sigma=0.5):
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(n_samples)
    sex = (rng.random(n_samples) < _sigmoid(delta * u)).astype(int)
    age = sample_age(n_samples, rng)
    edu, job, sav, amount, duration = sample_mediators(sex, age, u, rng, delta)
    p_good = good_credit_probability(sex, age, edu, job, sav, amount, duration)
    good = (rng.random(n_samples) < p_good).astype(int)
    # Separate stream: the proxy noise does not change any other variable.
    xi = np.random.default_rng(seed + 10_000).standard_normal(n_samples)
    return pd.DataFrame({
        "SEX": sex, "AGE": age, "PROXY": u + proxy_sigma * xi,
        "EDUCATION": edu, "JOB": job, "SAVINGS_GRP": sav,
        "CREDIT_AMOUNT": amount, "DURATION": duration,
        "GOOD_CREDIT": good, "U": u,
    })


def load_german_synth_confounded(n_samples: int = 10000, data_seed: int = 0,
                                 delta: float = 0.0,
                                 proxy_sigma: float = 0.5) -> pd.DataFrame:
    df = sample_population(n_samples, data_seed, delta, proxy_sigma)
    print(f"Confounding delta={delta:g}, proxy sigma={proxy_sigma:g}")
    print(f"Final rows : {len(df):,}; corr(SEX, U)="
          f"{np.corrcoef(df['SEX'], df['U'])[0, 1]:+.3f}")
    return df.drop(columns="U")


def _u_prior_draws(z, k, rng, proxy_sigma, uses_proxy):
    """Draws from p(U | z) for each row of z; shape [rows, k]."""
    rows = z.shape[0]
    if not uses_proxy:
        return rng.standard_normal((rows, k))
    v = z[:, 1][:, None]
    s2 = proxy_sigma ** 2
    mean, sd = v / (1.0 + s2), np.sqrt(s2 / (1.0 + s2))
    return mean + sd * rng.standard_normal((rows, k))


def make_oracle_sample_fn(scaler, seed=0, delta=0.0, proxy_sigma=0.5,
                          uses_proxy=False):
    """True P(W_b | z) sampler; same interface as the fitted-model sampler.

    ``z`` holds raw AGE (and raw PROXY when ``uses_proxy``).
    """
    import torch

    rng = np.random.default_rng(seed)

    def sample_fn(x, z, K=500):
        x_rep = x.cpu().repeat_interleave(K, dim=0)
        z_rep = z.cpu().repeat_interleave(K, dim=0)
        u = _u_prior_draws(z.cpu().numpy(), K, rng, proxy_sigma,
                           uses_proxy).reshape(-1)
        edu, job, sav, amount, duration = sample_mediators(
            x_rep[:, 0].numpy(), z_rep[:, 0].numpy(), u, rng, delta)
        w_cont_orig = np.column_stack([amount, duration]).astype(np.float32)
        w_cont = (scaler.transform(w_cont_orig).astype(np.float32)
                  if scaler is not None else w_cont_orig)
        return {
            "w_disc": torch.as_tensor(np.column_stack([edu, job, sav]),
                                      dtype=torch.long),
            "w_cont": torch.as_tensor(w_cont),
            "w_cont_orig": torch.as_tensor(w_cont_orig),
            "x_rep": x_rep, "z_rep": z_rep,
        }

    return sample_fn


def sample_true_plan(x, z, factual_disc, factual_cont, target_disc,
                     target_cont, mask_disc, mask_cont, k, rng, delta=0.0,
                     proxy_sigma=0.5, uses_proxy=False, n_importance=4000):
    """True interventional mediator draws for one recipient and one plan.

    Semantics match ``flows.interventions.sample_intervention_batch`` with
    ``preserve_factual``: blocks before the earliest acted block keep their
    factual values, acted coordinates are clamped, later blocks are redrawn,
    and unacted coordinates of the acted block stay factual. The latent U is
    drawn from p(U | x, z, factual mediators before the earliest acted block).
    Returns (disc [k, 3], cont_natural [k, 2]).
    """
    layers = [0, 1, 2, 3, 3]
    acted = [layers[j] for j in range(3) if mask_disc[j]] + \
            [3 for j in range(2) if mask_cont[j]]
    if not acted:
        return (np.repeat(np.asarray(factual_disc)[None], k, 0),
                np.repeat(np.asarray(factual_cont, dtype=float)[None], k, 0))
    first = min(acted)
    age = float(z[0])
    a = _age_std(np.array([age]))[0]

    # Importance resampling of U given x and factual upstream mediators.
    u = _u_prior_draws(np.asarray(z, dtype=float)[None], n_importance, rng,
                       proxy_sigma, uses_proxy)[0]
    p_x = _sigmoid(delta * u)
    log_w = np.log(np.where(x == 1, p_x, 1.0 - p_x) + 1e-300)
    prev = [None, None]
    for j in range(min(first, 3)):
        mean = _latent_means(j, float(x), a, u, delta, *prev)
        log_w += np.log(_ordinal_prob(int(factual_disc[j]), mean, _CUTS[j])
                        + 1e-300)
        prev = ([float(factual_disc[0]), None] if j == 0
                else [float(factual_disc[0]), float(factual_disc[1])])
    w = np.exp(log_w - log_w.max())
    u_draws = rng.choice(u, size=k, p=w / w.sum())

    clamp = {}
    for j in range(3):
        if layers[j] < first or (layers[j] == first and not mask_disc[j]):
            clamp[j] = np.full(k, int(factual_disc[j]))
        if mask_disc[j]:
            clamp[j] = np.full(k, int(target_disc[j]))
    edu, job, sav, amount, duration = sample_mediators(
        np.full(k, float(x)), np.full(k, age), u_draws, rng, delta, clamp)
    cont = np.column_stack([amount, duration]).astype(float)
    if first == 3:  # acted block is the continuous one: unacted stay factual
        cont[:] = np.asarray(factual_cont, dtype=float)
    for j in range(2):
        if mask_cont[j]:
            cont[:, j] = float(target_cont[j])
    return np.column_stack([edu, job, sav]), cont
