"""Semi-synthetic German Credit SCM with a strong mediated sex disparity.

Adapted from the semi-synthetic German Credit SCM of Karimi et al. (2020),
"Algorithmic recourse under imperfect causal knowledge: a probabilistic
approach" (NeurIPS). Their SCM (sex G, age A, education E, loan amount L,
duration D, income I, savings S) is rearranged to fit this repository's
mediator schema, which requires type-homogeneous blocks, discrete blocks
first, and one final joint continuous block:

    SEX, AGE                                      (roots, independent)
    EDUCATION     <- SEX, AGE                     (block 1, ordinal 0..3)
    JOB           <- SEX, AGE, EDUCATION          (block 2, ordinal 0..3)
    SAVINGS_GRP   <- SEX, AGE, JOB                (block 3, ordinal 0..3)
    [CREDIT_AMOUNT, DURATION]
                  <- AGE, JOB, SAVINGS_GRP        (block 4, joint continuous)
    GOOD_CREDIT   <- SEX, AGE, all mediators

Differences from Karimi et al. that matter for mediation analysis:

* Karimi's L -> D edge would sit inside one unordered block, so it is
  replaced by correlated exogenous noise (corr ``_RHO_LD``). The learned
  flow's "no within-block causal order" assumption therefore holds exactly.
* Karimi's continuous income is replaced by the ordinal JOB skill level
  (as in UCI German Credit), and savings is an ordinal bin.
* The SEX effects on all mediators point the same way (men advantaged), and
  the direct SEX -> Y term is small, so the natural indirect effect dominates
  the total effect. Karimi's SCM has no direct G -> Y term and opposing
  G -> L/D and G -> I paths.
* Ordinal mediators use ordered-logit mechanisms; continuous mediators are
  log-normal and clipped to the observed UCI ranges (amount 250-18,424 DM,
  duration 4-72 months).

Because the SCM is known, ``oracle_mediation_effects`` gives the true pure
NDE/NIE/TE on the P(Y=1) scale. There is no mediator-outcome confounding
beyond AGE, so these are identified by the mediation formula.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


_AGE_CENTER = 35.0
_AGE_SCALE = 11.0
_LOG_AMOUNT_CENTER = 7.75
_LOG_AMOUNT_SCALE = 0.75
_LOG_DURATION_CENTER = 2.9
_LOG_DURATION_SCALE = 0.45
_RHO_LD = 0.8
AMOUNT_RANGE = (250.0, 18424.0)
DURATION_RANGE = (4.0, 72.0)

# Ordered-logit cutpoints for the three ordinal mediators (4 levels each).
_EDU_CUTS = np.array([-1.0, 0.6, 2.2])
_JOB_CUTS = np.array([-0.5, 1.4, 3.4])
_SAV_CUTS = np.array([0.3, 1.8, 3.0])

# Structural coefficients.
_EDU = {"sex": 1.2, "age": 0.3}
_JOB = {"sex": 1.0, "age": 0.3, "edu": 1.0}
_SAV = {"sex": 0.9, "age": 0.5, "job": 0.7}
_AMOUNT = {"intercept": 7.7, "age": 0.1, "job": 0.1, "sav": -0.1,
           "noise": 0.7}
_DURATION = {"intercept": 3.35, "age": -0.05, "job": -0.15, "sav": -0.12,
             "noise": 0.4}
_Y = {"intercept": 0.85, "sex": 0.3, "age": 0.3, "edu": 0.2, "job": 0.6,
      "sav": 0.6, "amount": -0.7, "duration": -0.8, "sav_x_amount": 0.25}


def _ordered_logit(latent_mean, cuts, rng):
    latent = latent_mean + rng.logistic(size=latent_mean.shape)
    return np.searchsorted(cuts, latent).astype(int)


def _age_std(age):
    return (np.asarray(age, dtype=float) - _AGE_CENTER) / _AGE_SCALE


def sample_mediators(sex, age, rng, mediation_scale=1.0):
    """Draw (EDUCATION, JOB, SAVINGS_GRP, CREDIT_AMOUNT, DURATION).

    ``mediation_scale`` multiplies every SEX -> mediator coefficient, so it
    scales the indirect effect while leaving the female mediator law, and
    hence the direct effect, unchanged.
    """
    sex = np.asarray(sex, dtype=float) * float(mediation_scale)
    a = _age_std(age)
    edu = _ordered_logit(_EDU["sex"] * sex + _EDU["age"] * a, _EDU_CUTS, rng)
    job = _ordered_logit(
        _JOB["sex"] * sex + _JOB["age"] * a + _JOB["edu"] * edu,
        _JOB_CUTS, rng,
    )
    sav = _ordered_logit(
        _SAV["sex"] * sex + _SAV["age"] * a + _SAV["job"] * job,
        _SAV_CUTS, rng,
    )
    cov = np.array([[1.0, _RHO_LD], [_RHO_LD, 1.0]])
    u = rng.multivariate_normal(np.zeros(2), cov, size=len(sex))
    log_amount = (
        _AMOUNT["intercept"] + _AMOUNT["age"] * a + _AMOUNT["job"] * job
        + _AMOUNT["sav"] * sav + _AMOUNT["noise"] * u[:, 0]
    )
    log_duration = (
        _DURATION["intercept"] + _DURATION["age"] * a
        + _DURATION["job"] * job + _DURATION["sav"] * sav
        + _DURATION["noise"] * u[:, 1]
    )
    amount = np.clip(np.exp(log_amount), *AMOUNT_RANGE)
    duration = np.clip(np.exp(log_duration), *DURATION_RANGE)
    return edu, job, sav, amount, duration


def good_credit_probability(sex, age, edu, job, sav, amount, duration):
    """Structural P(GOOD_CREDIT=1 | parents)."""
    a = _age_std(age)
    amt = (np.log(amount) - _LOG_AMOUNT_CENTER) / _LOG_AMOUNT_SCALE
    dur = (np.log(duration) - _LOG_DURATION_CENTER) / _LOG_DURATION_SCALE
    logit = (
        _Y["intercept"] + _Y["sex"] * np.asarray(sex, dtype=float)
        + _Y["age"] * a + _Y["edu"] * (edu - 1.5) + _Y["job"] * (job - 1.5)
        + _Y["sav"] * (sav - 1.5) + _Y["amount"] * amt
        + _Y["duration"] * dur + _Y["sav_x_amount"] * (sav - 1.5) * amt
    )
    return 1.0 / (1.0 + np.exp(-logit))


def sample_age(n, rng):
    return np.clip(19.0 + rng.gamma(3.0, 6.0, size=n), 19.0, 75.0)


def sample_german_synthetic(n_samples, seed=0, p_male=0.5, mediation_scale=1.0):
    rng = np.random.default_rng(seed)
    sex = (rng.random(n_samples) < p_male).astype(int)
    age = sample_age(n_samples, rng)
    edu, job, sav, amount, duration = sample_mediators(
        sex, age, rng, mediation_scale)
    p_good = good_credit_probability(sex, age, edu, job, sav, amount, duration)
    good = (rng.random(n_samples) < p_good).astype(int)
    return pd.DataFrame({
        "SEX": sex,
        "AGE": age,
        "EDUCATION": edu,
        "JOB": job,
        "SAVINGS_GRP": sav,
        "CREDIT_AMOUNT": amount,
        "DURATION": duration,
        "GOOD_CREDIT": good,
    })


def load_german_synthetic(
    n_samples: int = 10000,
    data_seed: int = 0,
    p_male: float = 0.5,
    mediation_scale: float = 1.0,
) -> pd.DataFrame:
    """Draw a fixed synthetic population; the experiment seed only splits it."""
    df = sample_german_synthetic(n_samples, seed=data_seed, p_male=p_male,
                                 mediation_scale=mediation_scale)
    effects = oracle_mediation_effects(n_samples=200_000, seed=data_seed + 1,
                                       mediation_scale=mediation_scale)
    print(f"Mediation scale: {mediation_scale:g}")
    print(f"Final rows : {len(df):,}")
    print(f"Good-credit rate: {df['GOOD_CREDIT'].mean():.3f}")
    print(f"Female share: {(df['SEX'] == 0).mean():.3f}")
    print(
        "Oracle P(Y=1) effects: "
        + "  ".join(f"{k}={effects[k]:+.3f}" for k in ("nde", "nie", "te"))
    )
    return df


def oracle_mediation_effects(n_samples=200_000, seed=1, ages=None,
                             mediation_scale=1.0):
    """True pure NDE/NIE/TE of SEX (0 -> 1) on P(GOOD_CREDIT=1).

    ``mu_ab = E_AGE E[P(Y=1 | SEX=a, W) | W ~ W(SEX=b, AGE)]``, with the
    mediator draws for b=0 and b=1 taken independently.
    """
    rng = np.random.default_rng(seed)
    age = sample_age(n_samples, rng) if ages is None else np.asarray(ages)
    n = len(age)
    zeros, ones = np.zeros(n), np.ones(n)
    w0 = sample_mediators(zeros, age, rng, mediation_scale)
    w1 = sample_mediators(ones, age, rng, mediation_scale)
    mu = {
        "mu00": good_credit_probability(zeros, age, *w0).mean(),
        "mu10": good_credit_probability(ones, age, *w0).mean(),
        "mu01": good_credit_probability(zeros, age, *w1).mean(),
        "mu11": good_credit_probability(ones, age, *w1).mean(),
    }
    mu = {k: float(v) for k, v in mu.items()}
    mu["nde"] = mu["mu10"] - mu["mu00"]
    mu["nie"] = mu["mu01"] - mu["mu00"]
    mu["te"] = mu["mu11"] - mu["mu00"]
    mu["interaction"] = mu["mu11"] - mu["mu10"] - mu["mu01"] + mu["mu00"]
    return mu


def make_oracle_sample_fn(scaler, seed=0, mediation_scale=1.0):
    """True-SCM drop-in for ``flows.diagnostics.make_sample_fns``' sampler.

    ``z`` holds raw AGE, as in the cached tensors. Continuous draws are mapped
    into the flow's scaler space so fitted-classifier wrappers can be reused.
    """
    import torch

    rng = np.random.default_rng(seed)

    def sample_fn(x, z, K=500):
        x_rep = x.cpu().repeat_interleave(K, dim=0)
        z_rep = z.cpu().repeat_interleave(K, dim=0)
        edu, job, sav, amount, duration = sample_mediators(
            x_rep[:, 0].numpy(), z_rep[:, 0].numpy(), rng, mediation_scale
        )
        w_cont_orig = np.column_stack([amount, duration]).astype(np.float32)
        w_cont = (scaler.transform(w_cont_orig).astype(np.float32)
                  if scaler is not None else w_cont_orig)
        return {
            "w_disc": torch.as_tensor(np.column_stack([edu, job, sav]),
                                      dtype=torch.long),
            "w_cont": torch.as_tensor(w_cont),
            "w_cont_orig": torch.as_tensor(w_cont_orig),
            "x_rep": x_rep,
            "z_rep": z_rep,
        }

    return sample_fn


def describe_dgp(mediation_scale=1.0, n_samples=10_000, data_seed=0,
                 p_male=0.5, oracle_samples=200_000):
    """Record the full data-generating process of one benchmark version."""
    k = float(mediation_scale)
    df = sample_german_synthetic(n_samples, seed=data_seed, p_male=p_male,
                                 mediation_scale=k)
    effects = oracle_mediation_effects(oracle_samples, seed=data_seed + 1,
                                       mediation_scale=k)
    return {
        "mediation_scale": k,
        "population": {"n_samples": int(n_samples), "data_seed": int(data_seed),
                       "p_male": float(p_male),
                       "age": "19 + Gamma(shape=3, scale=6), clipped to [19, 75]",
                       "age_standardization": [_AGE_CENTER, _AGE_SCALE]},
        "education": {"sex": k * _EDU["sex"], "age": _EDU["age"],
                      "cutpoints": _EDU_CUTS.tolist(), "link": "ordered logit"},
        "job": {"sex": k * _JOB["sex"], "age": _JOB["age"], "edu": _JOB["edu"],
                "cutpoints": _JOB_CUTS.tolist(), "link": "ordered logit"},
        "savings": {"sex": k * _SAV["sex"], "age": _SAV["age"],
                    "job": _SAV["job"], "cutpoints": _SAV_CUTS.tolist(),
                    "link": "ordered logit"},
        "log_credit_amount": dict(_AMOUNT),
        "log_duration": dict(_DURATION),
        "amount_duration_noise_correlation": _RHO_LD,
        "clip_ranges": {"credit_amount": list(AMOUNT_RANGE),
                        "duration": list(DURATION_RANGE)},
        "outcome_logit": dict(_Y),
        "outcome_standardization": {
            "log_amount": [_LOG_AMOUNT_CENTER, _LOG_AMOUNT_SCALE],
            "log_duration": [_LOG_DURATION_CENTER, _LOG_DURATION_SCALE],
            "ordinal_centering": 1.5},
        "realized": {
            "good_credit_rate": float(df["GOOD_CREDIT"].mean()),
            "female_share": float((df["SEX"] == 0).mean()),
            "mediator_means_by_sex": {
                str(sex): {name: float(value) for name, value in row.items()}
                for sex, row in df.groupby("SEX")[
                    ["EDUCATION", "JOB", "SAVINGS_GRP"]].mean().iterrows()},
        },
        "oracle_effects_p_y1": {key: float(effects[key])
                                for key in ("nde", "nie", "te", "interaction")},
    }


SFM_CONFIG_GERMAN_SYNTH = {
    "sensitive": ["SEX"],
    "confounders": ["AGE"],
    "mediators_disc": ["EDUCATION", "JOB", "SAVINGS_GRP"],
    "mediators_cont": ["CREDIT_AMOUNT", "DURATION"],
    "outcome": "GOOD_CREDIT",
}

VOCAB_GERMAN_SYNTH = {"EDUCATION": 4, "JOB": 4, "SAVINGS_GRP": 4}

GROUP_LABELS_GERMAN_SYNTH = {0: "Female", 1: "Male"}


STRESS_TEST_VERSIONS = {
    "german_synth_m0": 0.0,
    "german_synth_m05": 0.5,
    "german_synth": 1.0,
    "german_synth_m2": 2.0,
}


def write_dgp_record(json_path="outputs/german_synth_dgp_versions.json"):
    """Write the data-generating process of every stress-test version."""
    import json
    import os

    record = {name: describe_dgp(scale)
              for name, scale in STRESS_TEST_VERSIONS.items()}
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(record, file, indent=2)
    for name, item in record.items():
        effects = item["oracle_effects_p_y1"]
        print(f"{name:18s} kappa={item['mediation_scale']:<4g} "
              f"NDE={effects['nde']:+.4f} NIE={effects['nie']:+.4f} "
              f"TE={effects['te']:+.4f} "
              f"P(Y=1)={item['realized']['good_credit_rate']:.3f}")
    return record


if __name__ == "__main__":
    write_dgp_record()
