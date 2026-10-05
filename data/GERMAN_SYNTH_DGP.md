# German-Synth data-generating process and mediation stress test

The semi-synthetic German Credit benchmark (`data/german_synth.py`) comes in
four versions. They share one structural causal model and differ only in the
mediation scale `kappa`, which multiplies every `SEX -> mediator` coefficient.
Changing `kappa` changes how strongly sex shifts the mediators, and so the
natural indirect effect (NIE). It leaves the female mediator law, the outcome
equation, and therefore the natural direct effect (NDE) unchanged.

The machine-readable record of every version, including realized summary
statistics and oracle effects, is `outputs/german_synth_dgp_versions.json`.
Regenerate it with `python -m data.german_synth`.

## Structural equations

Notation: `X` is sex (1 = male, advantaged), `A` is age, and
`a = (A - 35) / 11`. `OrdLogit(m; c1, c2, c3)` draws a latent
`m + Logistic(0, 1)` and returns the number of cutpoints below it (0..3).

| Variable | Equation |
|---|---|
| `X` (sex) | `Bernoulli(0.5)` |
| `A` (age) | `19 + Gamma(shape 3, scale 6)`, clipped to `[19, 75]` |
| `E` (education, 0..3) | `OrdLogit(1.2*kappa*X + 0.3*a; -1.0, 0.6, 2.2)` |
| `J` (job level, 0..3) | `OrdLogit(1.0*kappa*X + 0.3*a + 1.0*E; -0.5, 1.4, 3.4)` |
| `S` (savings, 0..3) | `OrdLogit(0.9*kappa*X + 0.5*a + 0.7*J; 0.3, 1.8, 3.0)` |
| `(u1, u2)` | bivariate normal, unit variances, correlation 0.8 |
| `L` (credit amount, DM) | `exp(7.7 + 0.1*a + 0.1*J - 0.1*S + 0.7*u1)`, clipped to `[250, 18424]` |
| `D` (duration, months) | `exp(3.35 - 0.05*a - 0.15*J - 0.12*S + 0.4*u2)`, clipped to `[4, 72]` |
| `Y` (good credit) | `Bernoulli(sigmoid(eta))` |

with `l = (log L - 7.75) / 0.75`, `d = (log D - 2.9) / 0.45`, and

```
eta = 0.85 + 0.3*X + 0.3*a + 0.2*(E - 1.5) + 0.6*(J - 1.5) + 0.6*(S - 1.5)
      - 0.7*l - 0.8*d + 0.25*(S - 1.5)*l
```

Credit amount and duration do not depend on `X` directly; they form one
unordered block whose dependence comes only from correlated noise. Each version
draws one population of 10,000 units with generator seed 0. Experimental seeds
42--46 only re-split it into 8,000 training and 2,000 validation units.

## Versions

Oracle effects are on the `P(Y = 1)` scale, from 200,000 Monte Carlo draws of
the true mechanisms (seed 1). The direct effect is the same in every version
because the female mediator law and the outcome equation do not change.

| Version (config) | `kappa` | Sex coefficients (E, J, S) | Oracle NDE | Oracle NIE | Oracle TE | NIE / TE | Good-credit rate |
|---|---:|---|---:|---:|---:|---:|---:|
| `german_synth_m0` | 0.0 | 0, 0, 0 | 0.042 | -0.001 | 0.041 | ~0 | 0.579 |
| `german_synth_m05` | 0.5 | 0.6, 0.5, 0.45 | 0.042 | 0.098 | 0.136 | 0.72 | 0.627 |
| `german_synth` | 1.0 | 1.2, 1.0, 0.9 | 0.042 | 0.188 | 0.221 | 0.85 | 0.668 |
| `german_synth_m2` | 2.0 | 2.4, 2.0, 1.8 | 0.042 | 0.313 | 0.334 | 0.94 | 0.728 |

Realized mean mediator level by sex (female / male) in the 10,000-unit pool:

| Version | Education | Job | Savings |
|---|---|---|---|
| `german_synth_m0` | 1.22 / 1.23 | 1.41 / 1.44 | 1.15 / 1.17 |
| `german_synth_m05` | 1.22 / 1.54 | 1.41 / 1.80 | 1.15 / 1.54 |
| `german_synth` | 1.22 / 1.85 | 1.41 / 2.14 | 1.15 / 1.91 |
| `german_synth_m2` | 1.22 / 2.40 | 1.41 / 2.64 | 1.15 / 2.48 |

`german_synth` (`kappa = 1`) is the benchmark used in the main results; the
others are stress tests. The `kappa = 0` version is a null case: sex has no
effect on the mediators, so any recourse that the parity constraint asks for
beyond validity is driven by within-group variation, not by a mediated
disparity.

## Relation to Karimi et al. (2020)

The SCM adapts the semi-synthetic German Credit model of Karimi et al.,
"Algorithmic recourse under imperfect causal knowledge" (NeurIPS 2020), to
type-homogeneous mediator blocks. Their loan amount -> duration edge would
sit inside one unordered block, so it is replaced by correlated noise. Their
continuous income is replaced by the ordinal job level. All sex effects on
mediators point the same way, and the direct sex term is small.
