# Abstract-facing multi-seed results

## Protocol

Seeds 42--46 each repeat the train/validation split, mediator-flow fit,
classifier fit, mediation sampling, recipient selection, and recourse
experiment. Reported uncertainty below is the sample standard deviation across
the five complete refits, not the within-fit recipient bootstrap interval.

The comparison fixes `eta=10` and `rho=10`; the full method uses the explicitly
labelled `|NDE/NIE|` heuristic for `lambda`. This ratio is not a calibrated or
Empirical-Bayes weight.

## Result safe to state

On Law School, the full method consistently reduces the magnitude of the
post-recourse mediated prediction disparity relative to ordinary actionable
recourse:

| Classifier | Ordinary post disparity | Full post disparity | Closure gain vs ordinary |
|---|---:|---:|---:|
| Logistic regression | 0.411 ± 0.011 | 0.373 ± 0.011 | 7.68 ± 0.78 pp |
| MLP | 0.388 ± 0.030 | 0.353 ± 0.020 | 7.94 ± 1.89 pp |
| Random forest | 0.379 ± 0.013 | 0.323 ± 0.019 | 13.10 ± 3.92 pp |

This is a result about post-recourse mediated prediction disparity. It must not
be described as “closing the NIE.” The corresponding pure NIE estimates are
0.214 ± 0.010, 0.213 ± 0.017, and 0.193 ± 0.009 on the fixed-classifier
prediction scale.

Suggested abstract sentence:

> Across five complete data-split and model refits on Law School,
> mediation-aware recourse reduced post-recourse mediated prediction disparity
> from 0.411 ± 0.011 to 0.373 ± 0.011 for logistic regression and from
> 0.388 ± 0.030 to 0.353 ± 0.020 for an MLP, improving disparity closure over
> ordinary actionable recourse by 7.7 ± 0.8 and 7.9 ± 1.9 percentage points.
> A random-forest robustness check reduced disparity from 0.379 ± 0.013 to
> 0.323 ± 0.019, a closure gain of 13.1 ± 3.9 percentage points.

## ACS does not support the same claim

At the same fixed setting, ACS results overshoot the natural advantaged-group
reference:

| Classifier | Ordinary post disparity | Full post disparity | Ordinary closure | Full closure | Closure gain vs ordinary |
|---|---:|---:|---:|---:|---:|
| Logistic regression | -0.236 ± 0.025 | -0.212 ± 0.021 | -27.94 ± 24.36% | -15.36 ± 21.73% | +12.57 ± 5.14 pp |
| MLP | -0.239 ± 0.021 | -0.268 ± 0.039 | -36.50 ± 22.74% | -52.54 ± 24.37% | -16.05 ± 18.92 pp |

Thus the logistic version improves on ordinary recourse but does not achieve
positive absolute closure, while the full MLP version is worse. The
transport-anchor-only ablation is better than the full method on ACS
(`D_post=-0.204 ± 0.019` logistic and `-0.224 ± 0.015` MLP), showing that the
effect-ratio-weighted direct-invariance term is not helping there.

Two diagnostics matter before treating ACS as confirmatory evidence:

1. Recipient validity and the disparity estimand conflict at this operating
   point. Recourse pushes true negatives above the soft classifier threshold,
   while their conditional natural advantaged reference can lie below it;
   valid recommendations can therefore cross the reference and increase the
   absolute disparity.
2. `WKHP` has only 60 integer values and 46.9% of observations equal 40 hours,
   yet it is fit as a continuous density. The large between-seed flow-NLL
   variation and the seed-45 mediation shift are consistent with a spline
   concentrating on these atoms. A submission-quality ACS rerun should either
   use principled dequantization or model work hours as ordinal/discrete, and
   should choose `eta`, `lambda`, and `rho` using a declared validation rule.

Until those points are resolved, use the Law School sentence in the abstract
and present ACS as a sensitivity/limitation result rather than claiming
cross-dataset improvement.
