This repository accompanies the paper, From ``What if" to ``How to": Mediation-Aware Causal Recourse via the Standard Fairness Model, and implements a framework for generating algorithmic recourse recommendations that are both causally grounded and equitable across sensitive subgroups, given a fixed predictive model.

## Overview
Standard algorithmic recourse asks how an individual can change their features to flip an unfavorable decision. When the predictor exhibits group-level disparities, however, recourse generated without reference to causal structure can entrench those disparities, recommending changes along pathways that are immutable in practice, or ignoring the mechanism by which the disparity arises in the first place.

We address this by embedding recourse in the Standard Fairness Model (SFM) of Plečko and Bareinboim, which partitions covariates into a sensitive attribute $X$, confounders $Z$, and mediators $W$.


## Pipeline
The implementation is organized as a five-stage pipeline:

1. SFM projection: project each dataset onto the $(X, Z, W, Y)$ template
2. Density estimation: train a mixed-type conditional flow for $\Pr(W \mid X, Z)$, factorized into $\Pr(W_d \mid X, Z) \cdot \Pr(W_c \mid W_d, X, Z)$, with a categorical head for the discrete mediators and a MAF for the continuous component.
3. Mediation decomposition: estimate NDE and NIE; the NIE estimator uses self-normalized importance sampling over exact flow densities.
4. Recourse generation: for each candidate, draw $K$ samples from the learned conditional and select by minimizing cost plus a local fairness penalty under a soft prediction-threshold constraint.
5. Evaluation: report decomposition diagnostics, recourse quality (cost, validity, success rate), and post-recourse fairness improvement.

## Datasets
The repository illustrate the use of this pipeline on two datasets:

- Law School: \texttt{race} is selected as the sensitive attribute, \texttt{LSAT} and undergraduate \texttt{GPA} as mediators, and bar passage (\texttt{bar}) as outcome.
- Folktables ACSIncome (California, 2018): \texttt{sex} is selected as the sensitive attribute, education group (\texttt{SCHL_GRP}) and weekly hours (\texttt{WKHP}) as mediators.