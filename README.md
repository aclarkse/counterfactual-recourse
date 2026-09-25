# Mediation-aware causal recourse

Research code for diagnosing whether classifier-satisfying recourse closes or
widens a disparity carried through actionable mediators. The estimands in this
repository are effects on a fixed classifier `f_hat`, not effects on the
observed outcome `Y`.

## Important revision note

The current source corrects four errors in the original experiment code:

- the old `NIE_post` was `f_hat(1,w',z)-f_hat(0,w',z)`, a pointwise direct
  effect. It is now a separately named direct-gap diagnostic;
- the old “lambda sweep” varied only the shortfall weight `eta`. The revised
  ablation varies `eta`, the invariance weight `lambda`, and the distributional
  anchor weight `rho` independently;
- the old flow checkpoint used conditionally independent discrete heads and did
  not condition the continuous flow on the discrete mediators. New training is
  ordered in explicit causal blocks; and
- the old recourse grid jointly fixed every mediator coordinate. Direct actions
  now clamp only selected coordinates and regenerate strict descendants through
  the learned mechanisms.

Existing checkpoints remain loadable for reproducibility, but submission
results must be regenerated from Stage 1 onward. Existing legacy tables should
not be cited as revised results.

## Setup

```bash
uv sync
```

All configuration is under `dataset.*`; for example,
`dataset.recourse.reference_K=500`.

## Pipeline

### 1. Train the mediator model

The revised mixed model uses configured partially ordered blocks. For ACS:

```text
p(Education,Occupation,Hours | X,Z)
  = p(Education | X,Z)
    p(Occupation | Education,X,Z)
    p(Hours | Education,Occupation,X,Z).
```

For Law School, `[LSAT,GPA]` is one unordered block modeled by one multivariate
conditional flow, `p(LSAT,GPA | X,Z)`. The autoregressive coordinate transform
inside that flow is only a density parameterization; it is not interpreted as
an LSAT-to-GPA or GPA-to-LSAT causal arrow. `sfm.mediator_layers` records these
assumptions and is saved in new checkpoints.

```bash
uv run python -m flows.train_flow
uv run python -m flows.train_flow dataset=bar
# Simpler conditional-Gaussian ablation (use a separate checkpoint):
uv run python -m flows.train_flow dataset.flow.continuous_family=gaussian \
  dataset.paths.flows=outputs/flows/acs/gaussian_models.pt
```

Outputs are written to `outputs/flows/` and `outputs/data/`.

### 2. Train fixed outcome classifiers

The Bar Passage configuration fits logistic regression, an MLP, and a
random-forest robustness model. ACS currently fits logistic regression and an
MLP.

```bash
uv run python -m outcome.train_outcome
uv run python -m outcome.train_outcome dataset=bar
```

### 3. Estimate classifier-scale mediation effects

For `mu_ab(z) = E[f_hat(x=a,W_x=b,z) | Z=z]`, the code reports

```text
pure NDE    = mu_10 - mu_00
pure NIE    = mu_01 - mu_00
total NDE   = mu_11 - mu_01
total NIE   = mu_11 - mu_10
TE          = mu_11 - mu_00
interaction = mu_11 - mu_10 - mu_01 + mu_00.
```

`TE` is sampled directly. The implementation does not assume `TE=NDE+NIE`.
Both mediator distributions are sampled directly; no importance weights are
reused.

```bash
uv run python -m evaluation.estimate_gap
uv run python -m evaluation.estimate_gap dataset.gap.K=1000 dataset.gap.n_inst=1000
```

The JSON output includes NDE, NIE, TE, the interaction residual,
`|NIE/TE|`, and `|NDE/NIE|`. The last quantity is explicitly labelled an
effect-ratio heuristic; it is not called empirical Bayes and is not the default
objective weight.

### 4. Generate recourse

Only disadvantaged-group true negatives receive recourse. Each candidate is a
direct action plan `a`, not a joint setting of every mediator. Acted
coordinates are clamped, strict descendants are drawn ancestrally, and
non-descendants remain factual. The objective is

```text
cost(a)
+ eta    * E_Qa[classifier_shortfall(W)]
+ lambda * E_Qa[|f_hat(1,W,z)-f_hat(0,W,z)|]
+ rho    * U_mix(Qa, P(W | X=advantaged,z)).
```

`U_mix` is expected categorical-Hamming/normalized-L1 ground cost under an
explicit independent empirical coupling. It is a computable upper bound on
the joint mixed-metric Wasserstein distance, not the optimal joint transport
itself. The exact mean marginal W1 is retained on each candidate as an
additional diagnostic. The defaults are in the dataset YAML and can be
overridden explicitly:

```bash
uv run python -m evaluation.compute_recourse \
  dataset.recourse.eta=10 \
  dataset.recourse.lambda_invariance=0 \
  dataset.recourse.rho_anchor=0
```

Here `lambda=0,rho=0` is the standard minimum-cost actionable-recourse
baseline and is explicitly labelled `ordinary_actionable_recourse` in the
ablation outputs. `rho>0` activates the distributional-anchoring variant.

For an intervention on one coordinate of an unordered same-level block, the
default `same_level_semantics=preserve_factual` keeps the other coordinates at
that individual's factual values. The optional `resample_marginal` mode draws
the joint natural block and then overwrites the acted coordinates. Neither mode
asserts causal order within the block.

### 5. Run independent ablations

```bash
uv run python -m evaluation.ablate_recourse
# Backward-compatible command; now routes to the same corrected ablation:
uv run python -m evaluation.sweep_lambda
```

The Cartesian grids `eta_grid`, `lambda_grid`, and `rho_grid` are configured
independently. Outputs are JSON, CSV, and LaTeX tables under
`outputs/recourse/ablation_*`. If available, `|NDE/NIE|` is inserted as a
marked heuristic lambda row, not used as a calibration.

## Post-recourse estimand

For recipient confounders `Z_R`, the reported post-recourse mediated disparity
is

```text
D_post = E_{Z_R}[ E_{W~P(W|X=advantaged,Z)} f_hat(X=disadvantaged,W,Z)
                  - E_{W~Q_a(.|Z)} f_hat(X=disadvantaged,W,Z) ].
```

The advantaged reference is sampled directly and is never sent through
recourse. The code also reports two pre-recourse checks on the same recipients:

- model-based `D_pre`, using direct draws from both natural mediator models;
- factual plug-in `D_pre_factual`, replacing the disadvantaged natural
  expectation with each recipient's observed mediator value.

Thus the output states which population supplies `Z`, whether the reference is
modified (it is not), and whether importance weights are reused (they are not).
Because `D_post` compares a natural distribution with an intervention-induced
one, it is not called a natural indirect effect.

The recipient-level percentage closed is descriptive and uses the recipients'
observed pre-recourse state:

```text
factual closure = 1 - |mean(D_post)| / |mean(D_pre_factual)|.
```

Model-based `D_pre` remains a natural-distribution mediation diagnostic, but is
not used as the closure denominator after selecting recipients by their factual
features and classifier decision.

The conservative mixed-metric coupling bound is reported beside `D_post` at
every ablation point so their association can be assessed directly.

The causal interpretation still requires the stated graph, consistency,
positivity, and no unmeasured mediator--outcome confounding after conditioning
on `Z`. Assigning measured variables to the SFM `Z` set makes the adjustment
set explicit; it does not by itself prove that no omitted confounders exist.

## Synthetic validation and tests

The synthetic SCM contains a known nonzero X-W interaction, so it tests both
effect recovery and direct TE estimation:

```bash
uv run python -m evaluation.synthetic_validation
uv run python -m unittest discover -s tests -v
```

## Uncertainty

The table intervals currently resample individuals while holding the fitted
flow and classifier fixed. They are conditional, not full-pipeline uncertainty
intervals. Submission experiments should additionally repeat data splitting,
flow fitting, and classifier fitting across seeds and summarize variation
across refits. The code and captions deliberately do not claim otherwise.

The complete multi-seed runner isolates every seed's tensors and fitted models,
then reports the mean, sample standard deviation, and range across refits:

```bash
uv run python -m evaluation.run_multiseed \
  --datasets bar acs --seeds 42 43 44 45 46 --resume
```

Per-seed artifacts and the combined `summary.{json,csv,md}` are written under
`outputs/multiseed/`. `--resume` skips a stage only when its complete expected
output set exists.

## Datasets

- ACS Income (California 2018): sex is `X`; education, occupation, and weekly
  hours are mediators.
- Law School: race is `X`; LSAT and undergraduate GPA are mediators.

For both supplied configurations, value 0 is the disadvantaged group and value
1 is the advantaged natural reference. These values are explicit YAML fields.

## Repository layout

```text
flows/                         block-ordered mediator model and interventions
outcome/                       fixed logistic/MLP classifiers
evaluation/estimate_gap.py     direct mediation decomposition
evaluation/compute_recourse.py primary recourse experiment
evaluation/ablate_recourse.py  eta x lambda x rho ablation
evaluation/recourse_metrics.py estimands and mixed-type metric
evaluation/synthetic_validation.py
tests/test_core.py
conf/dataset/                  dataset and experiment configuration
```
