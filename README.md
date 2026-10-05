# Mediation-aware causal recourse

Research code for diagnosing whether classifier-satisfying recourse closes or
widens a disparity carried through actionable mediators. The estimands in this
repository are effects on a fixed classifier `f_hat`, not effects on the
observed outcome `Y`.

## Important revision note

The current source corrects five errors in the original experiment code:

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
  the learned mechanisms; and
- the primary method no longer trades validity and fairness through soft
  objective weights. It minimizes action cost under explicit stochastic
  validity and recipient-conditional prediction-distribution constraints,
  abstaining when no candidate satisfies both.

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

The revised mixed model uses configured partially ordered blocks. For the 2019
ACS Income sample pooled across California and New York:

```text
p(Education,Occupation,Hours,Weeks | X,Z)
  = p(Education | X,Z)
    p(Occupation | Education,X,Z)
    p(Hours,Weeks | Education,Occupation,X,Z).
```

`[Hours,Weeks]` is one unordered joint continuous block, modeled by a
multivariate conditional flow. Here `Weeks` is weeks worked during the past 12
months. Commute time is deliberately excluded from this experiment. ACS
checkpoints are selected using held-out checks of categorical and continuous
marginals by sex, plus the correlation between the continuous mediators.

UCI Adult uses the same ordered block structure, with weekly hours represented
by eight ordered bins (including a separate bin for the point mass at 40).
Age is standardized using training-split statistics only inside the mediator
mechanisms. Adult checkpoints must also pass held-out posterior-predictive
gates on overall and sex-stratified categorical marginals and ordered-hours
Wasserstein distance. Raw files are combined before each seed-specific split;
see `data/ADULT_DATA_PROVENANCE.md` for provenance.

German Credit follows the graph `C -> {S,R,Y}`, `A -> {S,Y}`, `S -> {R,Y}`,
and `R -> Y`. Here `A` is sex, `C` contains age and loan purpose, `S` jointly
contains checking balance, savings, and housing, and `R` jointly contains
credit amount and repayment duration. The experiment adopts the compatible
all-`S`-paths estimand: the direct path and every path beginning `A -> S`,
including `A -> S -> R -> Y`, are treated as unfair. Same-level coordinates
inside `S` and `R` are jointly modeled without causal order. Each seed uses a
random 900/100 train/test split. Because the dataset has only 1,000
observations, it uses logistic regression and a class-balanced random forest;
see `data/GERMAN_CREDIT_DATA_PROVENANCE.md`.


Synthetic German Credit (`dataset=german_synth`) is a semi-synthetic SCM
adapted from Karimi et al. (2020), *Algorithmic recourse under imperfect
causal knowledge*, and designed to carry a large mediated disparity. Sex and
age are roots. The mediator chain is
`EDUCATION -> JOB -> SAVINGS_GRP -> [CREDIT_AMOUNT, DURATION]`: three ordinal
ordered-logit blocks, then one joint log-normal block. `AGE` is a parent of
every mediator and of `GOOD_CREDIT`, and sex shifts every discrete mediator in
favour of men. Karimi's `L -> D` edge is replaced by correlated noise
(rho=0.8), so the configured blocks match the generating process exactly.
Amount and duration medians are calibrated to UCI German Credit. The
population (10,000 rows, `data_seed=0`) is fixed, and the experiment seed only
changes the 8,000/2,000 split. `data.german_synth.oracle_mediation_effects`
gives the true effects on P(Y=1): NDE of about +0.04, NIE of about +0.19 and
TE of about +0.22, so roughly 85% of the gap is mediated. Use these as ground
truth when checking flow-based estimates.
Setting `dataset.gap.oracle_module` makes `estimate_gap` also score each
classifier with true-SCM mediator draws; it writes
`outputs/gaps/german_synth_gender_gap_oracle.{json,txt}`. This dataset uses a
validity target of `min_success_probability=0.5`. At 0.8, fewer than half of
the draws from the advantaged reference cleared the threshold, so ordinary
recourse overshot and closed the mediated gap by default. For a matched
comparison with Bar Passage, Bar also uses 0.5. The synthetic config copies
Bar's recourse sampling settings and Bar's random-forest hyperparameters, and
fits logistic regression, an MLP and a random forest. `run_multiseed --resume`
reruns any recourse artifact whose validity target differs from the config.

OULAD uses declared disability as the sensitive attribute and restricts the
target population to students who remain registered and have used the VLE by
day 60. Log click intensity, active days, and distinct resources accessed form
one same-level continuous engagement block; see
`data/OULAD_DATA_PROVENANCE.md`. The seed-42 screen found that this pathway is
small, so OULAD should be treated as a negative calibration result rather than
a headline recourse benchmark.

For Law School, `[LSAT,GPA]` is one unordered block modeled by one multivariate
conditional flow, `p(LSAT,GPA | X,Z)`. The autoregressive coordinate transform
inside that flow is only a density parameterization; it is not interpreted as
an LSAT-to-GPA or GPA-to-LSAT causal arrow. `sfm.mediator_layers` records these
assumptions and is saved in new checkpoints.

```bash
uv run python -m flows.train_flow
uv run python -m flows.train_flow dataset=bar
uv run python -m flows.train_flow dataset=adult
uv run python -m flows.train_flow dataset=german
uv run python -m flows.train_flow dataset=german_synth
uv run python -m flows.train_flow dataset=oulad
# Simpler conditional-Gaussian ablation (use a separate checkpoint):
uv run python -m flows.train_flow dataset.flow.continuous_family=gaussian \
  dataset.paths.flows=outputs/flows/acs/gaussian_models.pt
```

Outputs are written to `outputs/flows/` and `outputs/data/`.

### 2. Train fixed outcome classifiers

The Bar Passage configuration fits logistic regression, an MLP, and a
random-forest robustness model. ACS and Adult fit logistic regression and an
MLP; German Credit fits logistic regression and a class-balanced random forest.

```bash
uv run python -m outcome.train_outcome
uv run python -m outcome.train_outcome dataset=bar
uv run python -m outcome.train_outcome dataset=adult
uv run python -m outcome.train_outcome dataset=german
uv run python -m outcome.train_outcome dataset=german_synth
uv run python -m outcome.train_outcome dataset=oulad

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
non-descendants remain factual. The primary method solves

```text
minimize    cost(a)
subject to  Pr_{W~Q_a(.|z)}[f_hat(X=disadvantaged,W,z) >= 0.5] >= 0.8
            W1(f_hat(X=disadvantaged,W,z),
               f_hat(X=disadvantaged,W_adv,z)) <= 0.10,
            W_adv ~ P(W | X=advantaged,z).
```

The Wasserstein constraint is one-dimensional and acts on prediction draws,
not on raw mixed-type mediator coordinates. Both distributions use the
recipient's fixed `z` and disadvantaged sensitive value; only their mediator
distributions differ. If no candidate is feasible, the method returns the
zero-cost factual plan and records an abstention rather than dropping the
recipient or choosing a weighted compromise.

The earlier soft objective remains available as an ablation:

```text
cost(a) + eta*shortfall + lambda*direct_effect + rho*U_mix.
```

The primary baseline, `ordinary_actionable_recourse`, minimizes the same cost
under the same 80% factual-validity constraint and abstention rule, but omits
the prediction-W1 constraint. Thus the primary comparison isolates the W1
constraint. The weighted formulation is retained under
`soft_objective_recourse` and related legacy ablation labels. `U_mix` is an
independent-coupling upper bound on mixed-metric mediator transport, not the
primary method's prediction-space W1.

Each constrained run also records compatibility diagnostics before selecting an
action: the validity rate of the natural advantaged-mediator reference, the
minimum prediction-W1 attainable by any valid action for each recipient, and
joint constraint coverage over a pre-specified epsilon grid. These diagnostics
separate lack of actionable recourse from incompatibility between the validity
target and the chosen reference distribution; the grid is descriptive and is
not tuned on test performance.

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

The primary recipient-level percentage closed is descriptive and uses the
recipients' observed pre-recourse state without allowing positive and negative
recipient gaps to cancel:

```text
recipient-level closure
    = 1 - mean(|D_post,i|) / mean(|D_pre_factual,i|).
```

The code retains `1-|mean(D_post)|/|mean(D_pre_factual)|` as a legacy
aggregate-gap diagnostic, and additionally reports overshoot rate and magnitude
plus paired absolute-gap improvement over ordinary recourse. Model-based
`D_pre` remains a natural-distribution mediation diagnostic, but is not used
as the closure denominator after selecting recipients by their factual
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
  --datasets bar acs adult german oulad --seeds 42 43 44 45 46 --resume
```

Per-seed artifacts and the combined `summary.{json,csv,md}` are written under
`outputs/multiseed/`. `--resume` skips a stage only when its complete expected
output set exists.

## Datasets

- ACS Income (California and New York, 2019): sex is `X`; education and
  occupation are ordered discrete mediators, followed by one joint block of
  usual weekly hours and weeks worked during the past year. Commute time is
  excluded.
- UCI Adult (1994 Census extract): sex is `X`; education, occupation, and
  ordered binned weekly hours are mediators, using the same causal block
  structure as ACS.
- Law School: race is `X`; LSAT and undergraduate GPA are mediators.
- German Credit: sex is `X`; checking, savings, and housing form the financial
  status block, followed by the joint credit-amount/duration block. Age and
  loan purpose are context; good credit risk is the favorable outcome. The
  symbolic UCI file preserves auditable category meanings.

For all supplied configurations, value 0 is the disadvantaged group and value
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
