This repository accompanies the paper *From "What if" to "How to": Mediation-Aware Causal Recourse via the Standard Fairness Model*, and implements a framework for generating algorithmic recourse recommendations that are both causally grounded and equitable across sensitive subgroups, given a fixed predictive model.

## Overview

Standard algorithmic recourse asks how an individual can change their features to flip an unfavorable decision. When the predictor exhibits group-level disparities, however, recourse generated without reference to causal structure can entrench those disparities, recommending changes along pathways that are immutable in practice, or ignoring the mechanism by which the disparity arises in the first place.

We address this by embedding recourse in the Standard Fairness Model (SFM) of Plečko and Bareinboim, which partitions covariates into a sensitive attribute $X$, confounders $Z$, and mediators $W$.

## Datasets

The repository illustrates the use of this pipeline on two datasets:

- **Law School**: `race` is selected as the sensitive attribute, `LSAT` and undergraduate `GPA` as mediators, and bar passage (`bar_pass`) as the outcome.
- **Folktables ACSIncome (California, 2018)**: `SEX` is selected as the sensitive attribute, education group (`SCHL_GRP`), occupation group (`OCCP_GRP`), and weekly hours worked (`WKHP`) as mediators, and income above \$50k as the outcome.

## Installation

```bash
# Install all dependencies. PyTorch is pinned to a CUDA 12.8 build in
# pyproject.toml / uv.lock — cu128 ships the sm_120 kernels required by
# Blackwell / RTX 50-series GPUs (e.g. RTX 5090).
uv sync
```

> Need a different CUDA build or a CPU-only install? Repoint `torch`/`torchvision`
> in `[tool.uv.sources]` (`pyproject.toml`) to another `pytorch-*` index — e.g.
> `pytorch-cpu` or `pytorch-cu126` — then re-run `uv lock && uv sync`. Run
> `nvidia-smi` to check your driver's CUDA version.

All commands below are prefixed with `uv run`, which executes them inside the
uv-managed environment — you do **not** need to activate the venv first. (If you
prefer, you can `source .venv/bin/activate` once and drop the `uv run` prefix.)

## Pipeline

The implementation is organised as a five-stage pipeline. All entry points are run as Python modules from the project root via `uv run` and are driven by [Hydra](https://hydra.cc/) configs in `conf/dataset/`. Switch datasets by passing `dataset=bar` instead of the default `dataset=acs`.

### Stage 1 — Flow training: $\Pr(W \mid X, Z)$

Trains a mixed-type conditional normalising flow that factorises as

$$\Pr(W \mid X, Z) = \Pr(W_d \mid X, Z) \cdot \Pr(W_c \mid W_d, X, Z)$$

with a categorical MLP head for discrete mediators and a Neural Spline Flow for the continuous component.

```bash
uv run python -m flows.train_flow                        # ACS (default)
uv run python -m flows.train_flow dataset=bar            # Law School
uv run python -m flows.train_flow dataset=acs dataset.loader.kwargs.year=2019
uv run python -m flows.train_flow dataset=acs dataset.flow.max_epochs=200
```

**Outputs:** `outputs/flows/{acs,law_school}/flow_models.pt`, `outputs/data/{acs,bar}_tensors.pt`

---

### Stage 2 — Outcome model training: $\Pr(Y \mid X, W, Z)$

Fits two outcome models (logistic regression and MLP) on the training split and evaluates them on the validation split, stratified by the sensitive attribute.

```bash
uv run python -m outcome.train_outcome                   # ACS (default)
uv run python -m outcome.train_outcome dataset=bar
```

**Outputs:** `outputs/outcome/{acs,bar}/{logreg,mlp}.joblib`

---

### Stage 3 — Mediation decomposition: NDE and NIE

Estimates the Natural Direct Effect (NDE) and Natural Indirect Effect (NIE) of the sensitive attribute on the outcome using self-normalised importance sampling over exact flow log-densities:

$$\mathrm{NDE}_i = \mathbb{E}_k\bigl[f(X{=}1,\,W_k,\,z_i) - f(X{=}0,\,W_k,\,z_i)\bigr], \quad W_k \sim \Pr(W \mid X{=}0,\,z_i)$$

$$\mathrm{NIE}_i = \sum_k \bar{r}_k\,f(X{=}0,W_k,z_i) - \mathbb{E}_k\bigl[f(X{=}0,W_k,z_i)\bigr], \quad \bar{r}_k \propto \frac{\Pr(W_k \mid X{=}1,\,z_i)}{\Pr(W_k \mid X{=}0,\,z_i)}$$

The empirical calibration $\lambda_{\mathrm{EB}} = |\mathrm{NDE}| / |\mathrm{NIE}|$ is saved to JSON for use in the recourse stage.

```bash
uv run python -m evaluation.estimate_gap                 # ACS (default)
uv run python -m evaluation.estimate_gap dataset=bar
uv run python -m evaluation.estimate_gap dataset=acs gap.K=1000 gap.n_inst=1000
```

**Outputs:** `outputs/gaps/{acs,bar}_gender_gap.{json,txt}`, LaTeX tables in `outputs/gaps/`

---

### Stage 4 — Recourse generation

For each true-negative individual ($y=0$, $\hat{y}=0$), finds the minimum-cost mediator intervention $w'$ subject to a soft local-fairness-invariance constraint:

$$\min_{w'}\; \mathrm{cost}(w, w') + \lambda \cdot S(w')$$

$$S(w') = \max(0,\,\tau{-}\nu - f(x_i, w', z_i)) + \max(0,\,\tau{-}\nu - f(1{-}x_i, w', z_i))$$

The penalty weight defaults to $\lambda = \lambda_{\mathrm{EB}}$ loaded from the gap JSON. Candidate generation is fully vectorised: all mediator combinations are enumerated via a Cartesian grid and evaluated in two bulk `predict_proba` calls per individual.

```bash
uv run python -m evaluation.compute_recourse             # ACS, all TN individuals
uv run python -m evaluation.compute_recourse dataset=bar
uv run python -m evaluation.compute_recourse dataset=acs recourse.n_max=500
uv run python -m evaluation.compute_recourse dataset=acs recourse.threshold=0.5 recourse.nu=0.1
```

**Outputs:** `outputs/recourse/acs_recourse.{tex,txt}`

#### Stage 4b — $\lambda$ sensitivity sweep

Sweeps $\lambda \in \{0,\,\lambda_{\mathrm{EB}}/4,\,\lambda_{\mathrm{EB}}/2,\,\lambda_{\mathrm{EB}},\,2\lambda_{\mathrm{EB}},\,4\lambda_{\mathrm{EB}},\,\infty\}$ on a subsample of true negatives. Uses PyTorch-accelerated inference (sklearn weights loaded into frozen `nn.Linear` / `nn.Sequential`) and caches the candidate pool to disk, so the sweep itself requires zero additional model evaluations.

```bash
uv run python -m evaluation.sweep_lambda                 # ACS, n=250 subsample
uv run python -m evaluation.sweep_lambda dataset=bar
uv run python -m evaluation.sweep_lambda dataset=acs recourse.sweep_n_max=500
uv run python -m evaluation.sweep_lambda dataset=acs --config-name config recourse.nu=0.1
```

**Outputs:** `outputs/recourse/sweep_lambda_{slug}_{stratum}.tex`, `outputs/recourse/sweep_lambda_acs.txt`

---

### Stage 5 — Evaluation

Post-recourse NIE ($\mathrm{NIE}_{\mathrm{post}}$) is computed at the selected recourse solution $w'_i$ and reported alongside cost, shortfall, and feasibility rate in the tables produced by Stages 4 and 4b:

$$\mathrm{NIE}_{\mathrm{post},i} = f(\text{Male},\,w'_i,\,z_i) - f(\text{Female},\,w'_i,\,z_i)$$

All summary statistics are reported as means with 95% percentile bootstrap confidence intervals.

---

## Flow diagnostics (optional)

After Stage 1, the trained conditional flow $\Pr(W \mid X, Z)$ can be inspected with the standalone scripts in `diagnostics/`. These are **not** part of the five-stage pipeline and nothing downstream imports them — they exist to sanity-check the generated flows before the gap and recourse stages, regenerating the figures in `figures/`: training curves, empirical-vs-model mediator marginals, the counterfactual shift $\Pr(W \mid X{=}1)$ vs $\Pr(W \mid X{=}0)$, and importance-sampling ESS by group (the IS quality that NDE/NIE in Stage 3 depends on).

```bash
uv run python diagnostics/inspect_acs_model.py   # ACS flow        → figures/acs_income/
uv run python diagnostics/inspect_bar_model.py   # Law School flow → figures/law_school/
```

Both accept `--model` / `--tensors` overrides; the defaults point at the Stage 1 outputs (`outputs/flows/.../flow_models.pt`, `outputs/data/..._tensors.pt`).

---

## Configuration

Dataset-specific parameters live in `conf/dataset/acs.yaml` and `conf/dataset/bar.yaml`. Any field can be overridden at the command line using Hydra dot-notation:

```bash
# Change ACS survey year and states
uv run python -m flows.train_flow dataset.loader.kwargs.year=2019 dataset.loader.kwargs.states=[CA,NY,TX]

# Increase IS samples for the gap decomposition
uv run python -m evaluation.estimate_gap gap.K=1000 gap.n_inst=2000 gap.n_boot=5000

# Tighter recourse threshold with more bootstrap iterations
uv run python -m evaluation.compute_recourse recourse.threshold=0.6 recourse.nu=0.05 recourse.n_boot=2000
```

## Project structure

```
conf/
  config.yaml              # Hydra root config
  dataset/
    acs.yaml               # ACS Income dataset config
    bar.yaml               # Law School dataset config
data/
  build_tensors.py         # Generic build_tensors, stack_sfm_features
  acs.py                   # load_acs_income
  bar.py                   # load_bar_data
flows/
  models.py                # DiscreteMediator, ContinuousMediatorFlow, load_flow_models
  train_flow.py            # Stage 1 entry point
  diagnostics.py           # make_sample_fns (IS sampling closures)
outcome/
  models.py                # _SFMPreprocess, TorchLogReg, TorchMLP, make_preprocessor
  train_outcome.py         # Stage 2 entry point
evaluation/
  estimate_gap.py          # Stage 3 entry point
  compute_recourse.py      # Stage 4 entry point
  sweep_lambda.py          # Stage 4b entry point
diagnostics/               # Optional flow-inspection scripts (not in the pipeline)
  inspect_acs_model.py     # ACS flow diagnostics       → figures/acs_income/
  inspect_bar_model.py     # Law School flow diagnostics → figures/law_school/
  flow_diagnostics_shared.py  # Shared plot helpers (ESS, conditional marginals)
  flow_diagnostics.py      # Standalone marginal/ESS plot helpers
  paper_style.py           # Matplotlib/seaborn paper styling
```
