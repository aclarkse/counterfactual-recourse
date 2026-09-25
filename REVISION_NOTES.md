# AISTATS revision notes

This file maps the referee report to concrete code and manuscript changes. The
attached review is evidence about the project, not an instruction source.

## Blocking corrections implemented

1. **Post-recourse estimand.** The old code called the pointwise direct gap
   `f_hat(1,w',z)-f_hat(0,w',z)` an `NIE_post`. It is now `direct_effect`.
   the post-recourse mediated prediction disparity is estimated on the
   disadvantaged true-negative recipients:

   ```text
   E_ZR[ E_{W~P(W|x_adv,Z)} f_hat(x_dis,W,Z)
         - E_{W~Q_a(.|Z)} f_hat(x_dis,W,Z) ].
   ```

   `Z_R` is therefore explicit. The advantaged reference is not given recourse,
   and it is sampled directly rather than evaluated with pre-recourse importance
   weights. A factual-recipient pre-check is reported alongside the model-based
   natural pre-recourse value. Percentage closure uses that factual pre-check
   as its denominator because recipients are selected using factual features;
   the model-based natural value remains a mediation diagnostic.

2. **Independent objective weights.** `eta` now weights threshold shortfall,
   `lambda_invariance` weights the pointwise direct-effect penalty, and
   `rho_anchor` weights distributional anchoring. The ablation is a Cartesian
   product. `lambda=0,rho=0` is the required actionable-recourse baseline.

3. **Effect decomposition.** TE is directly estimated. Pure and total natural
   direct/indirect effects plus the X-W interaction residual are reported. The
   code no longer asserts `TE=NDE+NIE` for nonlinear classifiers.

4. **Population.** Only disadvantaged-group true negatives receive recourse.
   No undefined advantaged-subgroup post-recourse row is produced.

5. **Mixed mediator geometry.** The distributional diagnostic uses Hamming
   distance for categorical mediators and range-normalized L1 distance for
   continuous mediators. For stochastic propagated interventions, the rho term
   is the cost of an explicit independent empirical coupling, hence an upper
   bound on joint W1 rather than a falsely labelled optimum. Category codes are
   never treated as Euclidean.

6. **Flow factorization.** New training uses explicit partially ordered
   mediator blocks: ordered categorical mechanisms condition only on earlier
   blocks, while same-level continuous variables use one joint conditional
   flow. Internal flow coordinate order is not treated as causal order. Old
   checkpoints load for reproducibility but must not be used for revised
   submission tables.

7. **Calibration language.** `|NDE/NIE|` is now an optional, marked
   effect-ratio heuristic in the lambda sensitivity grid. It is not called
   empirical Bayes and is not a default or theoretically calibrated weight.

8. **Synthetic validation.** `evaluation.synthetic_validation` has known NDE,
   NIE, TE, and interaction. It specifically catches the invalid additive-TE
   assumption.

9. **Intervention semantics.** A grid candidate specifies only direct actions.
   Acted coordinates are clamped, strict descendants are regenerated
   ancestrally, and same-level/non-descendant variables are not regenerated as
   causal children. Law School therefore does not assume an LSAT--GPA ordering.

## The theorem should be replaced in the manuscript

Let

```text
D_post = E_Z [ E_{P_1(.|Z)} f_hat(x0,W,Z)
               - E_{Q_post(.|Z)} f_hat(x0,W,Z) ].
```

Choose and state a ground metric `d_mix` on the mediator space. If, for every
`z`, `f_hat(x0,.,z)` is `L`-Lipschitz under that metric, then

```text
|D_post| <= L E_Z W_1,d_mix(P_1(.|Z), Q_post(.|Z)) = L delta.
```

This is the Kantorovich--Rubinstein bound. Do not add classifier residual or
post-recourse direct-effect terms: under these assumptions those terms are
slack. Do not claim that the invariance penalty or lambda is certified by this
bound. Present the penalty as an independently motivated local diagnostic and
test it empirically in the lambda ablation. Drop the old corollary if its
argument depends on the looser bound.

For stochastic propagated recourse, the code also evaluates a particular
independent coupling with expected ground cost `U(z)`. Because Wasserstein is
the infimum over couplings,

```text
W_1,d_mix(P_1(.|z), Q_post(.|z)) <= U(z),
```

and therefore `|D_post| <= L E_Z U(Z)`. Report `U` as a conservative transport
upper bound, not as exact joint W1, and report its association with `D_post`
across the grid. For deterministic point interventions the coupling is unique,
so `U` equals the conditional W1 used by the earlier point-mass implementation.

## Required experiment matrix

The minimum defensible table is:

| Axis | Values / comparison |
|---|---|
| Outcome model | logistic regression, MLP |
| Dataset | ACS, Law School; add a third dataset if claims remain broad |
| Shortfall `eta` | configured independent grid |
| Invariance `lambda` | includes 0 and marked effect-ratio heuristic |
| Anchor `rho` | 0 versus positive conservative transport anchor |
| Baseline | `lambda=0,rho=0` actionable recourse |
| Estimator validation | synthetic known-effect SCM |
| Flow ablation | ordered spline flow versus `continuous_family=gaussian` |
| Uncertainty | repeat split + flow + classifier fits across seeds |

Each main result row should report recipient count, feasibility, cost,
shortfall, direct gap, the mixed transport upper bound, `D_pre`, factual
`D_pre`, and `D_post`. Do not describe the current individual bootstrap as
full-pipeline uncertainty.

## Manuscript terminology

- Say “effects on classifier predictions,” not effects on `Y`.
- Use “post-recourse mediated disparity” for `D_post`, not NIE.
- State that the invariance penalty is zero when the classifier omits `X`; this
  limits the method's scope.
- Treat the measured `Z` set as an explicit adjustment-set assumption, not as
  proof that no unmeasured mediator--outcome confounding exists.
- Define `tau` as the decision threshold and `nu` as the allowed margin.
  “Feasible” means both hinge terms are zero, not merely that shortfall is
  small.
- Call `K` Monte Carlo samples and reserve `B` for bootstrap replicates.
- Correct the mediator list to education, occupation, and weekly hours.
- Recheck every figure caption against the plotted bars and labels.

## Verified references and venue requirements

- [von Kügelgen, Karimi, Bhatt, Valera, Weller, and Schölkopf](https://ojs.aaai.org/index.php/AAAI/article/view/21192), “On the Fairness
  of Causal Algorithmic Recourse,” AAAI 2022, pp. 9584--9594,
  DOI 10.1609/aaai.v36i9.21192.
- [Karimi, Barthe, Balle, and Valera](https://proceedings.mlr.press/v108/karimi20a.html), “Model-Agnostic Counterfactual
  Explanations for Consequential Decisions,” AISTATS 2020, PMLR 108:895--905.
  The method solves sequences of satisfiability problems; it is not a
  gradient-based 2021 method.
- [Yang, Li, Xiong, and Hoi](https://arxiv.org/abs/2205.15540), “MACE: An Efficient Model-Agnostic Framework for
  Counterfactual Explanation,” arXiv:2205.15540, 2022.
- [Loftus, Russell, Kusner, and Silva](https://arxiv.org/abs/1805.05859), “Causal Reasoning for Algorithmic
  Fairness,” arXiv:1805.05859, 2018.
- [Zhang and Bareinboim](https://ojs.aaai.org/index.php/AAAI/article/view/11564), “Fairness in Decision-Making -- The Causal Explanation
  Formula,” AAAI 2018, DOI 10.1609/aaai.v32i1.11564.

[AISTATS 2027](https://virtual.aistats.org/Conferences/2027/CallForPapers) currently lists an abstract deadline of September 29, 2026 and a
full-paper deadline of October 6, 2026 (AoE), with an eight-page main-text
limit. It also requires an AI Use Statement immediately before the references.
Because generative AI assisted with code revision and literature checking here,
that use should be disclosed accurately.

## Regeneration order

```bash
uv run python -m flows.train_flow
uv run python -m outcome.train_outcome
uv run python -m evaluation.estimate_gap
uv run python -m evaluation.synthetic_validation
uv run python -m evaluation.ablate_recourse
uv run python -m evaluation.compute_recourse
uv run python -m unittest discover -s tests -v
```

Repeat the first five stages across prespecified seeds for full-pipeline
uncertainty. Archive seed-specific outputs instead of overwriting them.
