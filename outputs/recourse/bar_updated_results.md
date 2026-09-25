# Updated Law School results (seed 42)

These are revised, single-split results from the block-aware intervention
pipeline. Intervals bootstrap recipients while holding the fitted mediator and
outcome models fixed; they are not full-pipeline uncertainty intervals.

Population: Black held-out true negatives. The ordinary actionable-recourse
baseline uses the same action set, outcome threshold, costs, and intervention
semantics as the proposed method, with `lambda=0` and `rho=0`.

The comparison below fixes `eta=10`. The displayed mediation-aware sensitivity
row uses the marked effect-ratio heuristic for `lambda` and the strongest
tested anchor, `rho=10`; it should not be presented as a pre-registered primary
setting.

| Outcome model | Method | n | Post-recourse mediated prediction disparity | Factual disparity closed | Feasible | Cost |
|---|---:|---:|---:|---:|---:|---:|
| Logistic | Ordinary actionable baseline | 39 | 0.406 [0.397, 0.416] | 19.2% [15.8, 22.1] | 46.2% [30.8, 61.5] | 0.427 [0.308, 0.548] |
| Logistic | Mediation-aware sensitivity (`lambda=0.548`, `rho=10`) | 39 | 0.372 [0.358, 0.386] | 26.0% [24.0, 27.7] | 74.4% [59.0, 87.2] | 0.595 [0.459, 0.747] |
| MLP | Ordinary actionable baseline | 41 | 0.337 [0.321, 0.352] | 1.8% [1.0, 2.7] | 56.1% [41.5, 70.7] | 0.034 [0.015, 0.059] |
| MLP | Mediation-aware sensitivity (`lambda=0.614`, `rho=10`) | 41 | 0.319 [0.301, 0.334] | 7.3% [6.4, 8.4] | 78.0% [65.8, 90.2] | 0.228 [0.195, 0.264] |

Paired gains over ordinary actionable recourse:

- Logistic: +6.8 percentage points of factual disparity closure
  [4.7, 9.2], +28.2 points feasibility [15.4, 41.0], and +0.168 cost
  [0.111, 0.243].
- MLP: +5.5 percentage points of factual disparity closure [4.7, 6.5],
  +22.0 points feasibility [9.8, 34.1], and +0.194 cost [0.175, 0.215].

The direct-invariance component alone was essentially inactive on this
dataset. Most improvement came from the distributional anchor. This limitation
should be reported rather than attributing the full improvement to `lambda`.

Classifier-scale mediation diagnostics on 500 held-out confounder profiles:

- Logistic: pure NIE 0.211 [0.210, 0.213], directly sampled total effect
  0.268 [0.267, 0.270].
- MLP: pure NIE 0.192 [0.190, 0.193], directly sampled total effect
  0.263 [0.260, 0.265].

Do not call the post-recourse quantity an NIE. It compares the advantaged
natural mediator distribution with an intervention-induced distribution.
