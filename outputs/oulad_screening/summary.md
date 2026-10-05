# Complete multi-seed split–flow–classifier results

Seeds: 42. Values are mean ± sample SD across complete refits.

## OULAD

### Logistic Reg.

| Method | Post disparity | Recipient L1 closure | Overshoot | Constraint feasible | Cost | Paired absolute-gap gain |
|---|---:|---:|---:|---:|---:|---:|
| ordinary actionable recourse | -0.092 ± 0.000 | 15.758 ± 0.000% | 61.905 ± 0.000% | 100.000 ± 0.000% | 3.986 ± 0.000 | 0.000 ± 0.000 |
| mediation aware recourse | 0.142 ± 0.000 | 2.838 ± 0.000% | 0.000 ± 0.000% | 2.721 ± 0.000% | 0.032 ± 0.000 | -0.021 ± 0.000 |

Validation AUC: 0.755 ± 0.000; pure NIE: -0.003 ± 0.000; TE: +0.058 ± 0.000.

Constraint compatibility diagnostics:

Reference-valid: 8.2% ± 0.0%; valid-action coverage: 100.0% ± 0.0%; median minimum attainable W1: 0.186 ± 0.000.

| Epsilon | Joint coverage |
|---:|---:|
| 0.025 | 0.0% ± 0.0% |
| 0.050 | 0.0% ± 0.0% |
| 0.075 | 0.0% ± 0.0% |
| 0.100 | 2.7% ± 0.0% |
| 0.150 | 21.8% ± 0.0% |
| 0.200 | 59.2% ± 0.0% |
| 0.300 | 92.5% ± 0.0% |
| 0.400 | 99.3% ± 0.0% |
| 0.500 | 100.0% ± 0.0% |

### MLP (64--32)

| Method | Post disparity | Recipient L1 closure | Overshoot | Constraint feasible | Cost | Paired absolute-gap gain |
|---|---:|---:|---:|---:|---:|---:|
| ordinary actionable recourse | -0.116 ± 0.000 | 13.128 ± 0.000% | 61.972 ± 0.000% | 97.887 ± 0.000% | 2.920 ± 0.000 | 0.000 ± 0.000 |
| mediation aware recourse | 0.147 ± 0.000 | 2.138 ± 0.000% | 0.000 ± 0.000% | 1.408 ± 0.000% | 0.026 ± 0.000 | -0.019 ± 0.000 |

Validation AUC: 0.757 ± 0.000; pure NIE: -0.003 ± 0.000; TE: +0.067 ± 0.000.

Constraint compatibility diagnostics:

Reference-valid: 6.3% ± 0.0%; valid-action coverage: 97.9% ± 0.0%; median minimum attainable W1: 0.191 ± 0.000.

| Epsilon | Joint coverage |
|---:|---:|
| 0.025 | 0.0% ± 0.0% |
| 0.050 | 0.0% ± 0.0% |
| 0.075 | 0.7% ± 0.0% |
| 0.100 | 1.4% ± 0.0% |
| 0.150 | 17.6% ± 0.0% |
| 0.200 | 54.9% ± 0.0% |
| 0.300 | 95.8% ± 0.0% |
| 0.400 | 97.9% ± 0.0% |
| 0.500 | 97.9% ± 0.0% |
