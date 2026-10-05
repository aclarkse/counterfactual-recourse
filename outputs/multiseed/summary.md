# Complete multi-seed split–flow–classifier results

Seeds: 42, 43, 44, 45, 46. Values are mean ± sample SD across complete refits.

## ACS

### Logistic Reg.

| Method | Post disparity | Recipient L1 closure | Overshoot | Constraint feasible | Cost | Paired absolute-gap gain |
|---|---:|---:|---:|---:|---:|---:|
| ordinary actionable recourse | -0.209 ± 0.011 | -2.799 ± 6.531% | 81.040 ± 3.354% | 100.000 ± 0.000% | 0.647 ± 0.058 | 0.000 ± 0.000 |
| mediation aware recourse | 0.105 ± 0.016 | 36.985 ± 4.769% | 17.200 ± 2.117% | 27.520 ± 4.395% | 0.229 ± 0.033 | 0.084 ± 0.018 |

Validation AUC: 0.864 ± 0.001; pure NIE: +0.024 ± 0.015; TE: +0.097 ± 0.014.

Constraint compatibility diagnostics:

Reference-valid: 22.4% ± 3.2%; valid-action coverage: 100.0% ± 0.0%; median minimum attainable W1: 0.149 ± 0.009.

| Epsilon | Joint coverage |
|---:|---:|
| 0.025 | 0.0% ± 0.0% |
| 0.050 | 5.4% ± 1.4% |
| 0.075 | 15.1% ± 3.9% |
| 0.100 | 27.5% ± 4.4% |
| 0.150 | 50.4% ± 4.4% |
| 0.200 | 72.7% ± 3.7% |
| 0.300 | 88.4% ± 4.2% |
| 0.400 | 99.0% ± 0.8% |
| 0.500 | 99.9% ± 0.2% |

### MLP (64--32)

| Method | Post disparity | Recipient L1 closure | Overshoot | Constraint feasible | Cost | Paired absolute-gap gain |
|---|---:|---:|---:|---:|---:|---:|
| ordinary actionable recourse | -0.202 ± 0.014 | 2.343 ± 4.462% | 80.960 ± 4.201% | 99.120 ± 1.073% | 1.105 ± 0.170 | 0.000 ± 0.000 |
| mediation aware recourse | 0.109 ± 0.012 | 38.434 ± 5.039% | 16.080 ± 3.230% | 27.600 ± 3.137% | 0.274 ± 0.022 | 0.078 ± 0.010 |

Validation AUC: 0.879 ± 0.001; pure NIE: +0.010 ± 0.012; TE: +0.100 ± 0.011.

Constraint compatibility diagnostics:

Reference-valid: 21.6% ± 5.4%; valid-action coverage: 99.1% ± 1.1%; median minimum attainable W1: 0.155 ± 0.005.

| Epsilon | Joint coverage |
|---:|---:|
| 0.025 | 0.0% ± 0.0% |
| 0.050 | 2.3% ± 1.2% |
| 0.075 | 14.3% ± 2.9% |
| 0.100 | 27.6% ± 3.1% |
| 0.150 | 47.0% ± 2.7% |
| 0.200 | 68.7% ± 2.6% |
| 0.300 | 86.2% ± 3.4% |
| 0.400 | 96.7% ± 2.6% |
| 0.500 | 98.6% ± 1.7% |

### Random Forest

| Method | Post disparity | Recipient L1 closure | Overshoot | Constraint feasible | Cost | Paired absolute-gap gain |
|---|---:|---:|---:|---:|---:|---:|
| ordinary actionable recourse | -0.203 ± 0.017 | -2.019 ± 10.697% | 84.400 ± 4.118% | 98.720 ± 1.145% | 1.004 ± 0.141 | 0.000 ± 0.000 |
| mediation aware recourse | 0.112 ± 0.018 | 32.999 ± 8.968% | 16.320 ± 4.180% | 26.000 ± 6.835% | 0.269 ± 0.032 | 0.072 ± 0.018 |

Validation AUC: 0.880 ± 0.001; pure NIE: +0.013 ± 0.012; TE: +0.086 ± 0.013.

Constraint compatibility diagnostics:

Reference-valid: 21.6% ± 7.7%; valid-action coverage: 98.7% ± 1.1%; median minimum attainable W1: 0.150 ± 0.014.

| Epsilon | Joint coverage |
|---:|---:|
| 0.025 | 0.0% ± 0.0% |
| 0.050 | 2.2% ± 1.6% |
| 0.075 | 11.7% ± 6.4% |
| 0.100 | 26.0% ± 6.8% |
| 0.150 | 49.8% ± 6.4% |
| 0.200 | 70.2% ± 3.5% |
| 0.300 | 84.1% ± 2.1% |
| 0.400 | 96.7% ± 2.2% |
| 0.500 | 98.7% ± 1.1% |

## ADULT

### Logistic Reg.

| Method | Post disparity | Recipient L1 closure | Overshoot | Constraint feasible | Cost | Paired absolute-gap gain |
|---|---:|---:|---:|---:|---:|---:|
| ordinary actionable recourse | -0.244 ± 0.021 | -199.824 ± 28.882% | 53.280 ± 4.694% | 69.520 ± 4.625% | 0.650 ± 0.052 | 0.000 ± 0.000 |
| mediation aware recourse | 0.042 ± 0.013 | 0.000 ± 0.000% | 0.000 ± 0.000% | 0.000 ± 0.000% | 0.000 ± 0.000 | 0.167 ± 0.020 |

Validation AUC: 0.830 ± 0.007; pure NIE: +0.031 ± 0.002; TE: +0.183 ± 0.005.

Constraint compatibility diagnostics:

Reference-valid: 0.0% ± 0.0%; valid-action coverage: 69.5% ± 4.6%; median minimum attainable W1: 0.304 ± 0.010.

| Epsilon | Joint coverage |
|---:|---:|
| 0.025 | 0.0% ± 0.0% |
| 0.050 | 0.0% ± 0.0% |
| 0.075 | 0.0% ± 0.0% |
| 0.100 | 0.0% ± 0.0% |
| 0.150 | 1.0% ± 1.0% |
| 0.200 | 8.4% ± 2.5% |
| 0.300 | 33.7% ± 1.6% |
| 0.500 | 69.5% ± 4.6% |

### MLP (64--32)

| Method | Post disparity | Recipient L1 closure | Overshoot | Constraint feasible | Cost | Paired absolute-gap gain |
|---|---:|---:|---:|---:|---:|---:|
| ordinary actionable recourse | -0.247 ± 0.027 | -247.567 ± 23.672% | 50.160 ± 2.865% | 64.320 ± 4.303% | 0.530 ± 0.062 | 0.000 ± 0.000 |
| mediation aware recourse | 0.040 ± 0.006 | 0.000 ± 0.000% | 0.000 ± 0.000% | 0.000 ± 0.000% | 0.000 ± 0.000 | 0.180 ± 0.022 |

Validation AUC: 0.844 ± 0.007; pure NIE: +0.025 ± 0.002; TE: +0.175 ± 0.008.

Constraint compatibility diagnostics:

Reference-valid: 0.0% ± 0.0%; valid-action coverage: 64.3% ± 4.3%; median minimum attainable W1: 0.308 ± 0.009.

| Epsilon | Joint coverage |
|---:|---:|
| 0.025 | 0.0% ± 0.0% |
| 0.050 | 0.0% ± 0.0% |
| 0.075 | 0.0% ± 0.0% |
| 0.100 | 0.0% ± 0.0% |
| 0.150 | 0.0% ± 0.0% |
| 0.200 | 0.3% ± 0.5% |
| 0.300 | 26.1% ± 9.8% |
| 0.500 | 64.3% ± 4.3% |

### Random Forest

| Method | Post disparity | Recipient L1 closure | Overshoot | Constraint feasible | Cost | Paired absolute-gap gain |
|---|---:|---:|---:|---:|---:|---:|
| ordinary actionable recourse | -0.202 ± 0.014 | -183.462 ± 2.964% | 44.080 ± 4.024% | 59.440 ± 4.258% | 0.423 ± 0.063 | 0.000 ± 0.000 |
| mediation aware recourse | 0.036 ± 0.005 | 0.000 ± 0.000% | 0.000 ± 0.000% | 0.000 ± 0.000% | 0.000 ± 0.000 | 0.140 ± 0.008 |

Validation AUC: 0.843 ± 0.006; pure NIE: +0.024 ± 0.002; TE: +0.159 ± 0.004.

Constraint compatibility diagnostics:

Reference-valid: 0.0% ± 0.0%; valid-action coverage: 59.4% ± 4.3%; median minimum attainable W1: 0.303 ± 0.005.

| Epsilon | Joint coverage |
|---:|---:|
| 0.025 | 0.0% ± 0.0% |
| 0.050 | 0.0% ± 0.0% |
| 0.075 | 0.0% ± 0.0% |
| 0.100 | 0.0% ± 0.0% |
| 0.150 | 0.0% ± 0.0% |
| 0.200 | 0.0% ± 0.0% |
| 0.300 | 28.2% ± 4.4% |
| 0.500 | 59.4% ± 4.3% |

## BAR

### Logistic Reg.

| Method | Post disparity | Recipient L1 closure | Overshoot | Constraint feasible | Cost | Paired absolute-gap gain |
|---|---:|---:|---:|---:|---:|---:|
| ordinary actionable recourse | 0.351 ± 0.009 | 29.143 ± 1.524% | 0.000 ± 0.000% | 100.000 ± 0.000% | 0.862 ± 0.138 | 0.000 ± 0.000 |
| mediation aware recourse | 0.063 ± 0.004 | 87.366 ± 0.870% | 0.513 ± 1.147% | 100.000 ± 0.000% | 6.097 ± 0.462 | 0.289 ± 0.007 |

Validation AUC: 0.769 ± 0.010; pure NIE: +0.214 ± 0.010; TE: +0.272 ± 0.006.

Constraint compatibility diagnostics:

Reference-valid: 100.0% ± 0.0%; valid-action coverage: 100.0% ± 0.0%; median minimum attainable W1: 0.069 ± 0.003.

| Epsilon | Joint coverage |
|---:|---:|
| 0.025 | 0.0% ± 0.0% |
| 0.050 | 0.0% ± 0.0% |
| 0.075 | 85.3% ± 11.3% |
| 0.100 | 100.0% ± 0.0% |
| 0.150 | 100.0% ± 0.0% |
| 0.200 | 100.0% ± 0.0% |
| 0.300 | 100.0% ± 0.0% |
| 0.500 | 100.0% ± 0.0% |

### MLP (64--32)

| Method | Post disparity | Recipient L1 closure | Overshoot | Constraint feasible | Cost | Paired absolute-gap gain |
|---|---:|---:|---:|---:|---:|---:|
| ordinary actionable recourse | 0.321 ± 0.021 | 23.866 ± 5.261% | 0.000 ± 0.000% | 100.000 ± 0.000% | 1.039 ± 0.105 | 0.000 ± 0.000 |
| mediation aware recourse | 0.083 ± 0.052 | 78.753 ± 16.662% | 3.714 ± 3.466% | 85.665 ± 20.204% | 6.019 ± 1.023 | 0.237 ± 0.071 |

Validation AUC: 0.769 ± 0.005; pure NIE: +0.213 ± 0.017; TE: +0.277 ± 0.009.

Constraint compatibility diagnostics:

Reference-valid: 100.0% ± 0.0%; valid-action coverage: 100.0% ± 0.0%; median minimum attainable W1: 0.087 ± 0.008.

| Epsilon | Joint coverage |
|---:|---:|
| 0.025 | 0.0% ± 0.0% |
| 0.050 | 0.0% ± 0.0% |
| 0.075 | 16.9% ± 7.2% |
| 0.100 | 85.7% ± 20.2% |
| 0.150 | 100.0% ± 0.0% |
| 0.200 | 100.0% ± 0.0% |
| 0.300 | 100.0% ± 0.0% |
| 0.500 | 100.0% ± 0.0% |

### Random Forest

| Method | Post disparity | Recipient L1 closure | Overshoot | Constraint feasible | Cost | Paired absolute-gap gain |
|---|---:|---:|---:|---:|---:|---:|
| ordinary actionable recourse | 0.306 ± 0.017 | 28.094 ± 3.584% | 0.000 ± 0.000% | 100.000 ± 0.000% | 0.681 ± 0.147 | 0.000 ± 0.000 |
| mediation aware recourse | 0.063 ± 0.006 | 85.098 ± 1.284% | 0.000 ± 0.000% | 100.000 ± 0.000% | 4.433 ± 0.790 | 0.243 ± 0.016 |

Validation AUC: 0.755 ± 0.011; pure NIE: +0.193 ± 0.009; TE: +0.280 ± 0.005.

Constraint compatibility diagnostics:

Reference-valid: 100.0% ± 0.0%; valid-action coverage: 100.0% ± 0.0%; median minimum attainable W1: 0.058 ± 0.003.

| Epsilon | Joint coverage |
|---:|---:|
| 0.025 | 0.0% ± 0.0% |
| 0.050 | 14.6% ± 17.5% |
| 0.075 | 98.3% ± 1.5% |
| 0.100 | 100.0% ± 0.0% |
| 0.150 | 100.0% ± 0.0% |
| 0.200 | 100.0% ± 0.0% |
| 0.300 | 100.0% ± 0.0% |
| 0.500 | 100.0% ± 0.0% |

## GERMAN_SYNTH

### Logistic Reg.

| Method | Post disparity | Recipient L1 closure | Overshoot | Constraint feasible | Cost | Paired absolute-gap gain |
|---|---:|---:|---:|---:|---:|---:|
| ordinary actionable recourse | 0.124 ± 0.014 | 72.423 ± 1.266% | 15.600 ± 4.604% | 100.000 ± 0.000% | 0.657 ± 0.005 | 0.000 ± 0.000 |
| mediation aware recourse | 0.003 ± 0.005 | 91.486 ± 0.753% | 51.440 ± 3.683% | 98.560 ± 0.669% | 0.748 ± 0.017 | 0.100 ± 0.009 |

Validation AUC: 0.884 ± 0.008; pure NIE: +0.191 ± 0.004; TE: +0.219 ± 0.004.

Constraint compatibility diagnostics:

Reference-valid: 100.0% ± 0.0%; valid-action coverage: 100.0% ± 0.0%; median minimum attainable W1: 0.050 ± 0.002.

| Epsilon | Joint coverage |
|---:|---:|
| 0.025 | 3.6% ± 1.0% |
| 0.050 | 50.1% ± 4.9% |
| 0.075 | 89.9% ± 3.5% |
| 0.100 | 98.6% ± 0.7% |
| 0.150 | 99.7% ± 0.3% |
| 0.200 | 100.0% ± 0.0% |
| 0.300 | 100.0% ± 0.0% |
| 0.500 | 100.0% ± 0.0% |

### MLP (64--32)

| Method | Post disparity | Recipient L1 closure | Overshoot | Constraint feasible | Cost | Paired absolute-gap gain |
|---|---:|---:|---:|---:|---:|---:|
| ordinary actionable recourse | 0.098 ± 0.010 | 76.137 ± 1.238% | 17.040 ± 4.636% | 100.000 ± 0.000% | 0.645 ± 0.012 | 0.000 ± 0.000 |
| mediation aware recourse | 0.004 ± 0.004 | 91.233 ± 0.441% | 49.440 ± 4.687% | 98.640 ± 0.219% | 0.737 ± 0.005 | 0.076 ± 0.004 |

Validation AUC: 0.885 ± 0.005; pure NIE: +0.185 ± 0.016; TE: +0.219 ± 0.012.

Constraint compatibility diagnostics:

Reference-valid: 100.0% ± 0.0%; valid-action coverage: 100.0% ± 0.0%; median minimum attainable W1: 0.044 ± 0.003.

| Epsilon | Joint coverage |
|---:|---:|
| 0.025 | 6.4% ± 3.0% |
| 0.050 | 64.2% ± 6.9% |
| 0.075 | 93.3% ± 2.1% |
| 0.100 | 98.6% ± 0.2% |
| 0.150 | 99.5% ± 0.7% |
| 0.200 | 99.8% ± 0.4% |
| 0.300 | 100.0% ± 0.0% |
| 0.500 | 100.0% ± 0.0% |

### Random Forest

| Method | Post disparity | Recipient L1 closure | Overshoot | Constraint feasible | Cost | Paired absolute-gap gain |
|---|---:|---:|---:|---:|---:|---:|
| ordinary actionable recourse | 0.106 ± 0.006 | 75.495 ± 1.272% | 14.960 ± 2.907% | 100.000 ± 0.000% | 0.653 ± 0.010 | 0.000 ± 0.000 |
| mediation aware recourse | -0.007 ± 0.005 | 91.716 ± 0.488% | 59.120 ± 3.012% | 98.960 ± 0.607% | 0.747 ± 0.011 | 0.081 ± 0.006 |

Validation AUC: 0.885 ± 0.006; pure NIE: +0.182 ± 0.003; TE: +0.203 ± 0.004.

Constraint compatibility diagnostics:

Reference-valid: 100.0% ± 0.0%; valid-action coverage: 100.0% ± 0.0%; median minimum attainable W1: 0.048 ± 0.003.

| Epsilon | Joint coverage |
|---:|---:|
| 0.025 | 3.7% ± 1.0% |
| 0.050 | 56.1% ± 6.9% |
| 0.075 | 92.6% ± 2.7% |
| 0.100 | 99.0% ± 0.6% |
| 0.150 | 99.7% ± 0.3% |
| 0.200 | 100.0% ± 0.0% |
| 0.300 | 100.0% ± 0.0% |
| 0.500 | 100.0% ± 0.0% |
