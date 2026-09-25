# Complete multi-seed split–flow–classifier results

Seeds: 42, 43, 44, 45, 46. Values are mean ± sample SD across complete refits.

## ACS

### Logistic Reg.

| Method | Post disparity | Factual closure | Feasible | Cost | Closure gain vs baseline |
|---|---:|---:|---:|---:|---:|
| ordinary actionable recourse | -0.236 ± 0.025 | -27.936 ± 24.358% | 68.960 ± 2.090% | 0.572 ± 0.020 | 0.000 ± 0.000 pp |
| transport anchor only | -0.204 ± 0.019 | -10.994 ± 20.342% | 52.880 ± 5.362% | 0.768 ± 0.023 | 16.943 ± 5.796 pp |
| mediation aware recourse | -0.212 ± 0.021 | -15.363 ± 21.734% | 53.760 ± 5.724% | 0.766 ± 0.022 | 12.573 ± 5.137 pp |

Validation AUC: 0.862 ± 0.001; pure NIE: +0.038 ± 0.007; TE: +0.110 ± 0.006.

### MLP (64--32)

| Method | Post disparity | Factual closure | Feasible | Cost | Closure gain vs baseline |
|---|---:|---:|---:|---:|---:|
| ordinary actionable recourse | -0.239 ± 0.021 | -36.499 ± 22.744% | 61.120 ± 5.524% | 0.632 ± 0.046 | 0.000 ± 0.000 pp |
| transport anchor only | -0.224 ± 0.015 | -27.664 ± 17.536% | 49.520 ± 4.431% | 0.697 ± 0.042 | 8.835 ± 8.037 pp |
| mediation aware recourse | -0.268 ± 0.039 | -52.544 ± 24.367% | 54.640 ± 5.633% | 0.737 ± 0.057 | -16.045 ± 18.919 pp |

Validation AUC: 0.877 ± 0.001; pure NIE: +0.018 ± 0.008; TE: +0.106 ± 0.006.

## BAR

### Logistic Reg.

| Method | Post disparity | Factual closure | Feasible | Cost | Closure gain vs baseline |
|---|---:|---:|---:|---:|---:|
| ordinary actionable recourse | 0.411 ± 0.011 | 17.149 ± 1.532% | 54.821 ± 5.983% | 0.372 ± 0.064 | 0.000 ± 0.000 pp |
| transport anchor only | 0.373 ± 0.011 | 24.786 ± 0.869% | 80.692 ± 5.131% | 0.544 ± 0.078 | 7.637 ± 0.785 pp |
| mediation aware recourse | 0.373 ± 0.011 | 24.826 ± 0.853% | 80.692 ± 5.131% | 0.544 ± 0.078 | 7.677 ± 0.781 pp |

Validation AUC: 0.769 ± 0.010; pure NIE: +0.214 ± 0.010; TE: +0.272 ± 0.006.

### MLP (64--32)

| Method | Post disparity | Factual closure | Feasible | Cost | Closure gain vs baseline |
|---|---:|---:|---:|---:|---:|
| ordinary actionable recourse | 0.388 ± 0.030 | 8.152 ± 4.405% | 49.051 ± 6.162% | 0.172 ± 0.097 | 0.000 ± 0.000 pp |
| transport anchor only | 0.354 ± 0.021 | 16.050 ± 6.219% | 67.405 ± 6.549% | 0.412 ± 0.121 | 7.898 ± 1.853 pp |
| mediation aware recourse | 0.353 ± 0.020 | 16.088 ± 6.260% | 67.849 ± 6.562% | 0.411 ± 0.121 | 7.936 ± 1.889 pp |

Validation AUC: 0.769 ± 0.005; pure NIE: +0.213 ± 0.017; TE: +0.277 ± 0.009.

### Random Forest

| Method | Post disparity | Factual closure | Feasible | Cost | Closure gain vs baseline |
|---|---:|---:|---:|---:|---:|
| ordinary actionable recourse | 0.379 ± 0.013 | 10.982 ± 2.477% | 66.300 ± 8.203% | 0.125 ± 0.045 | 0.000 ± 0.000 pp |
| transport anchor only | 0.331 ± 0.017 | 22.254 ± 2.532% | 81.945 ± 7.223% | 0.340 ± 0.029 | 11.272 ± 4.187 pp |
| mediation aware recourse | 0.323 ± 0.019 | 24.079 ± 2.849% | 83.517 ± 8.492% | 0.345 ± 0.026 | 13.097 ± 3.915 pp |

Validation AUC: 0.755 ± 0.011; pure NIE: +0.193 ± 0.009; TE: +0.280 ± 0.005.
