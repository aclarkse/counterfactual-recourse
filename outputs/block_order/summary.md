# Mediator block-order sensitivity (gamma = 0.5, epsilon = 0.1)

Mediator model refitted with each block order on the seed's data split; classifiers, recipients, and references' conditioning are unchanged. Mean ± SD over seeds of the three-classifier average; closure, served, and flip in %. '| S' = among served recipients. 'True' columns evaluate the selected plans under the true SCM (German-Synth only).

## German-Synth

Orders: canonical: {EDUCATION} → {JOB} → {SAVINGS_GRP} → {CREDIT_AMOUNT, DURATION}; swap_job_savings: {EDUCATION} → {SAVINGS_GRP} → {JOB} → {CREDIT_AMOUNT, DURATION}; reversed: {SAVINGS_GRP} → {JOB} → {EDUCATION} → {CREDIT_AMOUNT, DURATION}; joint: {EDUCATION, JOB, SAVINGS_GRP} → {CREDIT_AMOUNT, DURATION}

| Order | NIE | Ord. closure | Ours closure | Ours served | Ours closure \| S | Ours cost \| S | Gain | Ord. closure true | Ours closure true | Gain true |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| canonical | 0.186 ± 0.007 | 74.7 ± 0.9 | 91.5 ± 0.5 | 99 ± 0 | 92.3 ± 0.3 | 0.75 ± 0.01 | 0.086 ± 0.004 | 74.1 ± 1.5 | 87.3 ± 0.7 | 0.068 ± 0.008 |
| swap_job_savings | 0.187 ± 0.007 | 76.8 ± 0.7 | 90.8 ± 1.2 | 97 ± 1 | 92.4 ± 0.4 | 0.78 ± 0.02 | 0.072 ± 0.006 | 65.3 ± 1.3 | 78.6 ± 1.5 | 0.068 ± 0.004 |
| reversed | 0.189 ± 0.006 | 77.2 ± 1.0 | 91.4 ± 0.7 | 98 ± 1 | 92.7 ± 0.5 | 0.75 ± 0.01 | 0.072 ± 0.006 | 62.9 ± 1.0 | 75.2 ± 1.8 | 0.063 ± 0.009 |
| joint | 0.186 ± 0.007 | 74.6 ± 0.9 | 90.4 ± 0.7 | 97 ± 1 | 92.7 ± 0.3 | 0.88 ± 0.02 | 0.081 ± 0.005 | 74.1 ± 1.5 | 86.1 ± 0.8 | 0.062 ± 0.008 |

## ACS Income

Orders: canonical: {SCHL_GRP} → {OCCP_GRP} → {WKHP, WKWN}; swap_edu_occ: {OCCP_GRP} → {SCHL_GRP} → {WKHP, WKWN}; joint: {SCHL_GRP, OCCP_GRP} → {WKHP, WKWN}

| Order | NIE | Ord. closure | Ours closure | Ours served | Ours closure \| S | Ours cost \| S | Gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| canonical | 0.016 ± 0.013 | -0.8 ± 3.9 | 36.1 ± 5.9 | 27 ± 4 | 88.1 ± 0.8 | 0.98 ± 0.17 | 0.078 ± 0.013 |
| swap_edu_occ | 0.013 ± 0.017 | -24.7 ± 5.1 | 48.1 ± 6.3 | 38 ± 4 | 86.7 ± 1.2 | 1.23 ± 0.09 | 0.153 ± 0.014 |
| joint | 0.016 ± 0.013 | 3.7 ± 4.5 | 35.2 ± 6.2 | 26 ± 4 | 89.8 ± 0.5 | 0.97 ± 0.18 | 0.067 ± 0.013 |

## Adult

Orders: canonical: {EDUCATION_GRP} → {OCCUPATION_GRP} → {HOURS_GRP}; swap_edu_occ: {OCCUPATION_GRP} → {EDUCATION_GRP} → {HOURS_GRP}; reversed: {HOURS_GRP} → {OCCUPATION_GRP} → {EDUCATION_GRP}; joint: {EDUCATION_GRP, OCCUPATION_GRP, HOURS_GRP}

| Order | NIE | Ord. closure | Ours closure | Ours served | Ours closure \| S | Ours cost \| S | Gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| canonical | 0.027 ± 0.002 | -210.3 ± 17.9 | 0.0 ± 0.0 | 0 ± 0 | -- | -- | 0.162 ± 0.016 |
| swap_edu_occ | 0.027 ± 0.002 | -203.9 ± 23.9 | 0.0 ± 0.0 | 0 ± 0 | -- | -- | 0.157 ± 0.017 |
| reversed | 0.026 ± 0.003 | -190.8 ± 9.5 | 0.0 ± 0.0 | 0 ± 0 | -- | -- | 0.148 ± 0.010 |
| joint | 0.027 ± 0.002 | -215.1 ± 12.3 | 0.0 ± 0.0 | 0 ± 0 | -- | -- | 0.167 ± 0.011 |
