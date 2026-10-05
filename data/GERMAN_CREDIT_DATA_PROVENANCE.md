# UCI German Credit data provenance

The German Credit experiment uses the symbolic `german.data` file from the
official UCI Statlog (German Credit Data) archive:

- Hofmann, H. (1994). *Statlog (German Credit Data)* [Dataset]. UCI Machine
  Learning Repository. https://doi.org/10.24432/C5NC77
- License: Creative Commons Attribution 4.0 (CC BY 4.0)
- Archive: `https://archive.ics.uci.edu/static/public/144/statlog%2Bgerman%2Bcredit%2Bdata.zip`
- SHA-256: `e12d9d5def6845c0622634a1cd2ab87fa470668c4298f1ec52a4e403376a435b`

The symbolic file has 1,000 rows and no missing values. The favorable outcome
is UCI class 1 (good credit risk); class 2 is encoded as 0.

UCI's attribute 9 jointly records personal status and sex. The loader follows
the dataset documentation: A92 and A95 are encoded as female (0), while A91,
A93, and A94 are encoded as male (1). Personal/marital status is otherwise
excluded because its categories are not comparable across sexes.

The causal roles follow the German Credit graph used by Chikahara et al.
(2021): sensitive attribute `A` is sex; context `C` contains age and loan
purpose; financial status `S` contains checking-account balance, savings, and
housing ownership; repayment terms `R` contain credit amount and duration; and
`Y` is good credit risk. Purpose is one-hot encoded because its source codes
are nominal. Age is standardized using training-split statistics only inside
the mediator mechanisms.

This implementation uses the compatible all-`S`-paths interpretation (option
1): `A -> Y` and all paths beginning `A -> S` are considered unfair, including
`A -> S -> R -> Y`. Thus it does not attempt to isolate only `A -> S -> Y`
while retaining the path through `R`, which would require stronger cross-world
structural assumptions than the conditional-density implementation supplies.

`S` is modeled as one joint categorical block conditional on `(A,C)`. Its
autoregressive coordinate order is a statistical factorization only. `R` is
modeled as one joint continuous flow conditional on `(S,C)`, deliberately
excluding `A` from that mechanism to encode the stated graph. No causal order
is assumed inside either block. Each experimental seed creates an independent
random split with 900 training rows and 100 test rows.

The primary recourse selector minimizes action cost subject to an 80% factual
success-probability constraint and a prediction-space Wasserstein-1 radius of
0.10 around the recipient-conditional advantaged-mediator reference. The
recipient's `C` is fixed in both distributions, and the classifier is evaluated
at the disadvantaged sensitive value. Infeasible recipients are retained as
explicit abstentions with their factual plan.
