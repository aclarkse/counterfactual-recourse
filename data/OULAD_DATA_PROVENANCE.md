# OULAD data provenance

- Source: Open University Learning Analytics Dataset (OULAD), UCI Machine
  Learning Repository, DOI: https://doi.org/10.24432/C5KK69.
- License: CC BY 4.0.
- Download used by the loader:
  `https://archive.ics.uci.edu/static/public/349/open%2Buniversity%2Blearning%2Banalytics%2Bdataset.zip`
- Verified archive SHA-256:
  `f2ed1902616c1fe8d2824d872c0b7d2d72be435bf0124d077044fe4be2c6d3e4`.

## Analysis cohort

Each row is a student-module-presentation enrollment. The experiment retains
students who were still registered after day 60 (or never unregistered) and
had at least one VLE interaction during days 0--60. Results therefore concern
students eligible for an engagement intervention at that landmark, not every
original OULAD enrollment.

The sensitive contrast is declared disability versus no declared disability.
Baseline adjustment includes age band, deprivation band, gender, prior module
attempts, concurrent studied credits, region, module, education on entry, and
presentation timing. The joint continuous mediator block contains log total
clicks, number of active days, and number of distinct resources accessed during
days 0--60. The favorable outcome is Pass or Distinction; Fail and Withdrawn
are unfavorable.

The observational data cannot rule out unmeasured mediator-outcome
confounding, and disability declaration may be measured with error. Estimated
mediation effects must therefore be described as conditional on the stated SFM
assumptions and on the day-60 target population.
