# UCI Adult data provenance

The Adult experiment downloads the canonical `adult.data` and `adult.test`
files from the official UCI Machine Learning Repository archive:

- Becker, B. & Kohavi, R. (1996). *Adult* [Dataset]. UCI Machine Learning
  Repository. https://doi.org/10.24432/C5XW20
- License: Creative Commons Attribution 4.0 (CC BY 4.0)
- Archive: `https://archive.ics.uci.edu/static/public/2/adult.zip`
- SHA-256: `7537312dd56c2b98035880805ce99e68183a30ee468aa5329d6df0fbb3cc21bb`

Raw files are cached under `data/adult/` and are excluded from version
control. The loader combines the canonical files, removes rows containing
UCI's `?` missing marker in an analyzed field, then lets the experiment runner
make its own seed-specific 80/20 split so multi-seed uncertainty includes the
data split.

Weekly hours are encoded as eight ordered categories: 1–19, 20–29, 30–34,
35–39, 40, 41–49, 50–59, and 60+. Age remains in its original units in saved
tensors and outcome models; its training-split mean and scale are applied only
within mediator mechanisms.
