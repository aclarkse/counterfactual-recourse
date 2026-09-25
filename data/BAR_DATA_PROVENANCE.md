# Law School data provenance

`bar.csv` is a local, git-ignored conversion of
`law.school.admissions` from CRAN package `fairml` 0.9.1. The source documents
20,800 records from the 1991 Law School Admission Council survey. The project
loader retains Black and White respondents and uses race, gender, family-income
bracket, LSAT, undergraduate GPA, and first-attempt bar passage.

Source archive downloaded on 2026-09-24:

- `https://cran.r-project.org/src/contrib/fairml_0.9.1.tar.gz`
- archive SHA-256:
  `85309f6b8581c13ad1e5e78f94c1e23054bc8df226cd3cca547a18b7e9c74245`
- `law.school.admissions.rda` SHA-256:
  `d270972df4787b2da3650f01e9176be53cdbcef2fd3032c8045a2a8d029c4308`

Conversion mapping:

- `ugpa` to `gpa`
- factor `bar` (`TRUE`/`FALSE`) to integer `pass_bar` (`1`/`0`)
- all other selected columns retain their source values

The raw CSV remains excluded by `.gitignore`; cite the underlying LSAC survey
and the `fairml` preprocessing in the paper rather than redistributing it
without checking its terms.
