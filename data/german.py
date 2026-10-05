"""UCI Statlog German Credit data loader for mediation-aware recourse.

SFM role assignment
-------------------
Sensitive A      : SEX (0=Female, 1=Male), decoded from personal status/sex
Confounders C    : AGE and one-hot loan PURPOSE
Financial S      : CHECKING_GRP, SAVINGS_GRP, HOUSING_GRP
Repayment R      : CREDIT_AMOUNT, DURATION
Outcome Y        : GOOD_CREDIT (1=good credit risk, 0=bad credit risk)

The symbolic source is used instead of UCI's derived numeric representation so
that category meanings remain auditable. Personal/marital status itself is not
used as a predictor: the source field jointly encodes it with sex and does not
provide comparable marital categories across sexes.

Source: Hofmann (1994), UCI Machine Learning Repository,
https://doi.org/10.24432/C5NC77 (CC BY 4.0).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd


UCI_GERMAN_URL = (
    "https://archive.ics.uci.edu/static/public/144/"
    "statlog%2Bgerman%2Bcredit%2Bdata.zip"
)
UCI_GERMAN_SHA256 = (
    "e12d9d5def6845c0622634a1cd2ab87fa470668c4298f1ec52a4e403376a435b"
)

_RAW_FILE = "german.data"
_COLUMNS = (
    "checking", "duration", "credit_history", "purpose", "credit_amount",
    "savings", "employment", "installment_rate", "personal_status_sex",
    "other_debtors", "residence_since", "property", "age",
    "other_installment_plans", "housing", "existing_credits", "job",
    "dependents", "telephone", "foreign_worker", "credit_risk",
)

_SAVINGS_CODES = {
    "A65": 0,  # unknown / no savings account
    "A61": 1,  # below 100 DM
    "A62": 2,  # 100 to under 500 DM
    "A63": 3,  # 500 to under 1000 DM
    "A64": 4,  # at least 1000 DM
}
_CHECKING_CODES = {
    "A11": 0,  # below 0 DM
    "A12": 1,  # 0 to under 200 DM
    "A13": 2,  # at least 200 DM / salary assignment
    "A14": 3,  # no checking account (nominal, not an ordinal maximum)
}
_HOUSING_CODES = {
    "A151": 0,  # rent
    "A152": 1,  # own
    "A153": 2,  # free
}
_PURPOSE_CODES = {
    "A40": 0, "A41": 1, "A42": 2, "A43": 3, "A44": 4,
    "A45": 5, "A46": 6, "A48": 7, "A49": 8, "A410": 9,
}
PURPOSE_COLUMNS = tuple(f"PURPOSE_{index}" for index in range(10))
_FEMALE_PERSONAL_STATUS_CODES = {"A92", "A95"}


def _download_archive(destination: Path) -> None:
    temporary = destination.with_suffix(".tmp")
    digest = hashlib.sha256()
    with urlopen(UCI_GERMAN_URL, timeout=60) as response, temporary.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            digest.update(chunk)
    if digest.hexdigest() != UCI_GERMAN_SHA256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("UCI German Credit archive checksum verification failed.")
    temporary.replace(destination)


def _ensure_raw_file(cache_dir: str | Path) -> Path:
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    raw_path = cache / _RAW_FILE
    if raw_path.exists():
        return raw_path

    archive = cache / "german-credit.zip"
    if not archive.exists():
        print(f"Downloading UCI German Credit data from {UCI_GERMAN_URL} ...")
        _download_archive(archive)
    elif hashlib.sha256(archive.read_bytes()).hexdigest() != UCI_GERMAN_SHA256:
        raise RuntimeError(
            f"Cached archive has an unexpected checksum: {archive}. "
            "Remove it and rerun to download the official archive."
        )

    with ZipFile(archive) as zipped, zipped.open(_RAW_FILE) as source, raw_path.open("wb") as output:
        while chunk := source.read(1024 * 1024):
            output.write(chunk)
    return raw_path


def load_german_credit(
    random_state: int = 42,
    cache_dir: str = "data/german",
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Load the official 1,000-row symbolic German Credit dataset."""
    raw_path = _ensure_raw_file(cache_dir)
    raw = pd.read_csv(raw_path, sep=r"\s+", names=_COLUMNS)

    purpose = raw["purpose"].map(_PURPOSE_CODES)
    if purpose.isna().any():
        unknown = sorted(raw.loc[purpose.isna(), "purpose"].unique())
        raise ValueError(f"Unknown German Credit purpose codes: {unknown}")
    purpose_dummies = pd.get_dummies(purpose.astype(int)).reindex(
        columns=range(len(PURPOSE_COLUMNS)), fill_value=0
    )
    purpose_dummies.columns = PURPOSE_COLUMNS

    df = pd.DataFrame({
        "SEX": (~raw["personal_status_sex"].isin(
            _FEMALE_PERSONAL_STATUS_CODES
        )).astype(int),
        "AGE": raw["age"].astype(float),
        "CHECKING_GRP": raw["checking"].map(_CHECKING_CODES).astype(int),
        "SAVINGS_GRP": raw["savings"].map(_SAVINGS_CODES).astype(int),
        "HOUSING_GRP": raw["housing"].map(_HOUSING_CODES).astype(int),
        "CREDIT_AMOUNT": raw["credit_amount"].astype(float),
        "DURATION": raw["duration"].astype(float),
        "GOOD_CREDIT": (raw["credit_risk"] == 1).astype(int),
    })
    df = pd.concat(
        [df.iloc[:, :2], purpose_dummies.astype(float), df.iloc[:, 2:]], axis=1
    )

    if max_rows is not None and len(df) > max_rows:
        df = df.sample(max_rows, random_state=random_state).reset_index(drop=True)
        print(f"Subsampled to {len(df):,} rows")

    print(f"Final rows : {len(df):,}")
    print(f"Good-credit rate: {df['GOOD_CREDIT'].mean():.3f}")
    print(f"Female share: {(df['SEX'] == 0).mean():.3f}")
    print(f"Male share  : {(df['SEX'] == 1).mean():.3f}")
    return df


SFM_CONFIG_GERMAN = {
    "sensitive": ["SEX"],
    "confounders": ["AGE", *PURPOSE_COLUMNS],
    "mediators_disc": ["CHECKING_GRP", "SAVINGS_GRP", "HOUSING_GRP"],
    "mediators_cont": ["CREDIT_AMOUNT", "DURATION"],
    "outcome": "GOOD_CREDIT",
}

VOCAB_GERMAN = {
    "CHECKING_GRP": 4,
    "SAVINGS_GRP": 5,
    "HOUSING_GRP": 3,
}

GROUP_LABELS_GERMAN = {0: "Female", 1: "Male"}
