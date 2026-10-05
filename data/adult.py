"""UCI Adult data loading for the ACS-analogous income experiment.

SFM role assignment
-------------------
Sensitive X      : SEX (0=Female, 1=Male)
Confounders Z    : AGE, NATIVE_US
Discrete W       : EDUCATION_GRP, OCCUPATION_GRP, HOURS_GRP
Outcome Y        : annual income > $50K

The mediator blocks mirror the ACS experiment: education precedes occupation,
which precedes weekly hours. Workclass, marital status, and relationship are
not included as actions. The latter two are especially unsuitable recourse
targets and can be downstream of sex; capital gain/loss are also omitted
because they are sparse accounting outcomes rather than stable actions.

Data source: Becker and Kohavi (1996), UCI Machine Learning Repository,
https://doi.org/10.24432/C5XW20 (CC BY 4.0).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd


UCI_ADULT_URL = "https://archive.ics.uci.edu/static/public/2/adult.zip"
UCI_ADULT_SHA256 = (
    "7537312dd56c2b98035880805ce99e68183a30ee468aa5329d6df0fbb3cc21bb"
)

_RAW_FILES = ("adult.data", "adult.test")
_COLUMNS = (
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "income",
)

# Stable codes prevent category IDs from changing across train/validation
# splits. The values are nominal; their numeric order has no interpretation.
_OCCUPATION_CODES = {
    name: code for code, name in enumerate((
        "Adm-clerical", "Armed-Forces", "Craft-repair", "Exec-managerial",
        "Farming-fishing", "Handlers-cleaners", "Machine-op-inspct",
        "Other-service", "Priv-house-serv", "Prof-specialty",
        "Protective-serv", "Sales", "Tech-support", "Transport-moving",
    ))
}


def _bucket_education(education_num: pd.Series) -> pd.Series:
    """Map Adult's 16 education levels to the six ACS education groups."""
    bins = [0, 8, 9, 12, 13, 14, 16]
    labels = [0, 1, 2, 3, 4, 5]
    return pd.cut(education_num, bins=bins, labels=labels).astype(int)


def _bucket_hours(hours: pd.Series) -> pd.Series:
    """Ordered bins that preserve Adult's large point mass at 40 hours."""
    clipped = hours.clip(1, 60)
    # 1–19, 20–29, 30–34, 35–39, 40, 41–49, 50–59, 60+
    bins = [0, 19, 29, 34, 39, 40, 49, 59, 60]
    labels = list(range(8))
    return pd.cut(clipped, bins=bins, labels=labels).astype(int)


def _download_archive(destination: Path) -> None:
    temporary = destination.with_suffix(".tmp")
    digest = hashlib.sha256()
    with urlopen(UCI_ADULT_URL, timeout=60) as response, temporary.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            digest.update(chunk)
    if digest.hexdigest() != UCI_ADULT_SHA256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("UCI Adult archive checksum verification failed.")
    temporary.replace(destination)


def _ensure_raw_files(cache_dir: str | Path) -> list[Path]:
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    paths = [cache / name for name in _RAW_FILES]
    if all(path.exists() for path in paths):
        return paths

    archive = cache / "adult.zip"
    if not archive.exists():
        print(f"Downloading UCI Adult data from {UCI_ADULT_URL} ...")
        _download_archive(archive)
    elif hashlib.sha256(archive.read_bytes()).hexdigest() != UCI_ADULT_SHA256:
        raise RuntimeError(
            f"Cached archive has an unexpected checksum: {archive}. "
            "Remove it and rerun to download the official archive."
        )

    with ZipFile(archive) as zipped:
        for name, path in zip(_RAW_FILES, paths):
            if not path.exists():
                with zipped.open(name) as source, path.open("wb") as output:
                    while chunk := source.read(1024 * 1024):
                        output.write(chunk)
    return paths


def _read_raw(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        names=_COLUMNS,
        skipinitialspace=True,
        na_values="?",
        comment="|",  # removes the metadata line at the top of adult.test
    )


def load_adult_income(
    random_state: int = 42,
    cache_dir: str = "data/adult",
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Load and clean the official UCI Adult train and test files.

    Returns columns in SFM order. The canonical UCI split is combined before
    the pipeline creates its seed-specific 80/20 split, matching the ACS
    experiment's repeated-refit design.
    """
    train_path, test_path = _ensure_raw_files(cache_dir)
    df = pd.concat(
        [_read_raw(train_path), _read_raw(test_path)], ignore_index=True
    )
    print(f"Raw rows: {len(df):,}")

    required = [
        "age", "education-num", "occupation", "sex", "hours-per-week",
        "native-country", "income",
    ]
    df = df.dropna(subset=required).copy()

    df["SEX"] = df["sex"].eq("Male").astype(int)
    df["AGE"] = df["age"].astype(float)
    df["NATIVE_US"] = df["native-country"].eq("United-States").astype(int)
    df["EDUCATION_GRP"] = _bucket_education(df["education-num"].astype(int))
    df["OCCUPATION_GRP"] = df["occupation"].map(_OCCUPATION_CODES)
    if df["OCCUPATION_GRP"].isna().any():
        unknown = sorted(
            df.loc[df["OCCUPATION_GRP"].isna(), "occupation"].unique()
        )
        raise ValueError(f"Unrecognized Adult occupation categories: {unknown}")
    df["OCCUPATION_GRP"] = df["OCCUPATION_GRP"].astype(int)
    df["HOURS_GRP"] = _bucket_hours(df["hours-per-week"])
    normalized_income = df["income"].astype(str).str.rstrip(".")
    df["income"] = normalized_income.eq(">50K").astype(int)

    keep = [
        "SEX", "AGE", "NATIVE_US", "EDUCATION_GRP", "OCCUPATION_GRP",
        "HOURS_GRP", "income",
    ]
    df = df[keep].reset_index(drop=True)
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(max_rows, random_state=random_state).reset_index(drop=True)
        print(f"Subsampled to {len(df):,} rows")

    print(f"Final rows : {len(df):,}")
    print(f"Income >$50k: {df['income'].mean():.3f}")
    print(f"Female share: {(df['SEX'] == 0).mean():.3f}")
    print(f"Male share  : {(df['SEX'] == 1).mean():.3f}")
    return df


SFM_CONFIG_ADULT = {
    "sensitive": ["SEX"],
    "confounders": ["AGE", "NATIVE_US"],
    "mediators_disc": ["EDUCATION_GRP", "OCCUPATION_GRP", "HOURS_GRP"],
    "mediators_cont": [],
    "outcome": "income",
}

VOCAB_ADULT = {
    "EDUCATION_GRP": 6,
    "OCCUPATION_GRP": len(_OCCUPATION_CODES),
    "HOURS_GRP": 8,
}

GROUP_LABELS_ADULT = {0: "Female", 1: "Male"}
