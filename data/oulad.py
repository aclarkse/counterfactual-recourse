"""OULAD loading for the early-engagement recourse experiment.

SFM role assignment
-------------------
Sensitive A      : NO_DECLARED_DISABILITY (0=declared, 1=not declared)
Confounders Z    : entry demographics, prior study, module, and presentation
Continuous W     : first-60-day VLE engagement summaries
Outcome Y        : Pass/Distinction versus Fail/Withdrawn

The target population contains students who are still registered at day 60
and have used the VLE by then. This makes the estimand explicitly conditional
on eligibility for a day-60 learning-support intervention. The three
engagement summaries form one same-level multivariate block; their coordinate
order is not interpreted as a causal order.

Source: Kuzilek, Hlosta, and Zdrahal (2015), UCI Machine Learning Repository,
https://doi.org/10.24432/C5KK69 (CC BY 4.0).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd


UCI_OULAD_URL = (
    "https://archive.ics.uci.edu/static/public/349/"
    "open%2Buniversity%2Blearning%2Banalytics%2Bdataset.zip"
)
UCI_OULAD_SHA256 = (
    "f2ed1902616c1fe8d2824d872c0b7d2d72be435bf0124d077044fe4be2c6d3e4"
)
_ARCHIVE = "oulad.zip"
_ENGINEERED = "engineered_day60_v2.csv"
_KEYS = ["code_module", "code_presentation", "id_student"]

_AGE_MIDPOINT = {"0-35": 17.5, "35-55": 45.0, "55<=": 60.0}
_EDUCATION = (
    "No Formal quals",
    "Lower Than A Level",
    "A Level or Equivalent",
    "HE Qualification",
    "Post Graduate Qualification",
)
_MODULES = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG")
_REGIONS = (
    "East Anglian Region", "East Midlands Region", "Ireland",
    "London Region", "North Region", "North Western Region", "Scotland",
    "South East Region", "South Region", "South West Region", "Wales",
    "West Midlands Region", "Yorkshire Region",
)

_NUMERIC_Z = (
    "AGE_MIDPOINT", "IMD_MIDPOINT", "NUM_PREV_ATTEMPTS", "STUDIED_CREDITS",
)
_BINARY_Z = (
    "IMD_MISSING", "GENDER_MALE", "TERM_J", "YEAR_2014",
)
_EDUCATION_Z = tuple(f"EDU_{index}" for index in range(len(_EDUCATION)))
_MODULE_Z = tuple(f"MODULE_{name}" for name in _MODULES)
_REGION_Z = tuple(f"REGION_{index}" for index in range(len(_REGIONS)))
Z_COLUMNS_OULAD = list(
    _NUMERIC_Z + _BINARY_Z + _EDUCATION_Z + _MODULE_Z + _REGION_Z
)


def _download_archive(destination: Path) -> None:
    temporary = destination.with_suffix(".tmp")
    digest = hashlib.sha256()
    with urlopen(UCI_OULAD_URL, timeout=120) as response, temporary.open("wb") as out:
        while chunk := response.read(1024 * 1024):
            out.write(chunk)
            digest.update(chunk)
    if digest.hexdigest() != UCI_OULAD_SHA256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("OULAD archive checksum verification failed.")
    temporary.replace(destination)


def _ensure_archive(cache_dir: str | Path) -> Path:
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / _ARCHIVE
    if not archive.exists():
        print(f"Downloading OULAD from {UCI_OULAD_URL} ...")
        _download_archive(archive)
    elif hashlib.sha256(archive.read_bytes()).hexdigest() != UCI_OULAD_SHA256:
        raise RuntimeError(
            f"Cached archive has an unexpected checksum: {archive}. "
            "Remove it and rerun to download the official archive."
        )
    return archive


def _aggregate_day60(archive: Path, cutoff_day: int = 60) -> pd.DataFrame:
    """Build enrollment-level early-engagement features from raw OULAD tables."""
    daily_parts, site_parts = [], []
    with ZipFile(archive) as zipped:
        with zipped.open("studentInfo.csv") as source:
            info = pd.read_csv(source)
        with zipped.open("studentRegistration.csv") as source:
            registration = pd.read_csv(source)
        with zipped.open("studentVle.csv") as source:
            for index, chunk in enumerate(pd.read_csv(source, chunksize=750_000)):
                chunk = chunk[(chunk["date"] >= 0) & (chunk["date"] <= cutoff_day)]
                if chunk.empty:
                    continue
                daily_parts.append(
                    chunk.groupby(_KEYS + ["date"], as_index=False)["sum_click"].sum()
                )
                site_parts.append(chunk[_KEYS + ["id_site"]].drop_duplicates())
                print(f"  aggregated VLE chunk {index + 1}", flush=True)

    daily = pd.concat(daily_parts, ignore_index=True)
    daily = daily.groupby(_KEYS + ["date"], as_index=False)["sum_click"].sum()
    engagement = daily.groupby(_KEYS).agg(
        EARLY_CLICKS=("sum_click", "sum"),
        ACTIVE_DAYS_60=("date", "nunique"),
    ).reset_index()
    sites = pd.concat(site_parts, ignore_index=True).drop_duplicates()
    resources = sites.groupby(_KEYS).size().rename("RESOURCES_60").reset_index()
    engagement = engagement.merge(resources, on=_KEYS, validate="one_to_one")

    registration["date_unregistration"] = pd.to_numeric(
        registration["date_unregistration"], errors="coerce"
    )
    data = info.merge(
        registration[_KEYS + ["date_unregistration"]], on=_KEYS,
        how="left", validate="one_to_one",
    ).merge(engagement, on=_KEYS, how="left", validate="one_to_one")
    for name in ("EARLY_CLICKS", "ACTIVE_DAYS_60", "RESOURCES_60"):
        data[name] = data[name].fillna(0)

    eligible = (
        (data["date_unregistration"].isna()
         | (data["date_unregistration"] > cutoff_day))
        & (data["EARLY_CLICKS"] > 0)
    )
    data = data.loc[eligible].copy()
    data["LOG_CLICKS_60"] = np.log1p(data["EARLY_CLICKS"].astype(float))
    return data.drop(columns=["EARLY_CLICKS", "date_unregistration"])


def _imd_values(values: pd.Series) -> tuple[pd.Series, pd.Series]:
    extracted = pd.to_numeric(
        values.astype(str).str.extract(r"^(\d+)", expand=False), errors="coerce"
    )
    missing = extracted.isna().astype(int)
    # The published bands have width 10; 45 is the midpoint of the central band.
    return (extracted.add(5).fillna(45.0).astype(float), missing)


def _one_hot(values: pd.Series, categories: tuple[str, ...], names) -> pd.DataFrame:
    unknown = sorted(set(values.dropna().unique()) - set(categories))
    if unknown:
        raise ValueError(f"Unexpected OULAD categories: {unknown}")
    categorical = pd.Categorical(values, categories=categories)
    encoded = pd.get_dummies(categorical, dtype=float)
    encoded.columns = list(names)
    return encoded


def load_oulad(
    random_state: int = 42,
    cache_dir: str = "data/oulad",
    cutoff_day: int = 60,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Load the official OULAD archive and construct the day-60 cohort."""
    archive = _ensure_archive(cache_dir)
    cache = Path(cache_dir) / _ENGINEERED
    if cache.exists():
        raw = pd.read_csv(cache)
    else:
        raw = _aggregate_day60(archive, cutoff_day=cutoff_day)
        raw.to_csv(cache, index=False)
    raw = raw.reset_index(drop=True)

    data = pd.DataFrame({
        "NO_DECLARED_DISABILITY": raw["disability"].eq("N").astype(int),
        "AGE_MIDPOINT": raw["age_band"].map(_AGE_MIDPOINT).astype(float),
        "NUM_PREV_ATTEMPTS": raw["num_of_prev_attempts"].astype(float),
        "STUDIED_CREDITS": raw["studied_credits"].astype(float),
        "GENDER_MALE": raw["gender"].eq("M").astype(int),
        "TERM_J": raw["code_presentation"].str.endswith("J").astype(int),
        "YEAR_2014": raw["code_presentation"].str.startswith("2014").astype(int),
        "LOG_CLICKS_60": raw["LOG_CLICKS_60"].astype(float),
        "ACTIVE_DAYS_60": raw["ACTIVE_DAYS_60"].astype(float),
        "RESOURCES_60": raw["RESOURCES_60"].astype(float),
        "SUCCESS": raw["final_result"].isin(["Pass", "Distinction"]).astype(int),
    })
    data["IMD_MIDPOINT"], data["IMD_MISSING"] = _imd_values(raw["imd_band"])
    data = pd.concat([
        data,
        _one_hot(raw["highest_education"], _EDUCATION, _EDUCATION_Z),
        _one_hot(raw["code_module"], _MODULES, _MODULE_Z),
        _one_hot(raw["region"], _REGIONS, _REGION_Z),
    ], axis=1)
    keep = [
        "NO_DECLARED_DISABILITY", *Z_COLUMNS_OULAD,
        "LOG_CLICKS_60", "ACTIVE_DAYS_60", "RESOURCES_60", "SUCCESS",
    ]
    data = data[keep].reset_index(drop=True)
    if max_rows is not None and len(data) > max_rows:
        data = data.sample(max_rows, random_state=random_state).reset_index(drop=True)

    print(f"Day-{cutoff_day} eligible rows: {len(data):,}")
    print(f"Pass/Distinction rate: {data['SUCCESS'].mean():.3f}")
    print(
        "Declared-disability share: "
        f"{(data['NO_DECLARED_DISABILITY'] == 0).mean():.3f}"
    )
    return data


SFM_CONFIG_OULAD = {
    "sensitive": ["NO_DECLARED_DISABILITY"],
    "confounders": Z_COLUMNS_OULAD,
    "mediators_disc": [],
    "mediators_cont": ["LOG_CLICKS_60", "ACTIVE_DAYS_60", "RESOURCES_60"],
    "outcome": "SUCCESS",
}

GROUP_LABELS_OULAD = {0: "Declared disability", 1: "No declared disability"}
