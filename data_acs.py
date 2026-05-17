"""
data_acs.py — ACSIncome (Folktables) data loading for causal recourse

SFM role assignment
-------------------
Sensitive  X : SEX               (0=Female, 1=Male)
Confounders Z: AGEP, POBP_US     (age; US-born vs not — pre-treatment, not actionable)
Mediators disc W_disc: SCHL_GRP, OCCP_GRP
Mediators cont W_cont: WKHP      (clipped to [1, 60] h/week; QuantileTransform → N(0,1))
Outcome    Y : PINCP > 50k

WKHP is kept continuous and passed through a QuantileTransformer so the flow
operates in a smooth N(0,1) space. After sampling, values are inverted back to
hours, clipped to [1, 60], and rounded to the nearest integer. The [1, 60]
ceiling removes implausibly long work-weeks from counterfactual recommendations.

Excluded from mediators
-----------------------
- COW (class of worker): weaker mediator of gender→income, less actionable for recourse
- MAR (marital status): ethically inappropriate as a recourse action; not cleanly on
  the gender→income causal path

Bucketing rationale
-------------------
- SCHL (24 levels) → 6 groups: < HS / HS / Some college / Bachelor's / Master's / Doctoral+
- OCCP (~500 codes) → 10 major SOC groups (Management, STEM, Service, etc.)
- POBP (~100 codes) → binary: US-born (1) vs foreign-born (0)
- RELP (17 levels)  → dropped: partly determined by SEX, creates collider risk
- RAC1P             → dropped: not used as sensitive attribute in this experiment
"""

import numpy as np
import pandas as pd
from folktables import ACSDataSource


# ── Education bucketing (SCHL: 1–24) ─────────────────────────────────────────
#  0: No HS diploma  (SCHL 1–15)
#  1: HS diploma     (SCHL 16–17)
#  2: Some college   (SCHL 18–20)
#  3: Bachelor's     (SCHL 21)
#  4: Master's       (SCHL 22–23)
#  5: Doctoral+      (SCHL 24)

def _bucket_schl(s: pd.Series) -> pd.Series:
    bins  = [0, 15, 17, 20, 21, 23, 24]
    labels = [0, 1, 2, 3, 4, 5]
    return pd.cut(s, bins=bins, labels=labels).astype(int)


# ── Occupation bucketing (OCCP: ACS codes → 10 SOC groups) ───────────────────
# Based on Standard Occupational Classification major groups
def _bucket_occp(s: pd.Series) -> pd.Series:
    def _map(code):
        if   10   <= code <= 430:   return 0  # Management
        elif 500  <= code <= 740:   return 1  # Business & Finance
        elif 800  <= code <= 960:   return 2  # STEM
        elif 1000 <= code <= 1240:  return 3  # STEM support
        elif 1300 <= code <= 1560:  return 4  # Arts, Media, Sports
        elif 1600 <= code <= 1980:  return 5  # Healthcare
        elif 2000 <= code <= 2060:  return 6  # Service
        elif 2100 <= code <= 2920:  return 6  # Service (cont.)
        elif 3000 <= code <= 3550:  return 7  # Sales & Office
        elif 3600 <= code <= 4160:  return 7  # Admin support
        elif 4200 <= code <= 4650:  return 8  # Construction & Production
        elif 4700 <= code <= 5940:  return 8  # Production (cont.)
        elif 6000 <= code <= 6130:  return 9  # Transport
        elif 6200 <= code <= 9750:  return 9  # Other / Military
        else:                       return 6  # catch-all → Service
    return s.map(_map).astype(int)


def load_acs_income(
    year: int = 2018,
    states: list[str] = None,   # None = all 50 states + DC (pooled)
    survey: str = "person",
    threshold: float = 50_000,
    random_state: int = 42,
    cache_dir: str = "data/acs",
    max_rows: int = None,        # optional subsample for fast iteration
) -> pd.DataFrame:
    """
    Download (or load from cache) ACS PUMS data and return a cleaned DataFrame
    with SFM roles already encoded and ready for build_tensors().

    Parameters
    ----------
    year         : ACS survey year (2018–2022)
    states       : list of state abbreviations, or None for all states (pooled)
    threshold    : income binarization threshold (default $50k)
    max_rows     : if set, randomly subsample to this many rows (useful during dev)

    Returns
    -------
    df : DataFrame with columns:
         SEX, AGEP, POBP_US, SCHL_GRP, OCCP_GRP, WKHP, income
    """
    # ── Download ──────────────────────────────────────────────────────────────
    data_source = ACSDataSource(
        survey_year=str(year),
        horizon="1-Year",
        survey=survey,
        root_dir=cache_dir,
    )

    if states is None:
         states=['CA']

    print(f"Downloading ACS {year} data for {len(states)} states...")
    acs_data = data_source.get_data(states=states, download=True)

    # Select only the columns we need directly from the raw PUMS data,
    # avoiding ACSIncome.df_to_numpy which may fail if the cached file was
    # written with a different folktables version (e.g., missing RELP).
    _COLS = ["AGEP", "SCHL", "OCCP", "POBP", "WKHP", "SEX", "PINCP", "PWGTP"]
    df = acs_data[[c for c in _COLS if c in acs_data.columns]].copy()

    # Apply the same filter as folktables' adult_filter
    df = df[
        (df["AGEP"] > 16) &
        (df["PINCP"] > 100) &
        (df["WKHP"] > 0) &
        (df["PWGTP"] >= 1)
    ]

    print(f"Raw rows: {len(df):,}")

    # ── Drop missing ──────────────────────────────────────────────────────────
    df = df.dropna(subset=["WKHP", "SCHL", "OCCP", "AGEP", "SEX", "POBP"])

    # ── Encode sensitive attribute ────────────────────────────────────────────
    # SEX: 1=Male → 1,  2=Female → 0
    df["SEX"] = (df["SEX"] == 1).astype(int)

    # ── Confounder: place of birth → US-born binary ───────────────────────────
    # POBP codes 1–56 are US states/territories; >56 are foreign countries
    df["POBP_US"] = (df["POBP"] <= 56).astype(int)

    # ── Mediators: bucketing + WKHP clip ─────────────────────────────────────
    df["SCHL_GRP"] = _bucket_schl(df["SCHL"].astype(int))
    df["OCCP_GRP"] = _bucket_occp(df["OCCP"].astype(int))
    df["WKHP"]     = df["WKHP"].clip(upper=60).astype(float)

    # ── Outcome ───────────────────────────────────────────────────────────────
    df["income"] = (df["PINCP"] > threshold).astype(int)

    # ── Select final columns in SFM order ────────────────────────────────────
    keep = [
        "SEX",        # X  (sensitive)
        "AGEP",       # Z  (confounder, continuous)
        "POBP_US",    # Z  (confounder, binary)
        "SCHL_GRP",   # W_disc
        "OCCP_GRP",   # W_disc
        "WKHP",       # W_cont  (clipped to [1, 60])
        "income",     # Y
    ]
    df = df[keep].reset_index(drop=True)

    # ── Optional subsample ────────────────────────────────────────────────────
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(max_rows, random_state=random_state).reset_index(drop=True)
        print(f"Subsampled to {len(df):,} rows")

    print(f"Final rows : {len(df):,}")
    print(f"Income >$50k: {df['income'].mean():.3f}")
    print(f"Female share: {(df['SEX'] == 0).mean():.3f}")
    print(f"Male share  : {(df['SEX'] == 1).mean():.3f}")

    return df


# ── SFM config for build_tensors() ───────────────────────────────────────────
SFM_CONFIG_ACS = {
    "sensitive":      ["SEX"],
    "confounders":    ["AGEP", "POBP_US"],
    "mediators_disc": ["SCHL_GRP", "OCCP_GRP"],
    "mediators_cont": ["WKHP"],
    "outcome":        "income",
}

# Vocab sizes for discrete mediators (for reference / unit tests)
VOCAB_ACS = {
    "SCHL_GRP": 6,   # education groups
    "OCCP_GRP": 10,  # occupation groups
}

# Group labels for diagnostic plots
GROUP_LABELS_ACS = {0: "Female", 1: "Male"}