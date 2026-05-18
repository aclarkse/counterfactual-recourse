"""
data/bar.py — Law School bar passage dataset loading for causal recourse

SFM role assignment
-------------------
  Sensitive   X : race           (0=Black, 1=White)
  Confounders Z : gender, fam_inc
  Mediators   W_disc : (none)
  Mediators   W_cont : lsat, gpa  (QuantileTransform → N(0,1))
  Outcome     Y : bar_pass
"""

import pandas as pd


# ── SFM config ────────────────────────────────────────────────────────────────

BAR_CFG = {
    "sensitive":      ["race"],
    "confounders":    ["gender", "fam_inc"],
    "mediators_disc": [],
    "mediators_cont": ["lsat", "gpa"],
    "outcome":        "bar_pass",
}

GROUP_LABELS = {0: "Black", 1: "White"}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_bar_data(path: str = "data/bar.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = ["gender", "race1", "lsat", "gpa", "fam_inc", "pass_bar"]
    df = df[cols].dropna(subset=["gender", "fam_inc"]).copy()

    df["gender"] = df["gender"].map({"male": 1, "female": 0})
    df["race"]   = df["race1"].map({"white": 1, "black": 0})
    df = df[df["race"].isin([0, 1])].drop(columns="race1")

    df["fam_inc"] = df["fam_inc"].astype(int)
    df["lsat"]    = df["lsat"].round(0).astype(float)
    df = df.rename(columns={"pass_bar": "bar_pass"})

    print(f"Rows: {len(df):,}")
    print(f"Bar pass rate: {df['bar_pass'].mean():.3f}")
    print(f"Black share  : {(df['race'] == 0).mean():.3f}")
    print(f"White share  : {(df['race'] == 1).mean():.3f}")
    return df.reset_index(drop=True)
