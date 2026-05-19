import os
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import seaborn as sns


PALETTE = sns.color_palette("colorblind")
COL_GRP0 = PALETTE[0]
COL_GRP1 = PALETTE[1]
COL_NDE = PALETTE[2]
COL_NIE = PALETTE[3]
COL_REF = "0.55"

ALPHA_KDE = 0.25
ALPHA_FILL = 0.20
LW = 1.6
MS = 4.0

SAVE_DIR = "figures"
SAVE_EXT = "png"
SAVE_DPI = 300

GROUP_LABELS = {0: "Black", 1: "White"}


def set_paper_style() -> None:
    """Apply a clean, publication-ready global plotting style."""
    sns.set_theme(
        style="white",
        context="paper",
        palette="colorblind",
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 120,
            "savefig.dpi": SAVE_DPI,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        },
    )


def ensure_save_dir(save_dir: str = SAVE_DIR) -> None:
    os.makedirs(save_dir, exist_ok=True)


def save_figure(
    fig: plt.Figure,
    filename: str,
    save: bool = True,
    save_dir: str = SAVE_DIR,
    ext: str = SAVE_EXT,
    dpi: int = SAVE_DPI,
) -> Optional[str]:
    """Save a figure with consistent export settings."""
    if not save:
        return None
    ensure_save_dir(save_dir)
    path = os.path.join(save_dir, f"{filename}.{ext}")
    fig.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    return path


def despine_all(axes: Sequence[plt.Axes]) -> None:
    for ax in axes:
        sns.despine(ax=ax)
