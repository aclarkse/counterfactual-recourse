import os
from typing import Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib.legend import Legend as _MplLegend
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
    """Apply a clean, publication-ready global plotting style with LaTeX typesetting."""
    sns.set_theme(
        style="white",
        context="paper",
        palette="colorblind",
        rc={
            # LaTeX typesetting
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "text.latex.preamble": r"\usepackage{amsmath}",
            # Font sizes (matched to body text in a JMLR-style paper)
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            # Spines and grid
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            # Legend frame (visible, subtle border like the reference figure)
            "legend.frameon": True,
            "legend.framealpha": 1.0,
            "legend.edgecolor": "0.7",
            "legend.fancybox": False,
            "legend.borderpad": 0.5,
            "legend.labelspacing": 0.4,
            "legend.handlelength": 1.8,
            # Background / export
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


def legend_outside(
    ax_or_fig,
    *,
    title: Optional[str] = None,
    pad: float = 1.02,
    **kwargs,
) -> _MplLegend:
    """Place a legend to the right of the axes (or figure), outside the plot area.

    Parameters
    ----------
    ax_or_fig : matplotlib Axes or Figure
        If an Axes is passed, the legend is anchored to that axes. If a Figure
        is passed, a single figure-level legend is created (useful for shared
        legends across a row of subplots).
    title : str, optional
        Legend title.
    pad : float
        Horizontal offset from the right edge of the axes/figure, in axes/figure
        coordinates. 1.02 leaves a thin gap; increase for more separation.
    **kwargs
        Forwarded to ``legend()`` (e.g. ``handles=``, ``labels=``).
    """
    anchor = (pad, 0.5)
    legend_kwargs = dict(
        loc="center left",
        bbox_to_anchor=anchor,
        borderaxespad=0.0,
        title=title,
    )
    legend_kwargs.update(kwargs)

    if isinstance(ax_or_fig, plt.Figure):
        return ax_or_fig.legend(**legend_kwargs)
    return ax_or_fig.legend(**legend_kwargs)