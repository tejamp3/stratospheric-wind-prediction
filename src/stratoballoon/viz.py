"""Shared plotting style.

One palette, one rc block, so every figure in results/ reads as one system.
Colours are the validated categorical slots 1-3 (blue / orange / aqua): worst
all-pairs CVD Delta E 9.2, normal-vision 24.0 on the light surface. Aqua sits at
2.74:1 contrast, below the 3:1 bar, so every figure carries a legend or direct
label and the numbers are also written to CSV under results/ - identity is
never colour-alone.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a897f"
GRID = "#e6e5e0"

# Categorical slots, fixed order, never cycled.
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
C4, C5 = "#eda100", "#e87ba4"
SERIES = [C1, C2, C3, C4, C5]


# Semantics used repeatedly across figures.
ACTUAL, PREDICTED, BASELINE = INK2, C1, C2

# Sequential blue ramp (100 -> 700) for magnitude.
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
            "#0d366b"]
# Diverging blue <-> red with a neutral gray midpoint.
DIV = ["#104281", "#256abf", "#5598e7", "#9ec5f4", "#f0efec",
       "#f3a3a2", "#e87977", "#e34948", "#b52d2c"]


def seq_cmap():
    return mpl.colors.LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)


def div_cmap():
    return mpl.colors.LinearSegmentedColormap.from_list("div_bluered", DIV)


def apply_style() -> None:
    # Matplotlib logs an INFO line every time a categorical axis is used, which
    # buries the output of every script that draws a bar chart.
    import logging
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.bbox": "tight",
        "savefig.dpi": 150,
        "figure.dpi": 110,
        "font.size": 10,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "text.color": INK,
        "axes.labelcolor": INK2,
        "axes.edgecolor": GRID,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.labelsize": 9.5,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        # Recessive grid and axes: the data carries the weight.
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 2.0,          # 2px lines
        "lines.markersize": 4.5,
        "lines.solid_capstyle": "round",
        "legend.frameon": False,
        "legend.fontsize": 9,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
    })


def annotate(ax, text: str, loc: str = "upper left") -> None:
    """Small recessive note inside the axes, in ink not series colour."""
    xy = {"upper left": (0.015, 0.97), "upper right": (0.985, 0.97),
          "lower left": (0.015, 0.03), "lower right": (0.985, 0.03)}[loc]
    ax.text(*xy, text, transform=ax.transAxes, fontsize=8.5, color=MUTED,
            ha="right" if "right" in loc else "left",
            va="top" if "upper" in loc else "bottom")


def finish(fig, path, note: str | None = None) -> None:
    """Attach an optional source note and save."""
    if note:
        fig.text(0.005, -0.015, note, fontsize=7.5, color=MUTED, ha="left", va="top")
    fig.savefig(path)
    plt.close(fig)


# Every figure says which kind of evidence it shows, so a simulated result is
# never mistaken for a measured one.
KIND = {"real": "REAL DATA: ERA5 reanalysis",
        "model": "MODEL PREDICTION evaluated against ERA5",
        "sim": "SIMULATION: simplified research model, not flight performance"}


def source_note(kind: str, extra: str = "") -> str:
    return KIND[kind] + (f". {extra}" if extra else "")
