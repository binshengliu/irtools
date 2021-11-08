import math
from typing import Union

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


def seaborn_setup_paper() -> None:
    sns.set_theme("paper", "darkgrid", font="Linux Biolinum")


def seaborn_setup_talk() -> None:
    sns.set_theme("talk", "darkgrid", font="Linux Biolinum")


def seaborn_setup_poster() -> None:
    sns.set_theme("poster", "darkgrid", font="Linux Biolinum")


def seaborn_setup() -> None:
    fmt = {
        "backend": "Cairo",
        "text.usetex": False,
        "ps.fonttype": 42,
        "pdf.fonttype": 42,
        "lines.linewidth": 5.0,
        "lines.markersize": 15.0,
        "patch.linewidth": 0.5,
        "legend.fancybox": False,
        "axes.grid": True,
        "patch.edgecolor": "#f0f0f0",
        "axes.titlesize": "x-large",
        "figure.facecolor": "#f0f0f0",
        "grid.linestyle": "-",
        "grid.linewidth": 0.8,
        "grid.color": "#cbcbcb",
        "axes.edgecolor": "#000000",
        "xtick.major.size": 10,
        "xtick.minor.size": 0,
        "ytick.major.size": 10,
        "ytick.minor.size": 0,
        "axes.linewidth": 1.0,
        "legend.handletextpad": 0.1,
        "legend.handlelength": 0.3,
        "legend.columnspacing": 0.1,
        "font.size": 44,
        "font.family": ["serif"],
        "font.serif": ["Linux Libertine"],
        "font.sans-serif": ["Computer Modern"],
        "axes.labelpad": 10.0,
        "xtick.major.pad": 10.0,
        "ytick.major.pad": 10.0,
        "lines.solid_capstyle": "butt",
        "savefig.edgecolor": "#000000",
        "savefig.facecolor": "#ffffff",
        "mathtext.default": "regular",
    }

    plt.rcParams.update(fmt)
    sns.set_style("darkgrid")
    plt.rcParams.update({"font.sans-serif": ["Linux Biolinum"]})


def rotate_labels(ax: matplotlib.axes.Axes, which: str, rotation: float) -> None:
    if which == "x":
        ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation)
    elif which == "y":
        ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation)
    else:
        raise ValueError("Unknown axis")


def annotate_bars(ax: matplotlib.axes.Axes, size: Union[str, int] = "small") -> None:
    if not ax.patches:
        return
    heights = [p.get_height() for p in ax.patches]
    heights = [0 if math.isnan(x) else x for x in heights]
    total = sum(heights)
    for p in ax.patches:
        text = "{} ({:.1%})".format(p.get_height(), p.get_height() / total)
        ax.annotate(
            text,
            (p.get_x() + p.get_width() / 2.0, p.get_height() + 0.05),
            ha="center",
            va="center",
            xytext=(0, 10),
            textcoords="offset points",
            size=size,
        )
