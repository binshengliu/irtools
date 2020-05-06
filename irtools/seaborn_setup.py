import matplotlib.pyplot as plt
import seaborn as sns


def seaborn_setup():
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
        "font.serif": ["Linux Libertine O"],
        "font.sans-serif": ["Computer Modern"],
        "axes.labelpad": 10.0,
        "xtick.major.pad": 10.0,
        "ytick.major.pad": 10.0,
        "lines.solid_capstyle": "butt",
        "savefig.edgecolor": "#000000",
        "savefig.facecolor": "#ffffff",
    }

    plt.rcParams.update(fmt)
    sns.set_style("darkgrid")
    plt.rcParams.update({"font.sans-serif": ["Linux Biolinum O"]})


def rotate_labels(ax, which, rotation):
    if which == "x":
        ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation)
    elif which == "y":
        ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation)
    else:
        raise ValueError("Unknown axis")
