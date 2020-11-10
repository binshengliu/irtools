#!/usr/bin/env python3
import argparse
import sys
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from irtools.seaborn_setup import seaborn_setup
from scipy.optimize import curve_fit

# from scipy.special import softmax


def comma_list(x: str) -> List[str]:
    return x.split(",")


def comma_int_list(x: str) -> List[int]:
    return [int(i) for i in x.split(",")]


def exp_func(x: np.ndarray, a: float, b: float) -> np.ndarray:
    value = a * np.exp(-b * x)
    return value


def exp_fit(scores: np.ndarray, bins: Union[str, int]) -> np.ndarray:
    data, bin_edges = np.histogram(scores, bins=bins, density=True)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    popt, pcov = curve_fit(exp_func, bin_centers, data, (1, 1))
    print(popt)
    print(pcov)

    X = bin_centers
    Y = exp_func(X, *popt)
    return X, Y, popt[1]


def norm_func(x: np.ndarray, B: float, mu: float, sigma: float) -> np.ndarray:
    return B * np.exp(-1.0 * (x - mu) ** 2 / (2 * sigma ** 2))


def norm_fit(scores: np.ndarray, bins: Union[str, int]) -> np.ndarray:
    data, bin_edges = np.histogram(scores, bins=bins, density=True)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    popt, pcov = curve_fit(norm_func, bin_centers, data, (1, 1, 1))
    print(popt)
    print(pcov)

    X = bin_centers
    Y = norm_func(X, *popt)
    return X, Y, popt[1], popt[2]


def load_run(path: str) -> pd.DataFrame:
    data = []
    with open(path, "r") as f:
        for line in f:
            splits = line.split()
            qid, _, did, rank = splits[:4]
            score = float(splits[4])
            data.append((qid, did, score))
    df = pd.DataFrame(data=data, columns=["Qid", "Did", "Score"])
    return df


def load_qrels(path: str, binarize: bool, min_rel: int) -> pd.DataFrame:
    data = []
    with open(path, "r") as f:
        for line in f:
            splits = line.split()
            qid, _, did, rel = splits[:4]
            if binarize:
                if int(rel) < min_rel:
                    data.append((qid, did, 0))
                else:
                    data.append((qid, did, 1))
            else:
                data.append((qid, did, int(rel)))
    df = pd.DataFrame(data=data, columns=["Qid", "Did", "Rel"])
    return df


def bins_type(bins: str) -> Union[str, int]:
    if bins == "auto":
        return bins
    else:
        return int(bins)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-i", "--input", default=sys.stdin, nargs="+")
    parser.add_argument("-o", "--output", default=sys.stdout)
    parser.add_argument("--sep", default="\t")
    parser.add_argument("--names", type=comma_list)
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--height", type=int, default=10)
    parser.add_argument("--palette", default="deep")
    parser.add_argument("--bins", type=bins_type, default="auto")
    parser.add_argument("--annotate", default=[50, 75, 95], type=comma_int_list)
    parser.add_argument("--y-log-scale", action="store_true")
    parser.add_argument("--x-log-scale", action="store_true")
    parser.add_argument("--xlabel", default="Value")
    parser.add_argument(
        "--stat",
        default="density",
        choices=["density", "frequency", "count", "probability"],
    )
    parser.add_argument("--remove-tail", type=int)

    args = parser.parse_args()
    if not args.names:
        args.names = [f"Sys{i}" for i in range(len(args.input))]

    return args


def main() -> None:
    args = parse_arguments()
    seaborn_setup()

    annotation_xy = []
    dfs = []
    for idx, one in enumerate(args.input):
        dftmp = pd.read_csv(one, names=["Value"], sep=args.sep, header=None)

        args.names[idx] = f"{args.names[idx]}"
        if args.remove_tail is not None:
            threshold = np.percentile(dftmp["Value"], args.remove_tail)
            dftmp = dftmp[dftmp["Value"] < threshold]

        height, width = np.histogram(dftmp["Value"], bins=args.bins)
        for q in args.annotate:
            value = np.percentile(dftmp["Value"], q)
            loc = ((width - value) > 0).nonzero()[0][0] - 1
            annotation_xy.append((value, height[loc], q))

        dfs.append(dftmp)
    df = pd.concat(dfs, names=["Sys"], keys=args.names)
    fig, axes = plt.subplots(1, 1, figsize=(args.width, args.height))
    ax = axes
    sns.histplot(
        x="Value",
        data=df,
        hue="Sys",
        hue_order=args.names,
        palette=args.palette,
        ax=ax,
        kde=False,
        element="step",
        stat=args.stat,
        common_norm=False,
        bins=args.bins,
    )
    if args.x_log_scale:
        ax.set_xscale("log")
    if args.y_log_scale:
        ax.set_yscale("log")

    for x, y, v in annotation_xy:
        ax.annotate(
            f"{v}%: {x:.0f}",
            (x, y),
            ha="center",
            va="center",
            xytext=(0, 50),
            textcoords="offset points",
            size="xx-small",
            arrowprops=dict(arrowstyle="-|>", color="black", connectionstyle="arc3"),
        )
    ax.get_legend().set_title("")
    ax.set_xlabel(args.xlabel)
    if isinstance(args.output, str):
        fig.tight_layout()
        fig.savefig(args.output)


if __name__ == "__main__":
    main()
