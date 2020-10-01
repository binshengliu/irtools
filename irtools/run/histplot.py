#!/usr/bin/env python3
import argparse
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
    # df.loc[:, "Score"] = df.groupby("Qid")["Score"].apply(lambda x: softmax(x))
    # df.loc[:, "Score"] = df.groupby("Qid")["Score"].apply(
    #     lambda x: (x - x.min()) / (x.max() - x.min())
    # )
    return df


def load_qrels(path: str, min_rel: int) -> pd.DataFrame:
    data = []
    with open(path, "r") as f:
        for line in f:
            splits = line.split()
            qid, _, did, rel = splits[:4]
            if int(rel) < min_rel:
                data.append((qid, did, 0))
            else:
                data.append((qid, did, 1))
    df = pd.DataFrame(data=data, columns=["Qid", "Did", "Rel"])
    return df


def bins_type(bins: str) -> Union[str, int]:
    if bins == "auto":
        return bins
    else:
        return int(bins)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("run", nargs="+")
    parser.add_argument("--save")
    parser.add_argument("--names", type=comma_list)
    parser.add_argument("--width", type=int, default=30)
    parser.add_argument("--height", type=int, default=15)
    parser.add_argument("--palette", default="deep")
    parser.add_argument("--qrels")
    parser.add_argument("--bins", type=bins_type, default="auto")
    parser.add_argument("--min-rel", type=int, default=1)

    args = parser.parse_args()

    if not args.names:
        args.names = [f"Sys{i}" for i in range(len(args.run))]
    return args


def main() -> None:
    args = parse_arguments()
    seaborn_setup()
    dfs = [load_run(x) for x in args.run]
    if args.qrels:
        qrels = load_qrels(args.qrels, args.min_rel)
        dfs = [x.merge(qrels, how="left").fillna(0) for x in dfs]
    else:
        for df in dfs:
            df.loc[:, "Rel"] = "All"

    fig, axes = plt.subplots(1, 1, figsize=(args.width, args.height))

    ax = axes
    uniq_rel = sorted(dfs[0]["Rel"].unique())
    assert uniq_rel == [0, 1]
    df = pd.concat(dfs, names=["Sys"], keys=args.names).reset_index()
    palette = sns.color_palette(args.palette)
    for rel in uniq_rel:
        sns.histplot(
            x="Score",
            data=df[df["Rel"] == rel],
            hue="Sys",
            palette=palette[: len(args.names)],
            ax=ax,
            kde=False,
            element="step",
            stat="density",
            common_norm=False,
            bins=args.bins,
            # alpha=0.9,
        )
        df_fit_all = []
        for sys in args.names:
            mask = (df["Rel"] == rel) & (df["Sys"] == sys)
            scores = df[mask].loc[:, "Score"].to_numpy()
            if rel == 1:
                X, Y, mu, sigma = norm_fit(scores, args.bins)
                df_fit = pd.DataFrame(data={"Score": X, "density": Y})
                df_fit["Sys"] = fr"{sys} Relevant: $\mu$={mu:.2f},$\sigma$={sigma:.2f}"
                print(f"{sys}: mu: {mu:f}, sigma: {sigma:f}")
            else:
                X, Y, lambda_ = exp_fit(scores, args.bins)
                df_fit = pd.DataFrame(data={"Score": X, "density": Y})
                df_fit["Sys"] = fr"{sys} Non-relevant: $\lambda$={lambda_:.2f}"
                print(f"{sys}: lambda: {lambda_:f}")
            df_fit_all.append(df_fit)
        df_fit = pd.concat(df_fit_all)
        sns.lineplot(
            data=df_fit,
            x="Score",
            y="density",
            hue="Sys",
            palette=palette[: len(args.names)],
            ax=ax,
            alpha=0.7,
            linewidth=5,
        )
        palette = palette[len(args.names) :]
    ax.legend()

    if isinstance(args.save, str):
        fig.tight_layout()
        fig.savefig(args.save)


if __name__ == "__main__":
    main()
