#!/usr/bin/env python3
import argparse
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from more_itertools import always_iterable

from .common import prepare_eval


def comma_list(x: str) -> List[str]:
    return x.split(",")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("eval", nargs="+")
    parser.add_argument("--save")
    parser.add_argument("--names")
    parser.add_argument("--no-xticks", action="store_true")
    parser.add_argument("--metric", type=comma_list)
    parser.add_argument(
        "--sort", choices=["ascending", "descending"], default="descending"
    )
    parser.add_argument("--sample", type=float)
    parser.add_argument("--avg", action="store_true")
    parser.add_argument("--width", type=int, default=30)
    parser.add_argument("--height", type=int, default=15)
    parser.add_argument("--palette", default="deep")
    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    dfs = prepare_eval(args)
    sorted_metrics = sorted(dfs[0].columns)

    num_metric = len(sorted_metrics)
    args.height *= num_metric
    fig, axes = plt.subplots(num_metric, 1, figsize=(args.width, args.height))

    base = dfs[0]
    dfs = dfs[1:]
    for x in dfs:
        for metric in sorted_metrics:
            x[metric] = x[metric] - base[metric]

    palette = sns.color_palette(args.palette)
    for metric, ax in zip(sorted_metrics, always_iterable(axes)):
        df = pd.concat(
            dfs, names=["Sys"], keys=[f"{x}-{args.names[0]}" for x in args.names[1:]]
        )
        df.index = df.index.set_names(["Sys", "Qid"])
        df = df.reset_index()
        df = df[df[metric] != 0]
        df.loc[:, "Change"] = "Good"
        df.loc[df[metric] < 0, "Change"] = "Bad"
        if args.sort:
            df = df.sort_values(metric, ascending=args.sort == "ascending")
        df.loc[df[metric] < 0, metric] = -df[metric]

        dfgood = df.loc[df["Change"] == "Good"]
        dfgood = (
            dfgood.sort_values(metric, ascending=args.sort == "ascending")
            .reset_index(drop=True)
            .reset_index()
        )
        dfbad = df.loc[df["Change"] == "Bad"]
        dfbad = (
            dfbad.sort_values(metric, ascending=args.sort == "ascending")
            .reset_index(drop=True)
            .reset_index()
        )
        df = pd.concat([dfgood, dfbad], names=["Change"], keys=["Good", "Bad"])

        sns.lineplot(
            x="index",
            y=metric,
            hue="Change",
            data=df,
            ax=ax,
            palette=args.palette,
            alpha=0.7,
            sort=False,
        )
        ax.fill_between(dfgood["index"], dfgood[metric], alpha=0.5)
        ax.fill_between(dfbad["index"], dfbad[metric], alpha=0.5)
        ax.set_xticks([dfbad["index"].max(), dfgood["index"].max()])
        ax.axhline(0, ls="--", color=palette[-1])
        label_metric = metric.replace("_cut_", "@").upper()
        ax.set_ylabel(fr"$\Delta${label_metric}")

    if isinstance(args.save, str):
        fig.tight_layout()
        fig.savefig(args.save)


if __name__ == "__main__":
    main()
