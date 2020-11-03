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
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--height", type=int, default=10)
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

    palette = sns.color_palette(args.palette)
    for metric, ax in zip(sorted_metrics, always_iterable(axes)):
        if args.sort:
            dfs[0] = dfs[0].sort_values(metric)
            dfs = [x.loc[dfs[0].index, :] for x in dfs]

        to_plot = dfs[0]
        to_plot.index = to_plot.index.set_names("Qid")
        to_plot = to_plot.reset_index()
        to_plot.loc[:, "Sys"] = args.names[0]
        sns.lineplot(
            x="Qid",
            y=metric,
            hue="Sys",
            style="Sys",
            data=to_plot,
            ax=ax,
            palette=palette[:1],
            alpha=0.7,
            sort=False,
        )
        df = pd.concat(dfs[1:], names=["Sys"], keys=args.names[1:])
        df.index = df.index.set_names(["Sys", "Qid"])
        df = df.reset_index()
        sns.scatterplot(
            x="Qid",
            y=metric,
            hue="Sys",
            style="Sys",
            data=df,
            ax=ax,
            palette=palette[1:][: len(args.names[1:])],
            alpha=0.7,
        )
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
            tick.set_ha("right")
        if args.no_xticks:
            ax.set_xticks([])
        ax.set_ylabel(metric.replace("_cut_", "@"))
        ax.legend(framealpha=0.7)

    if isinstance(args.save, str):
        fig.tight_layout()
        fig.savefig(args.save)


if __name__ == "__main__":
    main()
