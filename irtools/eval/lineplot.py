#!/usr/bin/env python3
import argparse
from collections import OrderedDict
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from irtools.evalfile import TrecEval
from irtools.seaborn_setup import seaborn_setup

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

    for metric, ax in zip(sorted_metrics, axes):
        if args.sort:
            dfs[0] = dfs[0].sort_values(metric)
            dfs = [x.loc[dfs[0].index, :] for x in dfs]

        df = pd.concat(dfs, names=["Sys"], keys=args.names)
        df.index = df.index.set_names(["Sys", "Qid"])
        df = df.reset_index()
        sns.lineplot(
            x="Qid",
            y=metric,
            hue="Sys",
            style="Sys",
            data=df,
            ax=ax,
            palette=args.palette,
            alpha=0.7,
            sort=False,
        )
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
            tick.set_ha("right")
        if args.no_xticks:
            ax.set_xticks([])

    if isinstance(args.save, str):
        fig.tight_layout()
        fig.savefig(args.save)


if __name__ == "__main__":
    main()
