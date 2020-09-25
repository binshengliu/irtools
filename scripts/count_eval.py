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


def comma_list(x: str) -> List[str]:
    return x.split(",")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("evals", nargs="+")
    parser.add_argument("--save")
    parser.add_argument("--names")
    parser.add_argument("--metric", type=comma_list)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    seaborn_setup()
    np.random.seed(2)
    evals = [TrecEval(x) for x in args.evals]
    if args.metric is None:
        args.metric = evals[0].metrics()
    names = (
        args.names.split(",") if args.names else [f"Sys{i}" for i in range(len(evals))]
    )
    assert len(names) == len(args.evals)

    # metric -> sys -> qid -> value
    table: Dict[str, Dict[str, Dict[str, float]]] = {}
    for name, eval_ in zip(names, evals):
        for metric, qid, value in eval_:
            if metric not in args.metric:
                continue
            table.setdefault(metric, {})
            table[metric].setdefault(name, OrderedDict())
            table[metric][name][qid] = value

    sorted_metrics = sorted(table.keys())
    num_metric = len(sorted_metrics)
    if not args.width:
        args.width = 30
    if not args.height:
        args.height = num_metric * 20
    fig, axes = plt.subplots(num_metric, 1, figsize=(args.width, args.height))
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    for metric, ax in zip(sorted_metrics, axes):
        dfs = []
        for sys in table[metric].keys():
            data = []
            for qid in table[metric][sys].keys():
                data.append([sys, qid, table[metric][sys][qid]])
            df = pd.DataFrame(data, columns=["Sys", "Qid", "Value"])
            dfs.append(df)

        df = pd.concat(dfs)
        if len(pd.unique(df["Value"])) > 11:
            df["Value"] = pd.cut(
                df["Value"], bins=np.linspace(0, 1, 11), include_lowest=True
            )

        sns.countplot(
            x="Value", hue="Sys", data=df, ax=ax, palette="colorblind",
        )
        ax.set_ylabel(f"{metric} count")

    if isinstance(args.save, str):
        fig.tight_layout()
        fig.savefig(args.save)


if __name__ == "__main__":
    main()
