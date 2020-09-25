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
    parser.add_argument("--no-xticks", action="store_true")
    parser.add_argument("--metric", type=comma_list)
    parser.add_argument("--sort", choices=["ascending", "descending"])
    parser.add_argument("--sample", default="auto")
    parser.add_argument("--avg", action="store_true")
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

    all_qids = set()
    # metric -> sys -> qid -> value
    table: Dict[str, Dict[str, Dict[str, float]]] = {}
    for name, eval_ in zip(names, evals):
        for metric, qid, value in eval_:
            if metric not in args.metric:
                continue
            table.setdefault(metric, {})
            table[metric].setdefault(name, OrderedDict())
            table[metric][name][qid] = value
            all_qids.add(qid)

    sample = -1
    if args.sample == "auto" and len(all_qids) > 100:
        sample = 100
    else:
        sample = int(args.sample)

    if sample > 0:
        qids = np.random.choice(list(all_qids), sample)
        for metric in table.keys():
            for sys in table[metric].keys():
                table[metric][sys] = OrderedDict(
                    [(k, v) for k, v in table[metric][sys].items() if k in qids]
                )

    if args.sort:
        for metric in table.keys():
            reverse = args.sort == "descending"
            base_qid_values = sorted(
                list(table[metric][names[0]].items()),
                key=lambda x: x[1],
                reverse=reverse,
            )
            qid_order = [x[0] for x in base_qid_values]
            for sys in table[metric].keys():
                table[metric][sys] = OrderedDict(
                    [(x, table[metric][sys][x]) for x in qid_order]
                )

    sorted_metrics = sorted(table.keys())
    num_metric = len(sorted_metrics)
    if not args.width:
        args.width = 30
    if not args.height:
        args.height = num_metric * 10
    fig, axes = plt.subplots(num_metric, 2, figsize=(args.width, args.height))
    if axes.ndim == 1:
        axes = axes[None, :]
    # for metric, ax in zip(sorted_metrics, axes[:, 0]):
    #     data = []
    #     for sys in table[metric].keys():
    #         for qid in table[metric][sys].keys():
    #             data.append([sys, qid, table[metric][sys][qid]])
    #     df = pd.DataFrame(data, columns=["Sys", "Qid", "Value"])
    #     sns.lineplot(
    #         x="Qid",
    #         y="Value",
    #         hue="Sys",
    #         style="Sys",
    #         data=df,
    #         sort=False,
    #         ax=ax,
    #         alpha=0.7,
    #     )
    #     ax.set_ylabel(metric)
    #     ax.tick_params(axis="x", rotation=90, labelsize=15)
    #     if args.no_xticks:
    #         ax.set_xticks([])

    palette = sns.color_palette("colorblind")
    for metric, ax_row in zip(sorted_metrics, axes):
        systems = list(table[metric].keys())
        base = systems[0]

        dfs: List[pd.DataFrame] = []
        for idx, (sys, color) in enumerate(zip(systems[1:], palette)):
            data = []
            for qid in table[metric][sys].keys():
                data.append(
                    [sys, qid, table[metric][sys][qid] - table[metric][base][qid]]
                )
            df = pd.DataFrame(data, columns=["Sys", "Qid", "Diff"])
            if idx == 0:
                df = df.sort_values(by=["Sys", "Diff"])
            else:
                df = df.set_index("Qid")
                df = df.loc[dfs[0]["Qid"]].reset_index()
            df.loc[:, "WTL"] = "Tie"
            df.loc[df["Diff"] > 0, "WTL"] = "Win"
            df.loc[df["Diff"] < 0, "WTL"] = "Loss"
            dfs.append(df)
        df = pd.concat(dfs)
        sns.scatterplot(
            x="Qid",
            y="Diff",
            hue="Sys",
            style="Sys",
            data=df,
            # sort=True,
            ax=ax_row[0],
            palette="colorblind",
            alpha=0.7,
        )
        ax_row[0].set_ylabel(f"{metric} diff")
        ax_row[0].set_xticks([])
        ax_row[0].set_xlabel(f"Delta Against {names[0]}")
        ax_row[0].axhline(0, ls="--", color=palette[-1])

        sns.countplot(
            x="WTL",
            hue="Sys",
            data=df,
            ax=ax_row[1],
            palette="colorblind",
            # stat="density",
            # kde=True,
            # bins=20,
            # alpha=0.7,
        )
        ax_row[1].set_xlabel(f"WTL Against {names[0]}")

    if isinstance(args.save, str):
        fig.tight_layout()
        fig.savefig(args.save)


if __name__ == "__main__":
    main()
