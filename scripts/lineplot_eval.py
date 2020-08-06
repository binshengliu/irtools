#!/usr/bin/env python3
import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from irtools.evalfile import TrecEval
from irtools.seaborn_setup import seaborn_setup


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("evals", nargs="+")
    parser.add_argument("--save")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--names")
    parser.add_argument("--no-xticks", action="store_true")
    parser.add_argument("--metric", nargs="*")
    parser.add_argument("--sort", choices=["ascending", "descending"])
    parser.add_argument("--sample", type=int)

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    seaborn_setup()
    evals = [TrecEval(x) for x in args.evals]
    fig, ax = plt.subplots(1, 1, figsize=(30, 15))
    names = args.names.split(",") if args.names else args.evals
    assert len(names) == len(args.evals)

    table = []
    for name, eval_ in zip(names, evals):
        one = [(name,) + x for x in eval_ if args.metric and x[0] in args.metric]
        df_one = pd.DataFrame(data=one, columns=["Name", "Metric", "Qid", "Value"])
        table.append(df_one)

    if args.sample:
        ids = table[0]["Qid"].drop_duplicates().sample(n=args.sample)
        table = [
            df.groupby(["Name", "Metric"]).apply(
                lambda x: x.set_index("Qid").reindex(ids).reset_index()
            )
            for df in table
        ]

    if args.sort:
        ids = table[0].sort_values("Value", ascending=args.sort == "ascending")["Qid"]
        table = [df.set_index("Qid").reindex(ids).reset_index() for df in table]

    data = pd.concat(table)

    sns.lineplot(
        x="Qid", y="Value", hue="Metric", style="Name", data=data, sort=False, ax=ax
    )
    ax.tick_params(axis="x", rotation=45)
    if args.no_xticks:
        ax.set_xticks([])

    if args.show:
        plt.show()
    if isinstance(args.save, str):
        fig.tight_layout()
        fig.savefig(args.save)


if __name__ == "__main__":
    main()
