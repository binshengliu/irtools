#!/usr/bin/env python3
import argparse
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from irtools.seaborn_setup import seaborn_setup


def comma_list(x: str) -> List[str]:
    return x.split(",")


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


def load_qrels(path: str) -> pd.DataFrame:
    data = []
    with open(path, "r") as f:
        for line in f:
            splits = line.split()
            qid, _, did, rel = splits[:4]
            data.append((qid, did, int(rel)))
    df = pd.DataFrame(data=data, columns=["Qid", "Did", "Rel"])
    return df


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("run", nargs="+")
    parser.add_argument("--save")
    parser.add_argument("--names", type=comma_list)
    parser.add_argument("--width", type=int, default=30)
    parser.add_argument("--height", type=int, default=15)
    parser.add_argument("--palette", default="deep")
    parser.add_argument("--qrels")

    args = parser.parse_args()

    if not args.names:
        args.names = [f"Sys{i}" for i in range(len(args.run))]
    return args


def main() -> None:
    args = parse_arguments()
    seaborn_setup()
    dfs = [load_run(x) for x in args.run]
    if args.qrels:
        qrels = load_qrels(args.qrels)
        dfs = [x.merge(qrels, how="left").fillna(0) for x in dfs]
    else:
        for df in dfs:
            df.loc[:, "Rel"] = "All"

    fig, axes = plt.subplots(1, 1, figsize=(args.width, args.height))

    ax = axes
    uniq_rel = sorted(dfs[0]["Rel"].unique())
    df = pd.concat(dfs, names=["Sys"], keys=args.names)
    for rel in uniq_rel:
        sns.histplot(
            x="Score",
            data=df[df["Rel"] == rel],
            hue="Sys",
            # color=color,
            palette=args.palette,
            ax=ax,
            kde=False,
            element="step",
            stat="density",
            line_kws={"lw": 2},
        )

    if isinstance(args.save, str):
        fig.tight_layout()
        fig.savefig(args.save)


if __name__ == "__main__":
    main()
