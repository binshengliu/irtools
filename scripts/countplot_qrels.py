#!/usr/bin/env python3
import argparse
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from irtools.seaborn_setup import seaborn_setup


def comma_list(x: str) -> List[str]:
    return x.split(",")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("qrels", nargs="+")
    parser.add_argument("--save", required=True)
    parser.add_argument("--names", type=comma_list)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)

    return parser.parse_args()


def load_qrels(path: str) -> List[Tuple[str, str, int]]:
    results = []
    with open(path, "r") as f:
        for line in f:
            splits = line.split()
            qid, did, rel = splits[0], splits[2], int(splits[3])
            results.append((qid, did, rel))
    return results


def main() -> None:
    args = parse_arguments()
    seaborn_setup()
    np.random.seed(2)
    qrels = [load_qrels(x) for x in args.qrels]
    if not args.names:
        args.names = [f"Sys{i}" for i in range(len(qrels))]
    assert len(args.names) == len(args.qrels)

    dfs = []
    for name, eval_ in zip(args.names, qrels):
        df = pd.DataFrame(data=eval_, columns=["Qid", "Did", "Rel"])
        df.loc[:, "Sys"] = name
        dfs.append(df)

    df = pd.concat(dfs)

    if not args.width:
        args.width = 30
    if not args.height:
        args.height = 20
    fig, axes = plt.subplots(1, 1, figsize=(args.width, args.height))

    ax = axes
    sns.countplot(
        x="Sys", hue="Rel", data=df, ax=ax, palette="colorblind",
    )

    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.0f}",
            (p.get_x() + p.get_width() / 2.0, p.get_height() + 0.05),
            ha="center",
            va="center",
            xytext=(0, 10),
            textcoords="offset points",
            size="small",
        )

    fig.tight_layout()
    fig.savefig(args.save)


if __name__ == "__main__":
    main()
