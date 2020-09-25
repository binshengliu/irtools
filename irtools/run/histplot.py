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
    parser.add_argument("run")
    parser.add_argument("--save")
    parser.add_argument("--width", type=int, default=30)
    parser.add_argument("--height", type=int, default=15)
    parser.add_argument("--palette", default="deep")
    parser.add_argument("--qrels")

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    seaborn_setup()
    df = load_run(args.run)
    if args.qrels:
        qrels = load_qrels(args.qrels)
        df = df.merge(qrels, how="left").fillna(0)
    else:
        df.loc[:, "Rel"] = "All"

    fig, axes = plt.subplots(1, 1, figsize=(args.width, args.height))

    ax = axes
    sns.histplot(
        x="Score",
        data=df,
        hue="Rel",
        palette=args.palette,
        ax=ax,
        kde=True,
        element="step",
        stat="density",
    )

    if isinstance(args.save, str):
        fig.tight_layout()
        fig.savefig(args.save)


if __name__ == "__main__":
    main()
