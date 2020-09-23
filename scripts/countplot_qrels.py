#!/usr/bin/env python3
import argparse
import re
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from irtools.seaborn_setup import seaborn_setup


def comma_list(x: str) -> List[str]:
    return x.split(",")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("qrels")
    parser.add_argument("--save", required=True)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--session", action="store_true")

    return parser.parse_args()


def load_qrels(path: str, session: bool) -> List[Tuple[str, str, int]]:
    results = []
    with open(path, "r") as f:
        for line in f:
            splits = line.split()
            qid, did, rel = splits[0], splits[2], int(splits[3])
            if session:
                match = re.match(r"(\w+)\b", qid)
                assert match is not None
                qid = match[1]
            results.append((qid, did, rel))
    return results


def main() -> None:
    args = parse_arguments()
    qrels = load_qrels(args.qrels, session=args.session)
    df = pd.DataFrame(data=qrels, columns=["Qid", "Did", "Rel"])

    seaborn_setup()

    if not args.width:
        args.width = 160
    if not args.height:
        args.height = 20

    fig, axes = plt.subplots(1, 1, figsize=(args.width, args.height))

    ax = axes
    sns.countplot(
        x="Qid", hue="Rel", data=df, ax=ax, palette="colorblind",
    )

    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.0f}",
            (p.get_x() + p.get_width() / 2.0, p.get_height() + 0.05),
            ha="center",
            va="center",
            xytext=(0, 10),
            textcoords="offset points",
            size=15,
        )
    ax.set_xticklabels(ax.get_xticklabels(), size=15)

    fig.tight_layout()
    fig.savefig(args.save)


if __name__ == "__main__":
    main()
