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
    parser.add_argument("--min-rel", type=int, default=1)

    return parser.parse_args()


def load_qrels(path: str, session: bool, min_rel: int) -> List[Tuple[str, str, int]]:
    results = []
    with open(path, "r") as f:
        for line in f:
            splits = line.split()
            qid, did, rel = splits[0], splits[2], int(splits[3])
            if rel < min_rel:
                continue
            if session:
                match = re.match(r"([a-zA-Z0-9]+)[^a-zA-Z0-9]", qid)
                assert match is not None
                qid = match[1]
            results.append((qid, did, rel))
    return results


def main() -> None:
    args = parse_arguments()
    qrels = load_qrels(args.qrels, session=args.session, min_rel=args.min_rel)
    df = pd.DataFrame(data=qrels, columns=["Qid", "Did", "Rel"])

    seaborn_setup()

    if not args.width:
        args.width = 20
    if not args.height:
        args.height = 10

    fig, axes = plt.subplots(1, 1, figsize=(args.width, args.height))

    ax = axes
    sns.countplot(
        x="Qid", data=df, ax=ax, palette="deep",  # , hue="Rel"
    )

    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.0f}",
            (p.get_x() + p.get_width() / 2.0, p.get_height() + 0.2),
            ha="center",
            va="center",
            xytext=(0, 10),
            textcoords="offset points",
            # size=20,
        )
    ax.set_ylabel("Count")
    ax.set_xticklabels(ax.get_xticklabels())
    if args.session:
        ax.set_xlabel("Session ID")
    else:
        ax.set_xlabel("Query ID")

    fig.tight_layout()
    fig.savefig(args.save)


if __name__ == "__main__":
    main()
