import argparse
import itertools as it
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statannotations.Annotator import Annotator


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("evals", nargs="+")
    parser.add_argument("--save")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--sep", default=" ")

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    sns.set_theme("notebook", "darkgrid", font="Linux Biolinum O")
    dfs = [pd.read_csv(x, sep=args.sep, names=["Metric", "Qid", "Value"]) for x in args.evals]

    keys = [Path(x).stem for x in args.evals]
    df = pd.concat(dfs, names=["Sys"], keys=keys).reset_index()

    g = sns.catplot(kind="box", x="Sys", y="Value", row="Metric", data=df)
    g.set_titles("")
    for ax, row_name in zip(g.axes.flatten(), g.row_names):
        ax.set_ylabel(row_name)

    pairs = list(it.combinations(keys, 2))
    annotator = Annotator(g.ax, pairs, data=df, x="Sys", y="Value")
    annotator.configure(
        test="t-test_paired",
        text_format="full",
        pvalue_format_string="{:.3f}",
        loc="outside",
    )
    annotator.apply_and_annotate()

    if args.show:
        plt.show()
    if isinstance(args.save, str):
        g.tight_layout()
        g.figure.savefig(args.save)


if __name__ == "__main__":
    main()
