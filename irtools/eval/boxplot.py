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
    parser.add_argument("--no-xticks", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    seaborn_setup()
    evals = [TrecEval(x) for x in args.evals]
    fig, ax = plt.subplots(1, 1, figsize=(30, 15))

    table = []
    for eval_ in evals:
        table.extend(list(eval_))
    data = pd.DataFrame(data=table, columns=["Metric", "Qid", "Value"])

    sns.boxplot(x="Qid", y="Value", hue="Metric", data=data, ax=ax)
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
