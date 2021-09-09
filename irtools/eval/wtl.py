#!/usr/bin/env python3
import argparse
from typing import Dict, List

import pandas as pd

from .common import prepare_eval


def comma_list(x: str) -> List[str]:
    return x.split(",")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("eval", nargs=2)
    parser.add_argument("--metric", type=comma_list)
    parser.add_argument("--threshold", type=float, default=0.1)

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    evals = prepare_eval(args)

    metrics = evals[0].columns
    data: Dict[str, Dict[str, str]] = {}
    for metric in metrics:
        df = pd.merge(
            evals[0][metric], evals[1][metric], left_index=True, right_index=True
        )
        metricx = f"{metric}_x"
        metricy = f"{metric}_y"

        data.setdefault(metric, {})
        wins = (df[metricy] > df[metricx] * (1 + args.threshold)).sum()
        losses = (df[metricy] < df[metricx] * (1 - args.threshold)).sum()
        ties = len(df) - wins - losses
        data[metric]["Win"] = str(wins)
        data[metric]["Tie"] = str(ties)
        data[metric]["Loss"] = str(losses)

    df = pd.DataFrame.from_dict(data, orient="index")
    print(f"Threshold: {args.threshold}")
    print(df.to_string())


if __name__ == "__main__":
    main()
