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
    parser.add_argument("--save")
    parser.add_argument("--names")
    parser.add_argument("--metric", type=comma_list)
    parser.add_argument("--threshold", type=float, default=0.1)

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    evals = prepare_eval(args)

    metrics = evals[0].columns
    data: Dict[str, Dict[str, int]] = {}
    for metric in metrics:
        data[metric] = {"W": 0, "T": 0, "L": 0}
        df = pd.merge(
            evals[1][metric], evals[0][metric], left_index=True, right_index=True
        )
        data[metric]["W"] = (
            df[f"{metric}_y"] > df[f"{metric}_x"] * (1 + args.threshold)
        ).sum()
        data[metric]["T"] = len(df) - data[metric]["W"] - data[metric]["L"]
        data[metric]["L"] = (
            df[f"{metric}_y"] < df[f"{metric}_x"] * (1 - args.threshold)
        ).sum()

    df = pd.DataFrame.from_dict(data, orient="index")
    df = df[["W", "T", "L"]]
    print(f"Threshold: {args.threshold}")
    print(df.to_string())


if __name__ == "__main__":
    main()
