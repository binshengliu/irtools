#!/usr/bin/env python3
import argparse
from itertools import chain
from operator import itemgetter
from typing import Dict, TextIO, Tuple

import pandas as pd
from irtools.eval_run import eval_run
from irtools.wtl import wtl_seq
from numpy import array_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sort run files based on measure.")

    parser.add_argument("--threshold", default=0.1, type=float)

    parser.add_argument("evals", nargs=2, type=argparse.FileType("r"))

    args = parser.parse_args()

    return args


def load_evals(io: TextIO) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    for line in io:
        metric, qid, score = line.split()
        results.setdefault(metric, {}).setdefault(qid, float(score))
    return results


def main() -> None:
    args = parse_args()

    eval0 = load_evals(args.evals[0])
    eval1 = load_evals(args.evals[1])

    common_metrics = eval0.keys() & eval1.keys()

    data = []
    for metric in common_metrics:
        qids = eval0[metric].keys()
        win, tie, loss = wtl_seq(
            [eval0[metric][x] for x in qids], [eval1[metric][x] for x in qids]
        )
        data.append([metric, win, tie, loss])

    df = pd.DataFrame(data, columns=["metric", "win", "tie", "loss"])
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
