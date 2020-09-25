#!/usr/bin/env python3
import argparse
from typing import Dict, List

import numpy as np
import pandas as pd
from irtools.evalfile import TrecEval
from irtools.seaborn_setup import seaborn_setup


def comma_list(x: str) -> List[str]:
    return x.split(",")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("evals", nargs=2)
    parser.add_argument("--save")
    parser.add_argument("--names")
    parser.add_argument("--metric", type=comma_list)
    parser.add_argument("--threshold", type=float, default=0.1)

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    seaborn_setup()
    np.random.seed(2)
    evals = [TrecEval(x) for x in args.evals]
    if args.metric is None:
        args.metric = evals[0].metrics()
    names = (
        args.names.split(",") if args.names else [f"Sys{i}" for i in range(len(evals))]
    )
    assert len(names) == len(args.evals)

    # metric -> sys -> qid -> value
    table: Dict[str, Dict[str, Dict[str, float]]] = {}
    for name, eval_ in zip(names, evals):
        for metric, qid, value in eval_:
            if metric not in args.metric:
                continue
            table.setdefault(metric, {})
            table[metric].setdefault(name, {})
            table[metric][name][qid] = value

    sorted_metrics = sorted(table.keys())
    data: Dict[str, Dict[str, int]] = {}
    for metric in sorted_metrics:
        base = names[0]
        target = names[1]

        for qid in table[metric][target].keys():
            if table[metric][target][qid] > table[metric][base][qid] * 1.1:
                data.setdefault(metric, {}).setdefault("W", 0)
                data[metric]["W"] += 1
            elif table[metric][target][qid] < table[metric][base][qid] * 0.9:
                data.setdefault(metric, {}).setdefault("L", 0)
                data[metric]["L"] += 1
            else:
                data.setdefault(metric, {}).setdefault("T", 0)
                data[metric]["T"] += 1

    results = {
        f"WTL ({args.threshold:.0%} threshold)": {
            metric: f"{sys_results['W']}/{sys_results['T']}/{sys_results['L']}"
            for metric, sys_results in data.items()
        }
    }
    df = pd.DataFrame(results)
    print(df.to_string())


if __name__ == "__main__":
    main()
