#!/usr/bin/env python3
import argparse
import re
from collections import OrderedDict
from typing import Dict, Set, Tuple

import numpy as np
import pandas as pd
from irtools.eprint import eprint
from scipy import stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="""Run student-t test (scipy.stats.ttest_rel) on paired `trec_eval -q` output, or
        one-way ANOVA test (scipy.stats.f_oneway) if there are more than two groups."""
    )

    parser.add_argument(
        "evals",
        metavar="EVALS",
        nargs="*",
        help="Run `trec_eval -q -m METRIC QREL RUN` to get eval output.",
        type=argparse.FileType("r"),
    )

    args = parser.parse_args()
    return args


def rbp_parse(line: str) -> Tuple[str, str, np.array]:
    match = re.match(
        r"p= *(\d+\.\d+) *q= *(\w+) *d= *(\w+) *rbp= *(\d+\.\d+) *\+(\d+\.\d+)", line,
    )
    assert match is not None
    rbp_p = match[1]
    qid = match[2]
    depth = match[3]
    rbp_value = float(match[4])
    rbp_res = float(match[5])
    metric = f"rbp_{rbp_p}@{depth}"
    return metric, qid, np.array([rbp_value, rbp_res])


def trec_parse(line: str) -> Tuple[str, str, np.array]:
    splits = line.split()
    metric, qid, value = splits[0], splits[1], np.array([float(splits[2])])
    return metric, qid, value


def main() -> None:
    args = parse_args()

    filenames = [x.name for x in args.evals]
    results: Dict[str, Dict[str, Dict[str, np.ndarray]]] = OrderedDict()
    file_metrics: Dict[str, Set[str]] = OrderedDict()
    parse_func = trec_parse
    for eval_ in args.evals:
        for line in eval_:
            if line.startswith("#"):
                continue
            if "rbp=" in line:
                parse_func = rbp_parse

            metric, qid, value = parse_func(line)
            file_metrics.setdefault(eval_.name, set()).add(metric)
            results.setdefault(metric, OrderedDict())
            results[metric].setdefault(eval_.name, {})
            results[metric][eval_.name][qid] = value

    common_metrics = set.intersection(*list(file_metrics.values()))
    eprint(f"Common metrics: {sorted(common_metrics)}")
    for filename, metrics in file_metrics.items():
        diff = metrics - common_metrics
        if diff:
            eprint(f"{filename}: disregard {sorted(diff)}")

    agg = {}
    data = []
    for metric in common_metrics:
        file_results = results[metric]
        qids = set.union(*[set(x.keys()) for x in file_results.values()])
        agg[metric] = {file_: x["all"] for file_, x in file_results.items()}
        qids.difference_update({"all"})
        scores = [
            [qid_scores[qid][0] for qid in qids] for qid_scores in file_results.values()
        ]
        if len(args.evals) == 2:
            _, pvalue = stats.ttest_rel(*scores)
        else:
            _, pvalue = stats.f_oneway(*scores)

        data.append((metric, *[np.round(agg[metric][x], 4) for x in filenames], pvalue))

    # Sort by metric names
    data = sorted(data)
    last_column = "p-value" if len(args.evals) == 2 else "p-value(anova)"
    df = pd.DataFrame(data, columns=["Measure", *filenames, last_column])
    print(df.to_string(index=False, float_format=lambda f: "{:.3f}".format(f)))


if __name__ == "__main__":
    main()
