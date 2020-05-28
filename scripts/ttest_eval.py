#!/usr/bin/env python3
import argparse
from collections import OrderedDict
from typing import Dict

import pandas as pd
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


def main() -> None:
    args = parse_args()

    filenames = [x.name for x in args.evals]
    results: Dict[str, Dict[str, Dict[str, float]]] = OrderedDict()
    for eval_ in args.evals:
        for line in eval_:
            metric, qid, value = line.split()
            results.setdefault(metric, OrderedDict())
            results[metric].setdefault(eval_.name, {})
            results[metric][eval_.name][qid] = float(value)

    agg = {}
    data = []
    for metric, file_results in results.items():
        qids = set.union(*[set(x.keys()) for x in file_results.values()])
        agg[metric] = {file_: x["all"] for file_, x in file_results.items()}
        qids.difference_update({"all"})
        scores = [
            [qid_scores[qid] for qid in qids] for qid_scores in file_results.values()
        ]
        if len(args.evals) == 2:
            _, pvalue = stats.ttest_rel(*scores)
        else:
            _, pvalue = stats.f_oneway(*scores)

        data.append((metric, *[agg[metric][x] for x in filenames], pvalue))

    last_column = "p-value" if len(args.evals) == 2 else "p-value(anova)"
    df = pd.DataFrame(data, columns=["Measure", *filenames, last_column])
    print(df.to_latex(index=False, float_format=lambda f: "{:.3f}".format(f)))


if __name__ == "__main__":
    main()
