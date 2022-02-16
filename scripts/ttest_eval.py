#!/usr/bin/env python3
import argparse
import re
from collections import OrderedDict, abc
from typing import Any, Callable, Dict, List, Optional, Set, TextIO, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import statsmodels.stats.api as sms
from scipy import stats
from statsmodels.stats.multicomp import MultiComparison

TParseRet = Tuple[bool, Optional[str], Optional[str], Optional[npt.NDArray[np.float64]]]

TNpFloat = npt.NDArray[np.float64]


def cohen_d(x, y):
    """https://stackoverflow.com/a/33002123/955952"""
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2

    return (np.mean(x) - np.mean(y)) / np.sqrt(
        ((nx - 1) * np.std(x, ddof=1) ** 2 + (ny - 1) * np.std(y, ddof=1) ** 2) / dof
    )


def comma_list(x: str) -> List[str]:
    return x.split(",")


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

    parser.add_argument("--precision", type=int, default=4)
    parser.add_argument("--names", type=comma_list)
    parser.add_argument("--correction", default="bonf", choices=["bonf", "holm"])
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    if not args.names:
        args.names = [f"Sys{i}" for i in range(len(args.evals))]

    if len(args.names) != len(args.evals):
        parser.error("--names and --evals should match.")
    return args


def rbp_parse(line: str) -> TParseRet:
    if line.startswith("#"):
        return True, None, None, None

    match = re.match(
        r"p= *(\d+\.\d+) *q= *(\w+) *d= *(\w+) *rbp= *(\d+\.\d+) *\+(\d+\.\d+)",
        line,
    )
    assert match is not None
    rbp_p = match[1]
    qid = match[2]
    depth = match[3]
    rbp_value = float(match[4])
    rbp_res = float(match[5])
    metric = f"rbp_{rbp_p}@{depth}"
    skip = qid == "all"
    return skip, metric, qid, [rbp_value, rbp_res]


def gdeval_parse(line: str) -> TParseRet:
    if "runid" in line:
        setattr(gdeval_parse, "metric", line.split(",")[2])
        return True, None, None, None

    splits = line.split(",")
    qid = splits[1]
    value = [float(splits[2]), float(splits[3])]
    skip = qid == "amean"
    return skip, getattr(gdeval_parse, "metric", "ndcg"), qid, value


def trec_parse(line: str) -> TParseRet:
    splits = line.split()
    metric, qid, value = splits[0], splits[1], [float(splits[2])]
    skip = qid == "all"
    return skip, metric, qid, value


def format_value(x: TNpFloat, precision: int) -> Any:
    if isinstance(x, float):
        return f"{x:.{precision}f}"
    elif isinstance(x, np.ndarray):
        if x.size == 1:
            return f"{x[0]:.{precision}f}"
        elif x.size == 2:
            return f"{x[0]:.{precision}f} +{x[1]:.{precision}f}"
        else:
            assert False, f"Unsupported format {x}"
    else:
        assert isinstance(x, abc.Sequence)
        return [format_value(cell, precision) for cell in x]


def interval2str(interval: TNpFloat, precision: int) -> str:
    return f"({interval[0]:.{precision}f}, {interval[1]:.{precision}f})"


def confint_mean(data: npt.ArrayLike, alpha: float) -> List[str]:
    cimean = np.array(sms.DescrStatsW(data).tconfint_mean(alpha=alpha))
    return cimean.T


def detect_eval(fp: TextIO) -> Callable[[str], TParseRet]:
    pos = fp.tell()
    line = fp.readline()
    fp.seek(pos)

    if "rbp_eval" in line:
        parse_func = rbp_parse
    elif "runid" in line:
        parse_func = gdeval_parse
    else:
        parse_func = trec_parse
    return parse_func


def main() -> None:
    args = parse_args()
    prec = args.precision

    results: Dict[str, Dict[str, Dict[str, TNpFloat]]] = OrderedDict()
    file_metrics: Dict[str, Set[str]] = OrderedDict()
    parse_func = detect_eval(args.evals[0])
    for eval_ in args.evals:
        for line in eval_:
            skip, metric, qid, value = parse_func(line)
            if skip:
                continue
            assert metric is not None
            assert qid is not None
            assert value is not None
            file_metrics.setdefault(eval_.name, set()).add(metric)
            results.setdefault(metric, OrderedDict())
            results[metric].setdefault(eval_.name, {})
            results[metric][eval_.name][qid] = value

    common_metrics = set.intersection(*list(file_metrics.values()))
    print(f"Common metrics: {sorted(common_metrics)}")
    for filename, metrics in file_metrics.items():
        diff = metrics - common_metrics
        if diff:
            print(f"{filename}: disregard {sorted(diff)}")

    print(f"Confidence level: {1-args.alpha:.0%}")
    for metric in sorted(common_metrics):
        print(f"# {metric}")
        file_results = results[metric]
        union = set.union(*[set(x.keys()) for x in file_results.values()])
        inter = set.intersection(*[set(x.keys()) for x in file_results.values()])
        # if union != inter:
        #     print(f"{metric} discarded ids: {sorted(union - inter)}")
        for filename in file_results.keys():
            for id_ in union - inter:
                file_results[filename].pop(id_, None)
        print(f"inter/union: {len(inter)}/{len(union)}")

        qids = sorted(inter)
        scores = []
        groups = []
        means = []
        for name, qid_scores in zip(args.names, file_results.values()):
            current = np.array([qid_scores[qid] for qid in qids])
            means = current.mean(axis=0, keepdims=False)
            confint = confint_mean(current, args.alpha)
            info = ", ".join(
                [
                    f"{x:.{prec}f} ({y[0]:.{prec}f}, {y[1]:.{prec}f})"
                    for x, y in zip(means, confint)
                ]
            )
            print(f"{name} {metric} confint: {info}")
            scores.extend([qid_scores[qid][0] for qid in qids])
            groups.extend([name for qid in qids])
        # for name, meanvalue in zip(args.names, means):
        #     print(f"{name}: {meanvalue}")
        cmp_result = MultiComparison(scores, groups, np.array(args.names)).allpairtest(
            stats.ttest_rel, method=args.correction
        )

        df = pd.DataFrame(cmp_result[0].data[1:], columns=cmp_result[0].data[0])
        for idx, row in df.iterrows():
            s1 = np.array([x for x, g in zip(scores, groups) if g == row["group1"]])
            s2 = np.array([x for x, g in zip(scores, groups) if g == row["group2"]])
            cohensd = cohen_d(s1, s2)
            df.loc[idx, "cohen's d"] = cohensd

            cimean = sms.DescrStatsW(s1 - s2).tconfint_mean(alpha=args.alpha)
            df.loc[idx, "diff confint"] = interval2str(cimean, prec)

        print(df.to_string(index=False, float_format=lambda x: f"{x:.{prec}f}"))


if __name__ == "__main__":
    main()
