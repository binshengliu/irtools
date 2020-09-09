#!/usr/bin/env python3
import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
from irtools.log import get_logger
from sklearn.model_selection import KFold

logger = get_logger(__name__)


examples = """Parameter names and values should be indicated in the file name in `name=value`
format and separated by `_` or `-`.

Naming examples:

task-epoch=0.eval or task_epoch=0.eval
task-epoch=1.eval or task_epoch=1.eval
task-epoch=2.eval or task_epoch=2.eval
"""


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross validation for trec_eval and rbp_eval output files.",
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--files", nargs="+")
    parser.add_argument(
        "-o", "--output", type=argparse.FileType("w"), default=sys.stdout
    )
    parser.add_argument("--seed", type=int, default=2)
    return parser.parse_args()


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


def rbp_format(metric: str, qid: str, values: np.ndarray) -> str:
    match = re.match(r"rbp_(\d+\.\d+)@(\w+)", metric)
    assert match is not None
    rbp_p = match[1]
    depth = match[2]
    s = (
        f"p= {rbp_p} q= {qid:>4s} d= {depth:>4s} "
        f"rbp= {values[0]:.4f} +{values[1]:.4f}"
    )
    return s


def trec_format(metric: str, qid: str, values: np.ndarray) -> str:
    return f"{metric:<22}\t{qid}\t{values[0]}"


def main() -> None:
    args = parse_arguments()

    # params -> metric -> qid -> values
    results: Dict[str, Dict[str, Dict[str, np.array]]] = {}
    qids_set: Set[str] = set()
    metrics = set()
    format_func = trec_format
    parse_func = trec_parse
    for one in args.files:
        stem = Path(one).stem
        match = re.findall(r"([^-_]+=[^-_]+)", stem)
        params = "-".join(match)
        with open(one, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue

                if "rbp=" in line:
                    format_func = rbp_format
                    parse_func = rbp_parse

                metric, qid, value = parse_func(line)
                if qid == "all":
                    continue
                metrics.add(metric)
                qids_set.add(qid)
                results.setdefault(params, {})
                results[params].setdefault(metric, {})
                results[params][metric][qid] = value

    metrics = sorted(metrics)  # type: ignore
    qids = sorted(qids_set)
    kfold = KFold(n_splits=5, shuffle=True, random_state=args.seed)

    # metric -> qid -> values
    test_results: Dict[str, Dict[str, np.array]] = {}
    for idx, (train_index, test_index) in enumerate(kfold.split(qids)):
        train_qids = [qids[x] for x in train_index]
        test_qids = [qids[x] for x in test_index]
        for metric in metrics:
            train_results: Dict[str, List[float]] = {}
            for params in results:
                for qid in train_qids:
                    train_results.setdefault(params, [])
                    train_results[params].append(results[params][metric][qid][0])
                train_results[params] = np.mean(train_results[params])
            best_params = max(train_results.items(), key=lambda x: x[1])[0]
            logger.info(f"Best params of fold {idx} metric {metric}: {best_params}")
            test_results.setdefault(metric, {})
            for qid in test_qids:
                assert qid not in test_results[metric]
                test_results[metric][qid] = results[best_params][metric][qid]

    for qid in qids:
        for metric in metrics:
            values = test_results[metric][qid]
            args.output.write(format_func(metric, qid, values) + "\n")

    for metric in metrics:
        agg = np.array(list(test_results[metric].values()))
        output = format_func(metric, "all", agg.mean(axis=0))
        args.output.write(output + "\n")
        logger.info(output)


if __name__ == "__main__":
    main()
