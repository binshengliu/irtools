#!/usr/bin/env python3
import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from irtools.log import get_logger
from sklearn.model_selection import KFold

logger = get_logger(__name__)


examples = """Parameter names and values should be indicated in the file name in `name=value`
format and separated by `_`, `-`, or `/`. When the folds are pre-defined, a `fold=?` is
required in the file path.

Examples:

{name} --files task-epoch=0.eval task-epoch=1.eval task-epoch=2.eval

{name} --train fold=0/train-epoch=0.eval fold=0/train-epoch=1.eval
{space} --test fold=0/test-epoch=0.eval fold=0/test-epoch=1.eval
""".format(
    name=Path(__file__).name, space=" " * len(Path(__file__).name)
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross validation for trec_eval and rbp_eval output files.",
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--train", metavar="FILES", nargs="+")
    parser.add_argument("--test", metavar="FILES", nargs="+")

    parser.add_argument("--files", nargs="+")
    parser.add_argument("--seed", type=int, default=2)

    parser.add_argument(
        "-o", "--output", type=argparse.FileType("w"), default=sys.stdout
    )
    args = parser.parse_args()

    if not args.files and not (args.train and args.test):
        parser.error("Please specify --files or --train/--test")

    if args.train and args.test and args.files:
        parser.error("--files and --train/--test are mutually exclusive")

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


def cv_automatic_folds(
    args: argparse.Namespace,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Any]:
    # params -> metric -> qid -> values
    results: Dict[str, Dict[str, Dict[str, np.array]]] = {}
    qids_set: Set[str] = set()
    metrics = set()
    format_func = trec_format
    parse_func = trec_parse
    for one in args.files:
        stem = one.rsplit(".", maxsplit=1)[0]
        match = re.findall(r"([^-_/]+=[^-_/]+)", stem)
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

    qids = sorted(qids_set)
    kfold = KFold(n_splits=5, shuffle=True, random_state=args.seed)

    # metric -> qid -> values
    test_results: Dict[str, Dict[str, np.ndarray]] = {}
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

    return test_results, format_func


def load_evals_manual_folds(
    files: List[str],
) -> Tuple[Dict[str, Dict[str, Dict[str, Dict[str, np.array]]]], Set[str], Any]:
    # fold -> params -> metric -> qid -> values
    results: Dict[str, Dict[str, Dict[str, Dict[str, np.array]]]] = {}
    metrics = set()
    format_func = trec_format
    parse_func = trec_parse
    for one in files:
        stem = one.rsplit(".", maxsplit=1)[0]
        fold_match = re.search(r"[-_/]?fold=([^-_/]+)", stem)
        assert fold_match is not None
        fold = fold_match[1]

        match = re.findall(r"([^-_/]+=[^-_/]+)", stem)
        match = [x for x in match if not x.startswith("fold=")]
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
                results.setdefault(fold, {}).setdefault(params, {})
                results[fold][params].setdefault(metric, {})
                results[fold][params][metric][qid] = value
    return results, metrics, format_func


def cv_manual_folds(
    args: argparse.Namespace,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Any]:
    # fold -> params -> metric -> qid -> values
    train_evals, metrics, format_func = load_evals_manual_folds(args.train)
    test_evals, _, _ = load_evals_manual_folds(args.test)

    # metric -> qid -> values
    test_results: Dict[str, Dict[str, np.ndarray]] = {}
    for fold in train_evals:
        for metric in metrics:
            train_results: Dict[str, List[float]] = {}
            for params in train_evals[fold]:
                train_results[params] = np.mean(
                    list(train_evals[fold][params][metric].values())
                )
            best_params = max(train_results.items(), key=lambda x: x[1])[0]
            logger.info(f"Best params of fold {fold} metric {metric}: {best_params}")
            test_results.setdefault(metric, {})
            test_results[metric].update(test_evals[fold][best_params][metric])

    return test_results, format_func


def main() -> None:
    args = parse_arguments()
    if args.files:
        test_results, format_func = cv_automatic_folds(args)
    else:
        test_results, format_func = cv_manual_folds(args)

    metrics = sorted(test_results.keys())
    qids = sorted(next(iter(test_results.values())).keys())
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
