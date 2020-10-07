#!/usr/bin/env python3
import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from irtools.log import get_logger
from more_itertools import first_true
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


class Params:
    def __init__(self, filename: str):
        stem = filename.rsplit(".", maxsplit=1)[0]
        match = re.findall(r"([^-_/]+=[^-_/]+)", stem)
        match = [x for x in match if not x.startswith("fold=")]
        self.filename = filename
        self.params = "-".join(match)

    def __str__(self) -> str:
        return self.params

    def __hash__(self) -> int:
        return hash(self.params)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Params):
            return NotImplemented
        return self.params == other.params


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
    parser.add_argument("--run-output")
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


def name_to_params(name: str) -> str:
    stem = name.rsplit(".", maxsplit=1)[0]
    match = re.findall(r"([^-_/]+=[^-_/]+)", stem)
    match = [x for x in match if not x.startswith("fold=")]
    params = "-".join(match)
    return params


def cv_automatic_folds(
    args: argparse.Namespace,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, Params]], Any]:
    # params -> metric -> qid -> values
    results: Dict[Params, Dict[str, Dict[str, np.array]]] = {}
    qids_set: Set[str] = set()
    metrics = set()
    format_func = trec_format
    parse_func = trec_parse
    for one in args.files:
        params = Params(one)
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
    test_params: Dict[str, Dict[str, Params]] = {}
    for idx, (train_index, test_index) in enumerate(kfold.split(qids)):
        train_qids = [qids[x] for x in train_index]
        test_qids = [qids[x] for x in test_index]
        for metric in metrics:
            train_results: Dict[Params, List[float]] = {}
            for params in results:
                for qid in train_qids:
                    train_results.setdefault(params, [])
                    train_results[params].append(results[params][metric][qid][0])
                train_results[params] = np.mean(train_results[params])
            best_params = max(train_results.items(), key=lambda x: x[1])[0]
            logger.info(
                f"Best params of fold {idx} metric {metric}: {best_params.params}"
            )
            test_results.setdefault(metric, {})
            for qid in test_qids:
                assert qid not in test_results[metric]
                test_results[metric][qid] = results[best_params][metric][qid]
                test_params.setdefault(metric, {})[qid] = best_params

    return test_results, test_params, format_func


def load_evals_manual_folds(
    files: List[str],
) -> Tuple[Dict[str, Dict[Params, Dict[str, Dict[str, np.array]]]], Set[str], Any]:
    # fold -> params -> metric -> qid -> values
    results: Dict[str, Dict[Params, Dict[str, Dict[str, np.array]]]] = {}
    metrics = set()
    format_func = trec_format
    parse_func = trec_parse
    for one in files:
        stem = one.rsplit(".", maxsplit=1)[0]
        fold_match = re.search(r"[-_/]?fold=([^-_/]+)", stem)
        assert fold_match is not None
        fold = fold_match[1]

        params = Params(one)
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
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, Params]], Any]:
    # fold -> params -> metric -> qid -> values
    train_evals, metrics, format_func = load_evals_manual_folds(args.train)
    test_evals, _, _ = load_evals_manual_folds(args.test)

    # metric -> qid -> values
    test_results: Dict[str, Dict[str, np.ndarray]] = {}
    test_params: Dict[str, Dict[str, Params]] = {}
    for fold in train_evals:
        for metric in metrics:
            train_results: Dict[Params, List[float]] = {}
            for params in train_evals[fold]:
                train_results[params] = np.mean(
                    list(train_evals[fold][params][metric].values())
                )
            best_params = max(train_results.items(), key=lambda x: x[1])[0]
            logger.info(
                f"Best params of fold {fold} metric {metric}: {best_params.params}"
            )
            test_results.setdefault(metric, {})
            test_results[metric].update(test_evals[fold][best_params][metric])
            for qid in test_evals[fold][best_params][metric]:
                test_best_instance = first_true(
                    test_evals[fold].keys(), pred=lambda x: x == best_params
                )
                assert test_best_instance is not None
                test_params.setdefault(metric, {})[qid] = test_best_instance

    return test_results, test_params, format_func


def load_run(path: str) -> Dict[str, List[str]]:
    buffers: Dict[str, List[str]] = {}
    with open(path, "r") as f:
        for line in f:
            qid = line.split(maxsplit=1)[0]
            buffers.setdefault(qid, []).append(line)
    return buffers


def main() -> None:
    args = parse_arguments()
    if args.files:
        test_results, test_params, format_func = cv_automatic_folds(args)
    else:
        test_results, test_params, format_func = cv_manual_folds(args)

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

    if args.run_output:
        logger.info("Producing run file")
        os.makedirs(args.run_output, exist_ok=True)
        run_cache = {}
        run_buffer: Dict[str, List[str]] = {}
        for qid in qids:
            for metric in metrics:
                run_file = test_params[metric][qid].filename.replace(".eval", ".run")
                if run_file not in run_cache:
                    run_cache[run_file] = load_run(run_file)
                else:
                    run_buffer.setdefault(metric, [])
                    run_buffer[metric].extend(run_cache[run_file][qid])
        for metric, content in run_buffer.items():
            path = os.path.join(args.run_output, f"{metric}.run")
            logger.info(path)
            with open(path, "w") as f:
                f.writelines(content)


if __name__ == "__main__":
    main()
