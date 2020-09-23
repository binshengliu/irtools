#!/usr/bin/env python3
import argparse
import re
import sys
from typing import Dict, Iterable, List, Set, Tuple

from more_itertools import peekable


def comma_list(x: str) -> List[str]:
    return x.split(",")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "input", type=argparse.FileType("r"), default=sys.stdin, help="default stdin",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=argparse.FileType("w"),
        default=sys.stdout,
        help="default stdout",
    )
    return parser.parse_args()


def proc_ans(inputs: Iterable[str]) -> None:
    rank: Dict[str, int] = {}
    seen: Set[Tuple[str, str]] = set()
    for line in inputs:
        qid, did, _ = line.split()
        if (qid, did) in seen:
            continue
        rank.setdefault(qid, 1)
        print(f"{qid}\t{did}\t{rank[qid]}")
        seen.add((qid, did))
        rank[qid] += 1


def proc_trec(inputs: Iterable[str]) -> None:
    rank: Dict[str, int] = {}
    seen: Set[Tuple[str, str]] = set()
    for line in inputs:
        qid, _, did, _, score, *rest = line.split()
        if (qid, did) in seen:
            continue
        rank.setdefault(qid, 1)
        print(f"{qid} Q0 {did} {rank[qid]} {score} {' '.join(rest)}")
        seen.add((qid, did))
        rank[qid] += 1


def strip_comment(line: str) -> str:
    """https://rosettacode.org/wiki/Strip_comments_from_a_string#Python"""
    match = re.match(r"^([^#]*)#(.*)$", line)
    if match:
        line = match[1]
    return line


def main() -> None:
    args = parse_arguments()
    inputs = peekable(args.input)
    line = strip_comment(inputs.peek())
    if len(line.split()) == 3:
        func = proc_ans
    elif len(line.split()) == 6:
        func = proc_trec
    else:
        assert False, "Unknown format"
    func(inputs)


if __name__ == "__main__":
    main()
