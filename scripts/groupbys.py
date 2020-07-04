#!/usr/bin/env python3
import argparse
import sys
from argparse import RawTextHelpFormatter
from itertools import groupby, islice
from typing import Iterator, Tuple

import more_itertools as mi
import numpy as np
from irtools.eprint import eprint


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="""
A simple command line wrapper for itertools.groupby. \
Input SHOULD be sorted by the --key column (default: 1).

Operation arguments:

nth: number
head: number
tail: number
sample: ratio_or_number
count: no argument
pad_row: number
collapse: separator
unique: [key_field]

Recipes:

1. Select top 10 lines from a run file:

cat bm25.run | groupbys.py --key 1 --op head --args 10 --input-delimiter ' '

2. Sample 10 lines from a run file:

cat bm25.run | groupbys.py --key 1 --op sample --args 10' --input-delimiter ' '

3. Count the number of docs per query:

cut -d' ' -f1,2 lm.run | groupbys.py --key 1 --op count --input-delimiter ' '

4. Collapse columns into one field:

echo -e '1\\t2\\n1\\t2\\n2\\t2\\n2\\t2' | \
groupbys.py --key 1 --op collapse --args ", "
1	2, 2
2	2, 2

5. Remove duplicate documents in a run file:

cat bm25.run | groupbys.py --key 1 --op unique --args 3 --input-delimiter ' '
    """,
        formatter_class=RawTextHelpFormatter,
    )

    parser.add_argument(
        "-i", "--input", type=argparse.FileType("r"), default=sys.stdin, help=""
    )

    parser.add_argument(
        "--input-delimiter", metavar="", default="\t", help="default to \\t"
    )
    parser.add_argument(
        "--output-delimiter", metavar="", default="\t", help="default to \\t"
    )

    parser.add_argument(
        "--key",
        metavar="int or List[int]",
        type=lambda x: int(x) - 1,
        default=0,
        help="Specify the column index (starts from 1 like sort) to use as the key.",
    )
    parser.add_argument(
        "--op",
        choices=[
            "nth",
            "head",
            "tail",
            "sample",
            "count",
            "pad_row",
            "collapse",
            "unique",
        ],
        required=True,
    )

    parser.add_argument("--args", nargs="*", help="args of `op`")

    parser.add_argument(
        "-o", "--output", type=argparse.FileType("w"), default=sys.stdout, help=""
    )

    return parser.parse_args()


def op_sample(
    giter: Iterator[Tuple[str, Iterator[str]]], args: argparse.Namespace,
) -> Iterator[Tuple[str, str]]:
    value = float(args.args[0])
    for k, gi in giter:
        g = list(gi)
        size = int(len(g) * value) if value < 1.0 else int(value)
        replace = size > len(g)
        indexes = np.random.choice(len(g), size=size, replace=replace)
        for one in indexes:
            yield k, g[one]


def op_pad_row(
    giter: Iterator[Tuple[str, Iterator[str]]], args: argparse.Namespace
) -> Iterator[Tuple[str, str]]:
    n_rows = int(args.args[0])

    for k, gi in giter:
        for i in islice(mi.repeat_last(gi), n_rows):
            yield k, i


def op_head(
    giter: Iterator[Tuple[str, Iterator[str]]], args: argparse.Namespace
) -> Iterator[Tuple[str, str]]:
    n_rows = int(args.args[0])

    for k, gi in giter:
        for i in islice(gi, n_rows):
            yield k, i


def op_tail(
    giter: Iterator[Tuple[str, Iterator[str]]], args: argparse.Namespace
) -> Iterator[Tuple[str, str]]:
    n_rows = int(args.args[0])

    for k, gi in giter:
        for i in mi.tail(n_rows, gi):
            yield k, i


def op_collapse(
    giter: Iterator[Tuple[str, Iterator[str]]], args: argparse.Namespace
) -> Iterator[Tuple[str, str]]:
    sep = args.args[0]

    for k, g in giter:
        yield k, sep.join(g)


def op_nth(
    giter: Iterator[Tuple[str, Iterator[str]]], args: argparse.Namespace
) -> Iterator[Tuple[str, str]]:
    n = int(args.args[0])

    for k, g in giter:
        yield k, mi.nth_or_last(g, n)


def op_count(
    giter: Iterator[Tuple[str, Iterator[str]]], args: argparse.Namespace
) -> Iterator[Tuple[str, str]]:
    for k, g in giter:
        yield k, str(mi.ilen(g))


def op_unique(
    giter: Iterator[Tuple[str, Iterator[str]]], args: argparse.Namespace
) -> Iterator[Tuple[str, str]]:
    def key_func(k: str, g: str) -> str:
        # By default compare entire line without key
        if not args.args:
            return g

        splits = g.split(args.input_delimiter)
        splits.insert(args.key, k)
        key_field = int(args.args[0]) - 1
        if key_field < 0 or key_field >= len(splits):
            eprint(f"Please specify unique field within range [1, {len(splits)}]")
            exit(1)
        return splits[key_field]

    for k, g in giter:
        for u in mi.unique_everseen(g, key=lambda x: key_func(k, x)):
            yield k, u


def main() -> None:
    args = parse_arguments()
    isep: str = args.input_delimiter
    osep: str = args.output_delimiter

    def split_value(x: str) -> str:
        splits = x.rstrip("\n").split(isep)
        return isep.join(splits[: args.key] + splits[args.key + 1 :])

    def split_key(x: str) -> str:
        key: str = x.rstrip("\n").split(isep)[args.key]
        return key

    giter: Iterator[Tuple[str, Iterator[str]]] = groupby(args.input, split_key)
    giter = map(lambda kg: (kg[0], map(split_value, kg[1])), giter)

    func_name = f"op_{args.op}"
    if func_name in globals():
        oiter = globals()[func_name](giter, args)
    else:
        eprint(f"Unimplemented operation `{args.op}`")
        exit(1)

    for k, v in oiter:
        args.output.write(osep.join([k, v]) + "\n")


if __name__ == "__main__":
    main()
