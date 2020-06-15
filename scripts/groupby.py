#!/usr/bin/env python3
import argparse
import ast
import inspect
import sys
from argparse import RawTextHelpFormatter
from typing import Dict, List, Union

import pandas as pd
from irtools.eprint import eprint


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="""
A simple command line wrapper for pandas.groupby.

Recipes:

1. Select top 10 lines from a run file:

cat bm25.run | groupby.py --by 0 --op head --args 10 --input-delimiter ' '

2. Sample 10 lines from a run file:

cat bm25.run | groupby.py --by 0 --op sample --args '{"n": 10, "replace": True}' \
--input-delimiter ' '

3. Count the number of docs per query:

cut -d' ' -f1,2 lm.run | groupby.py --by 0 --op count --input-delimiter ' '

4. Collapse columns into one field:

echo -e '1\\t2\\n1\\t2\\n2\\t2\\n2\\t2' | \
groupby.py --by 0 --op collapse --args '[1, ", "]'
1	2, 2
2	2, 2

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
        "--by",
        metavar="int or List[int]",
        type=ast.literal_eval,
        default=0,
        help="Specify the column index (starts from 0) to use as the key.",
    )
    parser.add_argument(
        "--op",
        choices=["nth", "head", "tail", "sample", "count", "pad_row", "collapse"],
        required=True,
    )

    parser.add_argument(
        "--args", type=ast.literal_eval, default=[], help="args of `op`"
    )

    parser.add_argument(
        "-o", "--output", type=argparse.FileType("w"), default=sys.stdout, help=""
    )

    return parser.parse_args()


def sample_wrapper(
    grouped: pd.core.groupby.DataFrameGroupBy,
    opargs: List[Union[int, float, str]],
    varkw: Dict[str, Union[int, float, str]],
) -> pd.DataFrame:
    return grouped.apply(lambda x: x.sample(*opargs, **varkw))


def pad_row(
    grouped: pd.core.groupby.DataFrameGroupBy,
    opargs: List[Union[int, float, str]],
    varkw: Dict[str, Union[int, float, str]],
) -> pd.DataFrame:
    assert len(opargs) == 1
    assert isinstance(opargs[0], int)
    n_rows = opargs[0]

    def _pad_row(df: pd.DataFrame) -> pd.DataFrame:
        if n_rows > len(df):
            return df.append(df.iloc[[-1] * (n_rows - len(df))])
        else:
            return df

    return grouped.apply(_pad_row)


def collapse(
    grouped: pd.core.groupby.DataFrameGroupBy,
    opargs: List[Union[int, float, str]],
    varkw: Dict[str, Union[int, float, str]],
) -> pd.DataFrame:
    assert len(opargs) == 2
    assert isinstance(opargs[0], int)
    field, sep = opargs[0], str(opargs[1])

    collapsed = grouped[field].apply(lambda x: x.astype(str).str.cat(sep=sep))
    collapsed.index.name = "unused"
    return collapsed.reset_index()


def main() -> None:
    args = parse_arguments()
    if isinstance(args.args, dict):
        opargs, varkw = [], args.args
    elif isinstance(args.args, (int, float, str)):
        opargs, varkw = [args.args], {}
    else:
        opargs, varkw = args.args, {}

    data = pd.read_csv(args.input, sep=args.input_delimiter, header=None)

    if args.op == "sample":
        grouped = data.groupby(by=args.by, as_index=False, sort=False)
        output = sample_wrapper(grouped, opargs, varkw)
    elif args.op == "pad_row":
        grouped = data.groupby(by=args.by, as_index=False, sort=False)
        output = pad_row(grouped, opargs, varkw)
    elif args.op == "collapse":
        grouped = data.groupby(by=args.by, sort=False)
        output = collapse(grouped, opargs, varkw)
    else:
        grouped = data.groupby(by=args.by, as_index=False, sort=False)
        func = getattr(grouped, args.op)
        spec = inspect.getfullargspec(func)
        try:
            output = func(*opargs, **varkw)
        except TypeError:
            eprint(f"Check `{args.op}` spec `{spec}`. Specified `{args.args}`")
            exit(1)

    output.to_csv(args.output, sep=args.output_delimiter, index=False, header=False)


if __name__ == "__main__":
    main()
