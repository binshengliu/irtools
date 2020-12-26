#!/usr/bin/env python3
import argparse
import sys
from typing import Any, Dict, List

import numpy as np
from irtools.eprint import eprint
from tqdm import tqdm


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--dtype", type=lambda x: x.split(","), required=True)

    parser.add_argument("--name", type=lambda x: x.split(","), required=True)

    parser.add_argument("-i", "--input", type=argparse.FileType("r"), default=sys.stdin)

    parser.add_argument("-o", "--output", required=True, type=argparse.FileType("wb"))

    args = parser.parse_args()
    if len(args.dtype) != len(args.name):
        raise ValueError("dtype and name do not match.")
    return args


def main() -> None:
    args = parse_arguments()
    assert len(args.dtype) == len(args.name), "dtype and name don't match"
    dtype = [eval(x) for x in args.dtype]
    to_save: Dict[str, List[Any]] = {name: [] for name in args.name}
    for line in tqdm(args.input, unit=" lines"):
        splits = line.split()
        if len(splits) != len(args.dtype):
            raise ValueError(
                f"dtype has {len(args.dtype)} values while the input has {len(splits)} fields"
            )
        for i in range(len(args.name)):
            name = args.name[i]
            to_save[name].append(dtype[i](splits[i]))

    arrays = {}
    for i in range(len(args.name)):
        name = args.name[i]
        arrays[name] = np.array(to_save[name], dtype=dtype[i])

    length = len(next(iter(to_save.values())))
    eprint(f"Saved {length} records with columns {args.name}.")
    np.savez(args.output, **arrays)


if __name__ == "__main__":
    main()
