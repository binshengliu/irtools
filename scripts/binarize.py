#!/usr/bin/env python3
import argparse
import sys
from typing import Tuple

import numpy as np
from irtools.eprint import eprint


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--dtype", choices=["int", "float"], default="int")

    parser.add_argument("-i", "--input", type=argparse.FileType("r"), default=sys.stdin)

    parser.add_argument("-o", "--output", required=True, type=argparse.FileType("wb"))

    return parser.parse_args()


def unpack(array: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray]:
    if array.ndim == 2:
        return len(array), [array.shape[1]] * len(array), array

    total = int(array[0])
    lens = array[1:][:total].astype(int)
    data = array[total + 1 :]

    splits = np.cumsum(lens)[:-1]
    payload = np.split(data, splits)
    return total, lens, payload


def main() -> None:
    args = parse_arguments()
    data = []
    lens = []
    dtype = eval(args.dtype)
    for line in args.input:
        arr = [dtype(x) for x in line.strip().split()]
        data.extend(arr)
        lens.append(dtype(len(arr)))

    if not data:
        return

    if len(np.unique(lens)) == 1:
        output = np.array(data).reshape(-1, int(lens[0]))
        eprint("Format: 2D")
    else:
        total = len(lens)
        output = np.array([dtype(total)] + lens + data)
        eprint(
            "Format: 1D jagged array "
            "[total, len1, len2, ..., data with len1, data with len2, ...]"
        )
    np.save(args.output, output)


if __name__ == "__main__":
    main()
