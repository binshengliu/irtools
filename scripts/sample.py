#!/usr/bin/env python3
import argparse
import sys
from collections import OrderedDict

import numpy as np
from tqdm import tqdm


def parse_arguments():
    def to_zero_base(x):
        x = int(x)
        if x < 1:
            raise argparse.ArgumentTypeError("Field starts from 1.")
        return x - 1

    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-i", "--input", type=argparse.FileType("r"), default=sys.stdin, help=""
    )

    parser.add_argument("-d", "--delimiter", default="\t", help="default to \\t")

    parser.add_argument("--mode", choices=["key", "value", "both"], required=True)
    parser.add_argument("--key-field", type=to_zero_base, default=0)
    parser.add_argument("--key-num", type=int, default=1)
    parser.add_argument("--value-num", type=int, default=1)
    parser.add_argument(
        "--selection", choices=["front", "back", "random"], required=True
    )

    parser.add_argument(
        "-o", "--output", type=argparse.FileType("w"), default=sys.stdout, help=""
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    data = OrderedDict()
    for line in args.input:
        if not line.strip():
            continue
        splits = line.strip().split(args.delimiter)
        data.setdefault(splits[args.key_field], []).append(line)

    if args.mode in ["key", "both"]:
        keys = list(data.keys())
        num = min(len(keys), args.key_num)
        if args.selection == "random":
            keys = np.random.choice(keys, num, replace=False)
        elif args.selection == "front":
            keys = keys[:num]
        elif args.selection == "back":
            keys = keys[-num:]
        else:
            assert False
        keys = set(keys)
        data = OrderedDict([(k, v) for k, v in data.items() if k in keys])

    if args.mode in ["value", "both"]:
        outputs = OrderedDict()
        for k, v in tqdm(data.items(), desc="Sampling values"):
            num = min(len(v), args.value_num)
            if args.selection == "random":
                v = np.random.choice(v, num, replace=False)
            elif args.selection == "front":
                v = v[:num]
            elif args.selection == "back":
                v = v[-num:]
            else:
                assert False
            outputs[k] = v
        data = outputs

    for values in data.values():
        args.output.writelines(list(values))


if __name__ == "__main__":
    main()
