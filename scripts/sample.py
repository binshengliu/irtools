#!/usr/bin/env python3
import sys

from tqdm import tqdm
import argparse
import numpy as np


def parse_arguments():
    def to_zero_base(x):
        x = int(x)
        if x < 1:
            raise argparse.ArgumentTypeError("Field starts from 1.")
        return x - 1

    parser = argparse.ArgumentParser(description='')

    parser.add_argument(
        '-i',
        '--input',
        type=argparse.FileType('r'),
        default=sys.stdin,
        help='')

    parser.add_argument(
        '-d', '--delimiter', default='\t', help='default to \\t')

    parser.add_argument(
        '--mode', choices=['key', 'value', 'both'], required=True)
    parser.add_argument('--key-field', type=to_zero_base, default=0)
    parser.add_argument('--key-num', type=int, default=1)
    parser.add_argument('--value-num', type=int, default=1)

    parser.add_argument(
        '-o',
        '--output',
        type=argparse.FileType('w'),
        default=sys.stdout,
        help='')

    return parser.parse_args()


def main():
    args = parse_arguments()
    data = {}
    for line in args.input:
        if not line.strip():
            continue
        splits = line.strip().split(args.delimiter)
        data.setdefault(splits[args.key_field], []).append(line)

    if args.mode in ['key', 'both']:
        keys = list(data.keys())
        keys = np.random.choice(
            keys, min(len(keys), args.key_num), replace=False)
        data = {k: data[k] for k in keys}

    if args.mode in ['value', 'both']:
        data = {
            k: np.random.choice(v, min(len(v), args.value_num), replace=False)
            for k, v in tqdm(data.items(), desc='Sampling values')
        }

    for values in data.values():
        args.output.writelines(list(values))


if __name__ == '__main__':
    main()
