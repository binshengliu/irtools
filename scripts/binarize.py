#!/usr/bin/env python3
import sys

import argparse
import pickle
import msgpack
import numpy as np

from irtools.jagged import pad_jagged


def serialize(data, fp):
    if fp.name.endswith('npy'):
        np.save(fp, np.asarray(data))
    elif fp.name.endswith('pkl'):
        pickle.dump(data, fp)
    elif fp.name.endswith('msgpack'):
        msgpack.pack(data, fp)
    else:
        np.save(fp, np.asarray(data))


def output_type(string):
    splits = string.rsplit('.', maxsplit=1)
    if len(splits) == 1 or splits[1] not in ['npy']:
        raise argparse.ArgumentTypeError("Unsupported format")
    return argparse.FileType('wb')(string)


def parse_arguments():
    def int_comma(line):
        parsed = [int(x) - 1 for x in str(line).split(',')]
        if any(x < 0 for x in parsed):
            raise argparse.ArgumentTypeError("fields are numbered from 1")
        return parsed

    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--dtype', choices=['int', 'float'], default='int')

    parser.add_argument('--pad')

    parser.add_argument(
        '-i', '--input', type=argparse.FileType('r'), default=sys.stdin)

    parser.add_argument(
        '-o', '--output', required=True, type=argparse.FileType('wb'))

    return parser.parse_args()


def main():
    args = parse_arguments()
    output = []
    max_len = 0
    for line in args.input:
        arr = [int(x) for x in line.strip().split()]
        max_len = max(max_len, len(arr))
        output.append(arr)

    if args.pad is not None:
        output = pad_jagged(output, args.pad, max_len, args.dtype)

    np.save(args.output, np.asarray(output))


if __name__ == '__main__':
    main()
