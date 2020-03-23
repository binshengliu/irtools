#!/usr/bin/env python3
import sys

import argparse
import pickle
import msgpack
import numpy as np

from irtools.eprint import eprint


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

    parser.add_argument(
        '-i', '--input', type=argparse.FileType('r'), default=sys.stdin)

    parser.add_argument(
        '-o', '--output', required=True, type=argparse.FileType('wb'))

    return parser.parse_args()


def unpack(array):
    if array.ndim == 2:
        return len(array), [array.shape[1]] * len(array), array

    total = array[0]
    lens = array[1:][:total]
    data = array[total + 1:]

    splits = np.cumsum(lens)[:-1]
    payload = np.split(data, splits)
    return total, lens, payload


def main():
    args = parse_arguments()
    data = []
    lens = []
    for line in args.input:
        arr = [int(x) for x in line.strip().split()]
        data.extend(arr)
        lens.append(len(arr))

    if len(np.unique(lens)) == 1:
        output = np.array(data).reshape(-1, lens[0])
        eprint('Format: 2D')
    else:
        total = len(lens)
        output = np.array([total] + lens + data)
        eprint('Format: 1D jagged array '
               '[total, len1, len2, ..., data with len1, data with len2, ...]')
    np.save(args.output, output)


if __name__ == '__main__':
    main()
