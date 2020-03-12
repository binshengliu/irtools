#!/usr/bin/env python3
import sys

import argparse
import pickle
import msgpack
import numpy as np


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
    if len(splits) == 1 or splits[1] not in ['npy', 'pkl', 'msgpack']:
        raise argparse.ArgumentTypeError("Unsupported format")
    return argparse.FileType('wb')(string)


def parse_arguments():
    def int_comma(line):
        parsed = [int(x) - 1 for x in str(line).split(',')]
        if any(x < 0 for x in parsed):
            raise argparse.ArgumentTypeError("fields are numbered from 1")
        return parsed

    parser = argparse.ArgumentParser(description='')

    parser.add_argument(
        '--dtype',
        choices=['int', 'float'],
        default='int',
        help='One-based field index, e.g. 1,2,3. Other fields will be dropped.'
    )

    parser.add_argument(
        '-i',
        '--input',
        type=argparse.FileType('r'),
        default=sys.stdin,
        help='Output in pickle format')

    parser.add_argument(
        '-o',
        '--output',
        required=True,
        type=output_type,
        help='Supported extensions: .npy .pkl .msgpack')

    return parser.parse_args()


def main():
    args = parse_arguments()
    output = [[int(x) for x in line.split()] for line in args.input]

    serialize(output, args.output)


if __name__ == '__main__':
    main()
