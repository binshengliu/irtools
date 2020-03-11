#!/usr/bin/env python3
import sys

import argparse
import pickle


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
        type=argparse.FileType('wb'),
        help='Output in pickle format')

    return parser.parse_args()


def main():
    args = parse_arguments()
    output = [[int(x) for x in line.split()] for line in args.input]

    pickle.dump(output, args.output)


if __name__ == '__main__':
    main()
