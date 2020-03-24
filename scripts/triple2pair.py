#!/usr/bin/env python3
import argparse
import sys


def parse_arguments():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument(
        '-i',
        '--input',
        type=argparse.FileType('r'),
        default=sys.stdin,
        help='default stdin')

    parser.add_argument(
        '-o',
        '--output',
        type=argparse.FileType('w'),
        default=sys.stdout,
        help='default stdout')

    return parser.parse_args()


def main():
    args = parse_arguments()
    for line in args.input:
        delim = ' ' if ' ' in line else '\t'
        id, pos, neg = line.split()
        args.output.write(delim.join([id, pos]) + '\n')
        args.output.write(delim.join([id, neg]) + '\n')


if __name__ == '__main__':
    main()
