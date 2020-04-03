#!/usr/bin/env python3
import argparse
import sys


def parse_arguments():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument(
        '--allow',
        type=argparse.FileType('r'),
        required=True,
    )

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

    parser.add_argument(
        '-f', '--field', type=int, default=1, help='default stdin')

    return parser.parse_args()


def main():
    args = parse_arguments()
    allow = {x.strip() for x in args.allow if x.strip()}
    field = args.field - 1
    assert field >= 0
    for line in args.input:
        f = line.split()[field]
        if f in allow:
            args.output.write(line)


if __name__ == '__main__':
    main()
