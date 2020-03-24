#!/usr/bin/env python3
import argparse
import sys
import os

from irtools.tokit import tokit


def parse_arguments():
    def int_comma(line):
        parsed = [int(x) - 1 for x in str(line).split(',')]
        if any(x < 0 for x in parsed):
            raise argparse.ArgumentTypeError("fields are numbered from 1")
        return parsed

    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '-d', '--delimiter', default='\t', help='default to \\t')
    parser.add_argument(
        '-j',
        '--threads',
        type=int,
        default=os.cpu_count() // 2,
        help='number of threads, default to half of cpu count')
    parser.add_argument(
        '-f',
        '--field',
        type=int_comma,
        help='one-based field index to process, e.g. 1,2,3.')

    parser.add_argument(
        '-m', '--mode', required=True, choices=tokit.get_all_modes())

    parser.add_argument(
        '-i', '--input', type=argparse.FileType('r'), default=sys.stdin)

    parser.add_argument(
        '-o', '--output', type=argparse.FileType('w'), default=sys.stdout)

    parser.add_argument(
        '--add-special-tokens', action='store_true', help='Add <cls> ...')

    parser.add_argument('--max-length', type=int, help='Max length')

    parser.add_argument(
        '--pad-to-max-length', action='store_true', help='Pad to max length')
    return parser.parse_args()


def main():
    args = parse_arguments()
    lines = tokit(args.mode, args.input, args.threads, args.delimiter,
                  args.field, '\n', True, args.add_special_tokens,
                  args.max_length, args.pad_to_max_length)

    args.output.writelines(lines)


if __name__ == '__main__':
    main()
