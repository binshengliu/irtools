#!/usr/bin/env python3
import argparse
import sys
import os

import pickle

from irtools.bertit import bertit


def output_type(string):
    mode = 'w'
    if string.endswith('.pkl') or string.endswith('.pickle'):
        mode = 'wb'
    return argparse.FileType(mode)(string)


def write_output(lines, out):
    if 'b' in out.mode:
        pickle.dump(lines, out)
    else:
        out.writelines(lines)


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
        '-o',
        '--output',
        required=True,
        type=output_type,
        help='Output in pickle format')

    parser.add_argument(
        '--add-special-tokens', action='store_true', help='Add <cls> ...')

    parser.add_argument('--max-length', type=int, help='Max length')

    parser.add_argument(
        '--pad-to-max-length', action='store_true', help='Pad to max length')
    return parser.parse_args()


def main():
    args = parse_arguments()
    text_mode = 'b' not in args.output.mode
    lines = bertit(sys.stdin, args.threads, args.delimiter, args.field, '\n',
                   text_mode, args.add_special_tokens, args.max_length,
                   args.pad_to_max_length)

    write_output(lines, args.output)


if __name__ == '__main__':
    main()
