#!/usr/bin/env python3
import argparse
import sys
import os

from irtools.tokit import tokit, get_all_models


def check_valid_model(model):
    if model not in get_all_models():
        raise argparse.ArgumentTypeError(f"Unknown model {model}")
    return model


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
        help=f'default to half of cpu count {os.cpu_count() // 2}')
    parser.add_argument(
        '-f',
        '--field',
        type=int_comma,
        help='one-based field index to process like `cut -f`, e.g. 1,2,3.')

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        '-m',
        '--model',
        type=check_valid_model,
        help='model name like bert-base-uncased. '
        'specify --list-models for all supported models.')

    group.add_argument(
        '-l',
        '--list-models',
        action='store_true',
        help='list all supported models, e.g. bert-base-uncased')

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
        '--add-special-tokens', action='store_true', help='default false')

    parser.add_argument('--max-length', type=int, help='default no limit')

    parser.add_argument(
        '--pad-to-max-length', action='store_true', help='default false')
    return parser.parse_args()


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    args = parse_arguments()
    if args.list_models:
        args.output.writelines([x + '\n' for x in get_all_models()])
        return
    lines = tokit(args.model, args.input, args.threads, args.delimiter,
                  args.field, '\n', True, args.add_special_tokens,
                  args.max_length, args.pad_to_max_length)

    args.output.writelines(lines)


if __name__ == '__main__':
    main()
