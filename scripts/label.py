#!/usr/bin/env python3
import argparse
import sys


def parse_arguments():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--qrels', type=argparse.FileType('r'), required=True)

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

    qrels = {}
    for line in args.qrels:
        splits = line.split()
        if len(splits) == 2:
            qno, dno, rel = *splits, 1
        elif len(splits) == 3:
            qno, dno, rel = splits
        elif len(splits) == 4:
            # Trec format
            qno, _, dno, rel = splits
        qrels.setdefault(qno, {})[dno] = int(rel)

    for line in args.input:
        delimeter = ' ' if ' ' in line else '\t'
        splits = line.split()
        if len(splits) == 2:
            qno, dno = splits
        elif len(splits) == 3:
            qno, dno, _ = splits
        elif len(splits) == 6:
            qno, _, dno, *_ = splits
        else:
            assert False, f'Unknown format {len(splits)} fields found'
        rel = qrels.get(qno, {}).get(dno, 0)

        args.output.write(line.rstrip('\n') + f'{delimeter}{rel}\n')


if __name__ == '__main__':
    main()
