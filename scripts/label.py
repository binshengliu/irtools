#!/usr/bin/env python3
from collections import OrderedDict
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

    parser.add_argument(
        '--append-missing-relevant',
        action='store_true',
        help='Append relevant documents to form training data. '
        'This implies output in a three column format: qno\tdno\trel')

    return parser.parse_args()


def parse_qrel_line(line):
    splits = line.split()
    if len(splits) == 2:
        qno, dno, rel = *splits, 1
    elif len(splits) == 3:
        qno, dno, rel = splits
    elif len(splits) == 4:
        # Trec format
        qno, _, dno, rel = splits
    return qno, dno, int(rel)


def parse_run_line(line):
    splits = line.split()
    if len(splits) == 2:
        qno, dno = splits
    elif len(splits) == 3:
        qno, dno, _ = splits
    elif len(splits) == 6:
        qno, _, dno, *_ = splits
    else:
        assert False, f'Unknown format {len(splits)} fields found'
    return qno, dno


def main():
    args = parse_arguments()

    qrels = {}
    for line in args.qrels:
        qno, dno, rel = parse_qrel_line(line)
        qrels.setdefault(qno, {})[dno] = rel

    delimeter = '\t'
    data = OrderedDict()
    for line in args.input:
        delimeter = '\t' if '\t' in line else ' '
        qno, dno = parse_run_line(line)
        rel = qrels.get(qno, {}).pop(dno, 0)

        if args.append_missing_relevant:
            data.setdefault(qno, []).append((dno, rel))
        else:
            args.output.write(line.rstrip('\n') + f'{delimeter}{rel}\n')

    if args.append_missing_relevant:
        for qno in data.keys():
            for dno, rel in qrels.get(qno, {}).items():
                data[qno].append((dno, rel))

        for qno, dno_rel in data.items():
            for dno, rel in dno_rel:
                args.output.write(delimeter.join([qno, dno, str(rel)]) + '\n')


if __name__ == '__main__':
    main()
