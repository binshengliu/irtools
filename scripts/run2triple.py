#!/usr/bin/env python3
import argparse
import sys

import numpy as np
from irtools.npgroupby import npgroupby


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
        type=argparse.FileType('wb'),
        required=True,
        help='Save path')

    return parser.parse_args()


def get_qno_dno(line):
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


def make_triples(data, labels):
    data = np.array(data)
    idx = np.unique(data[:, 0:2], return_index=True, axis=0)[1]
    data = data[idx]

    labels = sorted(list(labels), reverse=True)
    triples = []
    for group in npgroupby(data, 0):
        for label in labels[:-1]:
            pos = group[group[:, 2].astype(int) == label]
            neg = group[group[:, 2].astype(int) < label]
            comb = np.array(np.meshgrid(pos[:, 1], neg[:, 1])).T.reshape(-1, 2)
            ids = np.full((len(comb), 1), group[0, 0])
            triples.append(np.hstack((ids, comb)))
    triples = np.vstack(triples)
    return triples


def main():
    args = parse_arguments()

    data = []
    qrels = {}
    labels = {0}

    # Gather all positive
    for line in args.qrels:
        splits = line.split()
        qno, _, dno, rel = splits
        if qno.isdigit() and dno.isdigit():
            qno, dno = int(qno), int(dno)
        labels.add(int(rel))
        data.append([qno, dno, int(rel)])
        if int(rel) > 0:
            qrels.setdefault(qno, {})
            qrels[qno][dno] = int(rel)

    for line in args.input:
        qno, dno = get_qno_dno(line)

        if qno.isdigit() and dno.isdigit():
            qno, dno = int(qno), int(dno)

        if qno in qrels and qrels[qno].get(dno, 0) > 0:
            continue

        data.append((qno, dno, 0))

    triples = make_triples(data, labels)
    if triples.size == 0:
        return
    np.save(args.output, triples)


if __name__ == '__main__':
    main()
