#!/usr/bin/env python3
import argparse
import sys
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--qrels', type=argparse.FileType('r'))

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

    parser.add_argument(
        '--sort-by-relevance',
        action='store_true',
        help='Sort documents in descending relevance order.')

    return parser.parse_args()


def parse_qrel_line(line: str) -> Tuple[str, str, int]:
    splits = line.split()
    if len(splits) == 2:
        qno, dno = splits
        rel = "1"
    elif len(splits) == 3:
        qno, dno, rel = splits
    elif len(splits) == 4:
        # Trec format
        qno, _, dno, rel = splits
    return qno, dno, int(rel)


def parse_run_line(line: str) -> Tuple[str, str, Optional[float]]:
    splits = line.split()
    if len(splits) == 2:
        qno, dno = splits
        score = None
    elif len(splits) == 3:
        qno, dno, _ = splits
        score = None
    elif len(splits) == 6:
        qno, _, dno, _, score_str, _ = splits
        score = float(score_str)
    else:
        assert False, f'Unknown format {len(splits)} fields found'
    return qno, dno, score


def main() -> None:
    args = parse_arguments()

    qrels: Dict[str, Dict[str, int]] = {}
    if args.qrels:
        for line in args.qrels:
            qno, dno, rel = parse_qrel_line(line)
            qrels.setdefault(qno, OrderedDict())[dno] = rel

    delimeter = '\t'
    data: Dict[str, List[Tuple[str, int, Optional[float]]]] = OrderedDict()
    for line in args.input:
        delimeter = '\t' if '\t' in line else ' '
        qno, dno, score = parse_run_line(line)
        rel = qrels.get(qno, {}).pop(dno, 0)

        data.setdefault(qno, []).append((dno, rel, score))

    if args.append_missing_relevant:
        for qno in data.keys():
            qno_scores = [x[2] for x in data[qno] if x[2] is not None]
            max_score = max(qno_scores) if qno_scores else None
            for dno, rel in qrels.get(qno, {}).items():
                data[qno].append((dno, rel, max_score))

    if args.sort_by_relevance:
        for qno in data.keys():
            rel_array = np.array([x[1] for x in data[qno]])
            indexes = np.argsort(-rel_array, kind="stable")
            data[qno] = [data[qno][i] for i in indexes]

    for qno, dno_rel in data.items():
        for dno, rel, score in dno_rel:
            if score is not None:
                line = delimeter.join([qno, dno, str(score), str(rel)]) + '\n'
            else:
                line = delimeter.join([qno, dno, str(rel)]) + '\n'
            args.output.write(line)


if __name__ == '__main__':
    main()
