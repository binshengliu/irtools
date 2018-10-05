#!/usr/bin/env python3
from pathlib import Path
import argparse
import os
import sys


def eprint(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr, flush=True)


def filter_oracle_run(qrel_path,
                      run_path,
                      number,
                      min_rel=1,
                      score_type='uniform'):
    relevance = {}
    for line in qrel_path.read_text().splitlines():
        qno, _, docno, rel = line.split()
        relevance.setdefault(qno, {})[docno] = int(rel)

    run_lines = run_path.read_text().splitlines()
    filtered_docnos = {}
    while min_rel >= 0:
        for line in run_lines:
            qno, _, docno, _, score, _ = line.split()
            if len(filtered_docnos.get(qno, set())) >= number:
                continue
            rel = relevance.get(qno, {}).get(docno, 0)
            if rel < min_rel:
                continue

            filtered_docnos.setdefault(qno, set()).add(docno)
        min_rel -= 1

    ranking = {}
    formated = []
    for line in run_lines:
        qno, _, docno, _, score, _ = line.split()
        ranking.setdefault(qno, 1)
        if docno not in filtered_docnos.get(qno, set()):
            continue
        rel = relevance.get(qno, {}).get(docno, 0)
        if score_type == 'uniform':
            score = 0
        elif score_type == 'relevance':
            score = rel
        line = '{} Q0 {} {} {} rel{}'.format(qno, docno, ranking[qno], score,
                                             rel)
        formated.append(line)
        ranking[qno] += 1

    return formated


def parse_args():
    parser = argparse.ArgumentParser(
        description='Filter a run file for true relevant documents.')

    parser.add_argument(
        'qrels',
        type=Path,
        help='Documents with a score lower than the value will be removed.')

    parser.add_argument('run', type=Path, help='Run file.')

    parser.add_argument(
        '--number',
        '-n',
        type=int,
        default=5,
        help='Number of relevant documents per query.')

    parser.add_argument(
        '--min-relevance',
        '-r',
        type=int,
        default=1,
        help='Minimum relevance score.')

    parser.add_argument(
        '--score-type',
        '-s',
        default='uniform',
        choices=['uniform', 'relevance'],
        help='How to score documents.')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    formated = filter_oracle_run(args.qrels, args.run, args.number,
                                 args.min_relevance, args.score_type)
    print(os.linesep.join(formated))


if __name__ == '__main__':
    main()
