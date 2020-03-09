#!/usr/bin/env python3
from irtools.eval_run import eval_run
from irtools.wtl import wtl_seq
import argparse
from operator import itemgetter
from itertools import chain
from numpy import array_split
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description='Sort run files based on measure.')

    parser.add_argument(
        '--measure',
        required=True,
        help='Measure. For trec ndcg, use ndcg_cut.20; '
        'for gdeval ndcg, use gdeval@20 ...')

    parser.add_argument('--threshold', default=0.1, type=float)

    parser.add_argument('qrel', metavar='QREL', help='qrel')

    parser.add_argument('base')

    parser.add_argument('comparison')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    _, result1 = eval_run(args.measure, args.qrel, args.base)
    result1 = {qno: m[args.measure] for qno, m in result1.items()}

    _, result2 = eval_run(args.measure, args.qrel, args.comparison)
    result2 = {qno: m[args.measure] for qno, m in result2.items()}

    sorted_by_base = sorted(result1.items(), key=itemgetter(1))
    names = ['0-100', '0-25', '25-50', '50-75', '75-100']
    r = []
    for n, g in zip(names,
                    chain([sorted_by_base], array_split(sorted_by_base, 4))):
        values1 = [result1.get(qno, 0.0) for qno, _ in g]
        values2 = [result2.get(qno, 0.0) for qno, _ in g]
        win, tie, loss = wtl_seq(values1, values2, args.threshold)
        r.append((n, win, tie, loss))

    df = pd.DataFrame(r, columns=['range', 'win', 'tie', 'loss'])
    print(df.to_latex(index=False))


if __name__ == '__main__':
    main()
