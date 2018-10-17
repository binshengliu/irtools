#!/usr/bin/env python3
from eval_run import eval_run
import sys
import argparse
from operator import itemgetter
from itertools import zip_longest, chain
from numpy import array_split
import pandas as pd


def eprint(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr, flush=True)


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


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
        win, tie, loss = 0, 0, 0
        for qno, _ in g:
            s1 = result1.get(qno, 0.0)
            s2 = result2.get(qno, 0.0)
            if s2 > s1 and (s2 - s1) > s1 * args.threshold:
                win += 1
            elif s2 < s1 and (s1 - s2) > s1 * args.threshold:
                loss += 1
            else:
                tie += 1
        r.append((n, win, tie, loss))

    df = pd.DataFrame(r, columns=['range', 'win', 'tie', 'loss'])
    print(df.to_latex(index=False))


if __name__ == '__main__':
    main()
