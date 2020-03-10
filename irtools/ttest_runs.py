#!/usr/bin/env python3
import argparse
import sys
from scipy import stats
import pandas as pd
from eval_run import eval_run, eval_run_version
from concurrent.futures import ProcessPoolExecutor
import os
from itertools import zip_longest
from functools import reduce


def eprint(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr, flush=True)


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run cross validation on trec runs.')

    parser.add_argument(
        '-m',
        '--measure',
        metavar='MEASURE',
        required=True,
        help=('Comma separated measures. '
              'For example map,P@10,trec_ndcg@10,gdeval_ndcg@10.'))

    parser.add_argument('--names', help='Name for LaTeX header.')

    parser.add_argument('qrel', metavar='QREL', help='Qrel path')

    parser.add_argument('run', metavar='RUN', nargs='*', help='Run files')

    args = parser.parse_args()
    args.measure = args.measure.split(',')
    if args.names:
        args.names = args.names.split(',')
    else:
        args.names = ['RUN' + str(i) for i in range(len(args.run))]
    return args


def sort_measures(measures):
    try:
        measures.sort(key=lambda s: int(s.split('@')[1]))
        measures.sort(key=lambda s: s.split('@')[0])
    except Exception:
        try:
            measures.sort(key=lambda s: int(s.rsplit('_', maxsplit=1)[1]))
            measures.sort(key=lambda s: s.rsplit('_', maxsplit=1)[0])
        except Exception:
            measures.sort()


def main():
    args = parse_args()
    eval_run_version()

    eval_args = []
    for measure in args.measure:
        for run in args.run:
            eval_args.append((measure, args.qrel, run))

    processes = min(len(os.sched_getaffinity(0)) - 1, len(eval_args))
    with ProcessPoolExecutor(max_workers=processes) as executor:
        eval_results = executor.map(eval_run, *zip(*eval_args))

    nruns = len(args.run)
    table_lines = []
    for measure, runs_results in zip(args.measure, grouper(
            eval_results, nruns)):
        query_ids = list(
            reduce(lambda x, y: x & y, [_[1].keys() for _ in runs_results]))

        ret_measures = list(
            reduce(lambda x, y: x & y, [_[0].keys() for _ in runs_results]))
        sort_measures(ret_measures)

        for m in ret_measures:
            scores = [[single[1][q][m] for q in query_ids]
                      for single in runs_results]

            if nruns == 2:
                _, pvalue = stats.ttest_rel(*scores)
            else:
                _, pvalue = stats.f_oneway(*scores)

            table_lines.append((m, *[run[0][m] for run in runs_results],
                                pvalue))

    df = pd.DataFrame(table_lines, columns=['Measure', *args.names, 'p-value'])
    print(df.to_latex(index=False, float_format=lambda f: '{:.3f}'.format(f)))


if __name__ == '__main__':
    main()
