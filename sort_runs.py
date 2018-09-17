#!/usr/bin/env python3
import argparse
import sys
import pandas as pd
import itertools
from eval_run import eval_run, eval_run_version


def eprint(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr, flush=True)


def split_comma(s):
    return s.split(',')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Sort run files based on measure.')

    parser.add_argument(
        '--measure',
        required=True,
        type=split_comma,
        help='Measure. For trec ndcg, use ndcg_cut.20; '
        'for gdeval ndcg, use gdeval@20 ...')

    parser.add_argument(
        '--sort-by',
        '-s',
        help='Sort by, like gdeval-ndcg@20, otherwise the first measure')

    parser.add_argument('qrel', metavar='QREL', help='qrel')

    parser.add_argument('run', nargs='+', metavar='RUN', help='run files')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    eval_run_version()

    results = {}
    measures = []
    for filename, measure in itertools.product(args.run, args.measure):
        results.setdefault(filename, [])
        result, _ = eval_run(measure, args.qrel, filename)

        for m, value in result.items():
            if m not in measures:
                measures.append(m)
            results[filename].append(value)

    results = [[f] + values for f, values in results.items()]
    df = pd.DataFrame(results, columns=['FILE'] + measures)
    if args.sort_by in df.columns:
        df = df.sort_values(by=args.sort_by.upper(), ascending=False)
    else:
        df = df.sort_values(by=df.columns[1], ascending=False)
    print(df.to_latex(index=False, float_format=lambda f: '{:.3f}'.format(f)))


if __name__ == '__main__':
    main()
