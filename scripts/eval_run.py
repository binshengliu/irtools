#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import argparse
import itertools

import pandas as pd
from irtools.eval_run import eval_run_version, eval_run


def split_comma(s):
    return s.split(',')


def find_qrel(s):
    if Path(s).exists():
        return s
    default = {
        'robust04': str(Path(__file__).resolve().with_name('robust04.qrels'))
    }
    path = default.get(s, None)
    if not path:
        raise argparse.ArgumentTypeError(
            'Unknown qrel path or identifier {}'.format(s))

    return path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Sort run files based on measure.')

    parser.add_argument(
        '-m',
        '--measure',
        required=True,
        type=split_comma,
        help='comma separated measurements; for trec ndcg, use trec_ndcg@20; '
        'for gdeval ndcg, use gdeval_ndcg@20; e.g. -m map,P@10,gdeval_ndcg@10')

    parser.add_argument(
        '-s',
        '--sort',
        metavar='MEASURE',
        help='sort by a MEASURE specified in --measure')

    parser.add_argument(
        '-f',
        '--format',
        choices=['latex', 'csv'],
        default='latex',
        help='overall output format; default latex')

    parser.add_argument(
        '-i',
        '--individual',
        action='store_true',
        help='write per query evaluation into individual csv files')

    parser.add_argument('-v', '--version')

    parser.add_argument(
        'qrel', metavar='QREL', help='Qrels path', type=find_qrel)

    parser.add_argument('run', nargs='*', metavar='RUN', help='run files')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    if not args.run:
        args.run = [f.strip('\n') for f in sys.stdin]

    if args.version:
        eval_run_version()

    args.run = [r for r in args.run if os.stat(r).st_size != 0]

    measures = []
    eval_args = []
    for filename, measure in itertools.product(args.run, args.measure):
        eval_args.append((measure, args.qrel, filename, None, False))

    # processes = min(len(os.sched_getaffinity(0)) - 1, len(eval_args))
    eval_results = list(map(eval_run, *zip(*eval_args)))
    eval_results, individuals = zip(*eval_results)

    overall_results = {}
    individual_result = {}
    for (filename, measure), result, individual in zip(
            itertools.product(args.run, args.measure), eval_results,
            individuals):
        overall_results.setdefault(filename, [])

        for m, value in result.items():
            if m not in measures:
                measures.append(m)
            overall_results[filename].append(value)

        individual_result.setdefault(filename, {})
        for qno, m in individual.items():
            individual_result[filename].setdefault(qno, {})
            individual_result[filename][qno].update(m)

    if args.individual:
        for filename, individual in individual_result.items():
            df = pd.DataFrame.from_dict(data=individual, orient='index')
            df = df.reset_index().rename(columns={'index': 'number'})
            df.to_csv(Path(filename).with_suffix('.csv'), index=False)

    overall_results = [[f] + values for f, values in overall_results.items()]
    df = pd.DataFrame(overall_results, columns=['File'] + measures)
    if args.sort is not None and args.sort in df.columns:
        df = df.sort_values(by=args.sort, ascending=False)

    if args.format == 'latex':
        print(df.to_latex(index=False, float_format='%.3f'))
    elif args.format == 'csv':
        print(df.to_csv(index=False, float_format='%.3f'))
    else:
        assert False, 'Unsupported format {}'.format(args.format)


if __name__ == '__main__':
    main()
