#!/usr/bin/env python3
import argparse
import subprocess
import sys
import pandas as pd
from pathlib import Path
import itertools


def eprint(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr, flush=True)


def trec_eval(measure, qrel_path, run_path):
    trec_eval = str(Path(__file__).resolve().with_name('trec_eval'))
    args = [trec_eval, '-m', measure, qrel_path, run_path]
    eprint(' '.join(args))
    proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    aggregated = {}
    for line in proc.stdout.decode('utf-8').splitlines():
        measure, qno, value = line.split()
        if qno != 'all':
            raise ValueError('Unrecognizable trec_eval output')
        aggregated[measure] = float(value)

    return aggregated


def gdeval(k, qrel_path, run_path):
    gdeval = str(Path(__file__).resolve().with_name('gdeval.pl'))
    args = [gdeval, '-k', str(k), qrel_path, run_path]
    eprint(' '.join(args))
    proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    lines = proc.stdout.decode('utf-8').splitlines()
    _, _, *measures = lines[0].split(',')

    aggregated = {}
    _, qno, *values = lines[-1].split(',')
    for m, v in zip(measures, values):
        aggregated[m] = float(v)

    return aggregated


def gdeval_all(qrel_path, run_path):
    aggregated = {}
    for k in [5, 10, 15, 20, 30, 100, 200, 500, 1000]:
        agg = gdeval(k, qrel_path, run_path)
        aggregated.update(agg)
    return aggregated


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

    count = len(args.run) * len(args.measure)
    results = {}
    measures = []
    for i, (filename, measure) in enumerate(
            itertools.product(args.run, args.measure)):
        results.setdefault(filename, [])
        if measure.startswith('gdeval'):
            if '@' in measure:
                k = measure.split('@')[1]
                result = gdeval(k, args.qrel, filename)
            else:
                result = gdeval_all(args.qrel, filename)
        else:
            result = trec_eval(measure, args.qrel, filename)

        for m, value in result.items():
            m = m.replace('_cut_', '@')
            m = m.replace('_', '@')
            m = 'GDEVAL-' + m if measure.startswith('gdeval') else m
            m = 'TREC-' + m if measure.startswith('ndcg') else m
            m = m.upper()
            if m not in measures:
                measures.append(m)
            results[filename].append(value)
        # eprint(m, result)

        eprint('{}/{}\r'.format(i, count), end='')

    results = [[f] + values for f, values in results.items()]
    df = pd.DataFrame(results, columns=['FILE'] + measures)
    if args.sort_by in df.columns:
        df = df.sort_values(by=args.sort_by.upper(), ascending=False)
    else:
        df = df.sort_values(by=df.columns[1], ascending=False)
    print(df.to_latex(index=False, float_format=lambda f: '{:.3f}'.format(f)))


if __name__ == '__main__':
    main()
