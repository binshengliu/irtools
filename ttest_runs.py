#!/usr/bin/env python3
import argparse
import sys
from scipy import stats
import pandas as pd
from eval_run import eval_run, eval_run_version


def eprint(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run cross validation on trec runs.')

    parser.add_argument(
        '-m',
        '--measure',
        metavar='MEASURE',
        required=True,
        help='Comma separated measures. Choose from ' + ', '.join(
            ['P', 'map', 'ndcg', 'gdeval', 'ndcg_cut']))

    parser.add_argument(
        '--names', help='Name for LaTeX header.', default='RUN1,RUN2')

    parser.add_argument('qrel', metavar='QREL', help='Qrel path')

    parser.add_argument('run1', metavar='RUN1', help='Run file 1')

    parser.add_argument('run2', metavar='RUN2', help='Run file 2')

    args = parser.parse_args()
    args.measure = args.measure.split(',')
    args.names = args.names.split(',')
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

    table_lines = []
    for measure in args.measure:
        aggregated1, result1 = eval_run(measure, args.qrel, args.run1)
        aggregated2, result2 = eval_run(measure, args.qrel, args.run2)

        query_ids = list(set(result1.keys()) & set(result2.keys()))

        ret_measures = list(aggregated1.keys())
        sort_measures(ret_measures)

        for m in ret_measures:
            scores1 = [result1[q][m] for q in query_ids]
            scores2 = [result2[q][m] for q in query_ids]

            _, pvalue = stats.ttest_rel(scores1, scores2)

            table_lines.append((m, aggregated1[m], aggregated2[m], pvalue))

    df = pd.DataFrame(
        table_lines,
        columns=['Measure', args.names[0], args.names[1], 'p-value'])
    print(df.to_latex(index=False, float_format=lambda f: '{:.3f}'.format(f)))


if __name__ == '__main__':
    main()
