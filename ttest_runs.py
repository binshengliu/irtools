import argparse
import sys
from scipy import stats
import pandas as pd
import subprocess
from io import BytesIO
from pathlib import Path


def eprint(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr, flush=True)


def gdeval_version():
    gdeval = str(Path(__file__).parent / 'gdeval.pl')
    args = [gdeval, '-version']
    proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    eprint(proc.stdout.decode('utf-8').strip())


def trec_eval_version():
    trec_eval = str(Path(__file__).parent / 'trec_eval')
    args = [trec_eval, '--version']
    proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    eprint(proc.stderr.decode('utf-8').strip())


def gdeval(k, qrel_path, run_path):
    gdeval = str(Path(__file__).parent / 'gdeval.pl')
    args = [gdeval, '-k', str(k), qrel_path, run_path]
    eprint(' '.join(args))
    proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    df = pd.read_csv(BytesIO(proc.stdout))
    qno_results = {}
    aggregated = {}
    for _, row in df.iterrows():
        if row['topic'] == 'amean':
            aggregated[df.columns[2]] = row[df.columns[2]]
            aggregated[df.columns[3]] = row[df.columns[3]]
            continue
        qno_results.setdefault(row['topic'],
                               {})[df.columns[2]] = row[df.columns[2]]
        qno_results.setdefault(row['topic'],
                               {})[df.columns[3]] = row[df.columns[3]]

    return aggregated, qno_results


def gdeval_all(qrel_path, run_path):
    aggregated = {}
    qno_results = {}
    for k in [5, 10, 15, 20, 30, 100, 200, 500, 1000]:
        agg, result = gdeval(k, qrel_path, run_path)
        aggregated.update(agg)
        for qno, measure in result.items():
            qno_results.setdefault(qno, {}).update(measure)
    return aggregated, qno_results


def trec_eval(measure, qrel_path, run_path):
    trec_eval = str(Path(__file__).parent / 'trec_eval')
    args = [trec_eval, '-q', '-m', measure, qrel_path, run_path]
    eprint(' '.join(args))
    proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    df = pd.read_csv(
        BytesIO(proc.stdout),
        delim_whitespace=True,
        names=['measure', 'qno', 'value'])
    qno_results = {}
    aggregated = {}
    for _, row in df.iterrows():
        if row['qno'] == 'all':
            aggregated[row['measure']] = row['value']
            continue
        qno_results.setdefault(row['qno'], {})[row['measure']] = row['value']

    return aggregated, qno_results


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
    trec_eval_version()
    gdeval_version()

    table_lines = []
    for m in args.measure:
        if m == 'gdeval':
            aggregated1, result1 = gdeval_all(args.qrel, args.run1)
            aggregated2, result2 = gdeval_all(args.qrel, args.run2)
        else:
            aggregated1, result1 = trec_eval(m, args.qrel, args.run1)
            aggregated2, result2 = trec_eval(m, args.qrel, args.run2)
        query_ids = list(set(result1.keys()) & set(result2.keys()))

        ret_measures = list(next(iter(result1.values())).keys())
        sort_measures(ret_measures)

        for ret_measure in ret_measures:
            scores1 = [result1[q][ret_measure] for q in query_ids]
            scores2 = [result2[q][ret_measure] for q in query_ids]

            _, pvalue = stats.ttest_rel(scores1, scores2)

            display = ret_measure.replace('_cut_', '@').upper()
            if m == 'gdeval':
                display = 'GDEVAL-' + display
            elif m.startswith('ndcg'):
                display = 'TREC-' + display
            table_lines.append((display, aggregated1[ret_measure],
                                aggregated2[ret_measure], pvalue))

    df = pd.DataFrame(
        table_lines,
        columns=['Measure', args.names[0], args.names[1], 'p-value'])
    print(df.to_latex(index=False, float_format=lambda f: '{:.3f}'.format(f)))


if __name__ == '__main__':
    main()
