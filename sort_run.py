from operator import itemgetter
import argparse
import subprocess
import sys
import pandas as pd
from io import BytesIO


def eprint(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr, flush=True)


def trec_eval_summary(measure, qrel, run_file):
    args = ['trec_eval', '-m', measure, qrel, run_file]
    proc = subprocess.run(args, stdout=subprocess.PIPE)

    output = proc.stdout.decode('utf-8')

    parts = output.split()
    return float(parts[2])


def gdeval(k, qrel_path, run_path):
    gdeval = 'gdeval'
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


def parse_args():
    parser = argparse.ArgumentParser(
        description='Sort run files based on measure.')

    parser.add_argument(
        '--measure',
        required=True,
        help='Measure. For trec ndcg, use ndcg_cut.20; '
        'for gdeval ndcg, use gdeval@20 ...')

    parser.add_argument('--qrel', required=True, help='qrel')

    parser.add_argument('run', nargs='+', help='run files')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    count = len(args.run)
    results = []
    for i, filename in enumerate(args.run, 1):
        try:
            if args.measure.startswith('gdeval'):
                k = args.measure.split('@')[1]
                result, _ = gdeval(k, args.qrel, filename)
                result = result['ndcg@' + k]
            else:
                result = trec_eval_summary(args.measure, args.qrel, filename)
            results.append([filename, result])
        except Exception as e:
            print(e.description(), file=sys.stderr)

        print('{}/{}\r'.format(i, count), end='', flush=True, file=sys.stderr)
    print('')
    results.sort(key=itemgetter(1), reverse=True)
    for (filename, result) in results:
        print('{:.3f} {}'.format(result, filename))


if __name__ == '__main__':
    main()
