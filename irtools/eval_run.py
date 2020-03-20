#!/usr/bin/env python3
import sys
import subprocess
from tempfile import NamedTemporaryFile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import argparse
import itertools
import pandas as pd
import time


def eprint(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr, flush=True)


def gdeval_version():
    gdeval = str(Path(__file__).resolve().with_name('gdeval.pl'))
    args = [gdeval, '-version']
    proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    eprint(proc.stdout.decode('utf-8').strip())


def trec_eval_version():
    args = [trec_eval_path(), '--version']
    proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    eprint(proc.stderr.decode('utf-8').strip())


def eval_run_version():
    trec_eval_version()
    gdeval_version()


def gdeval(k, qrel_path, run_path, show_cmd=True):
    args = [gdeval_path(), '-k', str(k), qrel_path, run_path]
    if show_cmd:
        eprint(' '.join(args))
    proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    lines = proc.stdout.decode('utf-8').splitlines()
    _, topic, ndcg, err = lines[0].split(',')
    if topic != 'topic':
        raise ValueError('Unrecognizable gdeval output')

    aggregated = {}
    _, qno, ndcg_value, err_value = lines[-1].split(',')
    if qno != 'amean':
        raise ValueError('Unrecognizable gdeval output')
    aggregated[ndcg] = float(ndcg_value)
    aggregated[err] = float(err_value)

    qno_results = {}
    for line in lines[1:-1]:
        _, qno, ndcg_value, err_value = line.split(',')
        qno_results.setdefault(qno, {})
        qno_results[qno][ndcg] = float(ndcg_value)
        qno_results[qno][err] = float(err_value)

    return aggregated, qno_results


def gdeval_all(qrel_path, run_path, show_cmd=True):
    aggregated = {}
    qno_results = {}
    gd_args = [(k, qrel_path, run_path, show_cmd)
               for k in [5, 10, 15, 20, 30, 100, 200, 500, 1000]]
    processes = min(len(os.sched_getaffinity(0)) - 1, len(gd_args))
    with ProcessPoolExecutor(max_workers=processes) as executor:
        gd_results = executor.map(gdeval, *zip(*gd_args))

    for agg, result in gd_results:
        aggregated.update(agg)
        for qno, measure in result.items():
            qno_results.setdefault(qno, {}).update(measure)
    return aggregated, qno_results


def gdeval_path():
    return str(Path(__file__).resolve().with_name('gdeval.pl'))


def trec_eval_path():
    return str(Path(__file__).resolve().with_name('trec_eval'))


def rbp_eval_path():
    return str(Path(__file__).resolve().with_name('rbp_eval'))


def trec_support():
    cmd = f'{trec_eval_path()} -h -m all_trec'
    proc = subprocess.run(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True)
    lines = proc.stdout.splitlines()
    start = None
    for i in enumerate(len(lines)):
        if lines[i].startswith('Individual measure documentation'):
            start = i + 1
            break
    measures = [l for l in lines[:start] if not l.startswith(' ')]
    return measures


def trec_eval(measure, qrel_path, run_path, show_cmd=True):
    args = [trec_eval_path(), '-p', '-q', '-m', measure, qrel_path, run_path]
    if show_cmd:
        eprint(' '.join(args))
    proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    lines = proc.stdout.decode('utf-8').splitlines()

    aggregated = {}
    qno_results = {}
    for line in lines:
        measure, qno, value = line.split()
        if qno == 'all':
            aggregated[measure] = float(value)
        else:
            qno_results.setdefault(qno, {})
            qno_results[qno][measure] = float(value)

    return aggregated, qno_results


def rbp_eval(p, qrel_path, run_path, show_cmd=True):
    args = [rbp_eval_path(), '-H', '-q', '-p', p, qrel_path, run_path]
    if show_cmd:
        eprint(' '.join(args))
    proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    lines = proc.stdout.decode('utf-8').splitlines()

    aggregated = {}
    qno_results = {}
    for line in lines:
        _, _, _, qno, _, _, _, value, res = line.split()
        if 'nan' in value:
            continue
        if qno == 'all':
            aggregated['rbp@{}'.format(p)] = float(value)
            aggregated['rbp@{}_res'.format(p)] = float(res)
        else:
            qno_results.setdefault(qno, {})
            qno_results[qno]['rbp@{}'.format(p)] = float(value)
            qno_results[qno]['rbp@{}_res'.format(p)] = float(res)

    return aggregated, qno_results


def match_prefix(s, prefix):
    return s.startswith(prefix)


def match_exact(s, prefix):
    return s == prefix


def match_true(s, prefix):
    return True


def eval_gdeval_k(measure, qrel_path, run_path, show_cmd=True):
    if measure.startswith('gdeval_ndcg@'):
        kind = 'ndcg'
    elif measure.startswith('gdeval_err@'):
        kind = 'err'
    else:
        raise ValueError

    k = int(measure.split('@')[1])
    gdeval_name = '{}@{}'.format(kind, k)
    norm_name = 'gdeval_{}@{}'.format(kind, k)
    aggregated, qno_results = gdeval(str(k), qrel_path, run_path, show_cmd)
    aggregated = {norm_name: aggregated[gdeval_name]}
    qno_results = {
        qno: {
            norm_name: ms[gdeval_name]
        }
        for qno, ms in qno_results.items()
    }

    return aggregated, qno_results


def eval_gdeval_cut(measure, qrel_path, run_path, show_cmd=True):
    if measure == 'gdeval_ndcg_cut':
        kind = 'ndcg'
    elif measure == 'gdeval_err_cut':
        kind = 'err'
    else:
        raise ValueError

    def transform(name):
        return 'gdeval_{}@{}'.format(kind, name.split('@')[1])

    aggregated, qno_results = gdeval_all(qrel_path, run_path, show_cmd)
    aggregated = {
        transform(m): v
        for m, v in aggregated.items() if m.startswith(kind)
    }
    qno_results = {
        qno: {transform(m): v
              for m, v in ms.items() if m.startswith(kind)}
        for qno, ms in qno_results.items()
    }

    return aggregated, qno_results


def eval_trec_ndcg_k(measure, qrel_path, run_path, show_cmd=True):
    k = int(measure.split('@')[1])

    trec_input = 'ndcg_cut.{}'.format(k)
    trec_output = 'ndcg_cut_{}'.format(k)
    aggregated, qno_results = trec_eval(
        trec_input, qrel_path, run_path, show_cmd=True)
    aggregated = {measure: aggregated[trec_output]}
    qno_results = {
        qno: {
            measure: ms[trec_output]
        }
        for qno, ms in qno_results.items()
    }

    return aggregated, qno_results


def eval_trec_ndcg_cut(measure, qrel_path, run_path, show_cmd=True):
    def transform(m):
        return 'trec_' + m.replace('_cut_', '@')

    aggregated, qno_results = trec_eval('ndcg_cut', qrel_path, run_path,
                                        show_cmd)
    aggregated = {transform(m): v for m, v in aggregated.items()}
    qno_results = {
        qno: {transform(m): v
              for m, v in ms.items()}
        for qno, ms in qno_results.items()
    }

    return aggregated, qno_results


def eval_trec_map_k(measure, qrel_path, run_path, show_cmd=True):
    k = int(measure.split('@')[1])
    trec_input = 'map_cut.{}'.format(k)
    trec_output = 'map_cut_{}'.format(k)
    aggregated, qno_results = trec_eval(trec_input, qrel_path, run_path,
                                        show_cmd)
    aggregated = {measure: aggregated[trec_output]}
    qno_results = {
        qno: {
            measure: ms[trec_output]
        }
        for qno, ms in qno_results.items()
    }

    return aggregated, qno_results


def eval_trec_map_cut(measure, qrel_path, run_path, show_cmd=True):
    def transform(m):
        return m.replace('_cut_', '@')

    aggregated, qno_results = trec_eval('map_cut', qrel_path, run_path,
                                        show_cmd)
    aggregated = {transform(m): v for m, v in aggregated.items()}
    qno_results = {
        qno: {transform(m): v
              for m, v in ms.items()}
        for qno, ms in qno_results.items()
    }

    return aggregated, qno_results


def eval_trec_default(measure, qrel_path, run_path, show_cmd=True):
    aggregated, qno_results = trec_eval(measure, qrel_path, run_path, show_cmd)
    return aggregated, qno_results


def eval_trec_general_k(measure, qrel_path, run_path, show_cmd=True):
    trec_input = measure.replace('@', '.')
    trec_output = measure.replace('@', '_')
    aggregated, qno_results = trec_eval(trec_input, qrel_path, run_path,
                                        show_cmd)
    aggregated = {measure: aggregated[trec_output]}
    qno_results = {
        qno: {
            measure: ms[trec_output]
        }
        for qno, ms in qno_results.items()
    }

    return aggregated, qno_results


def eval_trec_general_cut(measure, qrel_path, run_path, show_cmd=True):
    def transform(m):
        return m.replace('_', '@')

    trec_input = measure.rstrip('_cut')
    aggregated, qno_results = trec_eval(trec_input, qrel_path, run_path,
                                        show_cmd)
    aggregated = {transform(m): v for m, v in aggregated.items()}
    qno_results = {
        qno: {transform(m): v
              for m, v in ms.items()}
        for qno, ms in qno_results.items()
    }

    return aggregated, qno_results


def eval_rbp_p(measure, qrel_path, run_path, show_cmd=True):
    p = measure.split('@')[1]
    aggregated, qno_results = rbp_eval(p, qrel_path, run_path, show_cmd)
    # aggregated = {measure: aggregated[trec_output]}
    # qno_results = {
    #     qno: {
    #         measure: ms[trec_output]
    #     }
    #     for qno, ms in qno_results.items()
    # }

    return aggregated, qno_results


class EvalEntry(object):
    def __init__(self, match_str, match_function, eval_func):
        self.match_str = match_str
        self.match_function = match_function
        self.eval_func = eval_func


functions = [
    EvalEntry('gdeval_ndcg@', match_prefix, eval_gdeval_k),
    EvalEntry('gdeval_ndcg_cut', match_exact, eval_gdeval_cut),
    EvalEntry('gdeval_err@', match_prefix, eval_gdeval_k),
    EvalEntry('gdeval_err_cut', match_exact, eval_gdeval_cut),
    EvalEntry('trec_ndcg@', match_prefix, eval_trec_ndcg_k),
    EvalEntry('trec_ndcg_cut', match_prefix, eval_trec_ndcg_cut),
    EvalEntry('map@', match_prefix, eval_trec_map_k),
    EvalEntry('map_cut', match_exact, eval_trec_map_cut),
    EvalEntry('map', match_exact, eval_trec_default),
    EvalEntry('P@', match_prefix, eval_trec_general_k),
    EvalEntry('P_cut', match_exact, eval_trec_general_cut),
    EvalEntry('recip_rank', match_exact, eval_trec_default),
    EvalEntry('mrr', match_exact, eval_trec_default),
    EvalEntry('rr', match_exact, eval_trec_default),
    EvalEntry('rbp@', match_prefix, eval_rbp_p),
    EvalEntry('', match_true, eval_trec_default),
]


def eval_run(measure, qrel_path, run, show_cmd=True):
    """Supported measure: All names supported by trec_eval. \"gdeval\" and
    \"gdeval@k\" are also supported but are not official names.
    """
    if isinstance(run, str):
        f = NamedTemporaryFile(mode='wt', delete=False)
        f.write(run)
        f.close()
        run_path = f.name
    else:
        run_path = run

    for entry in functions:
        if entry.match_function(measure, entry.match_str):
            aggregated, qno_results = entry.eval_func(measure, qrel_path,
                                                      run_path, show_cmd)
            return aggregated, qno_results

    if isinstance(run, str):
        os.unlink(f.name)

    raise ValueError('Unrecognizable measure {}'.format(measure))


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

    parser.add_argument(
        'qrel', metavar='QREL', help='Qrels path', type=find_qrel)

    parser.add_argument('run', nargs='*', metavar='RUN', help='run files')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    if not args.run:
        args.run = [f.strip('\n') for f in sys.stdin]

    eval_run_version()

    args.run = [r for r in args.run if os.stat(r).st_size != 0]

    measures = []
    eval_args = []
    for filename, measure in itertools.product(args.run, args.measure):
        eval_args.append((measure, args.qrel, filename))

    processes = min(len(os.sched_getaffinity(0)) - 1, len(eval_args))
    with ProcessPoolExecutor(max_workers=processes) as executor:
        eval_results = list(executor.map(eval_run, *zip(*eval_args)))
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
