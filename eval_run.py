#!/usr/bin/env python3
import sys
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import os


def eprint(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr, flush=True)


def gdeval_version():
    gdeval = str(Path(__file__).resolve().with_name('gdeval.pl'))
    args = [gdeval, '-version']
    proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    eprint(proc.stdout.decode('utf-8').strip())


def trec_eval_version():
    trec_eval = str(Path(__file__).resolve().with_name('trec_eval'))
    args = [trec_eval, '--version']
    proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    eprint(proc.stderr.decode('utf-8').strip())


def eval_run_version():
    trec_eval_version()
    gdeval_version()


def gdeval(k, qrel_path, run_path):
    gdeval = str(Path(__file__).resolve().with_name('gdeval.pl'))
    args = [gdeval, '-k', str(k), qrel_path, run_path]
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


def gdeval_all(qrel_path, run_path):
    aggregated = {}
    qno_results = {}
    gd_args = [(k, qrel_path, run_path)
               for k in [5, 10, 15, 20, 30, 100, 200, 500, 1000]]
    processes = min(len(os.sched_getaffinity(0)) - 1, len(gd_args))
    with ProcessPoolExecutor(max_workers=processes) as executor:
        gd_results = executor.map(gdeval, *zip(*gd_args))

    for agg, result in gd_results:
        aggregated.update(agg)
        for qno, measure in result.items():
            qno_results.setdefault(qno, {}).update(measure)
    return aggregated, qno_results


def trec_eval(measure, qrel_path, run_path):
    trec_eval = str(Path(__file__).resolve().with_name('trec_eval'))
    args = [trec_eval, '-q', '-m', measure, qrel_path, run_path]
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


def match_prefix(s, prefix):
    return s.startswith(prefix)


def match_exact(s, prefix):
    return s == prefix


def match_true(s, prefix):
    return True


def eval_gdeval_k(measure, qrel_path, run_path):
    if measure.startswith('gdeval_ndcg@'):
        kind = 'ndcg'
    elif measure.startswith('gdeval_err@'):
        kind = 'err'
    else:
        raise ValueError

    k = int(measure.split('@')[1])
    gdeval_name = '{}@{}'.format(kind, k)
    norm_name = 'gdeval_{}@{}'.format(kind, k)
    aggregated, qno_results = gdeval(str(k), qrel_path, run_path)
    aggregated = {norm_name: aggregated[gdeval_name]}
    qno_results = {
        qno: {
            norm_name: ms[gdeval_name]
        }
        for qno, ms in qno_results.items()
    }

    return aggregated, qno_results


def eval_gdeval_cut(measure, qrel_path, run_path):
    if measure == 'gdeval_ndcg_cut':
        kind = 'ndcg'
    elif measure == 'gdeval_err_cut':
        kind = 'err'
    else:
        raise ValueError

    def transform(name):
        return 'gdeval_{}@{}'.format(kind, name.split('@')[1])

    aggregated, qno_results = gdeval_all(qrel_path, run_path)
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


def eval_trec_ndcg_k(measure, qrel_path, run_path):
    k = int(measure.split('@')[1])

    trec_input = 'ndcg_cut.{}'.format(k)
    trec_output = 'ndcg_cut_{}'.format(k)
    aggregated, qno_results = trec_eval(trec_input, qrel_path, run_path)
    aggregated = {measure: aggregated[trec_output]}
    qno_results = {
        qno: {
            measure: ms[trec_output]
        }
        for qno, ms in qno_results.items()
    }

    return aggregated, qno_results


def eval_trec_ndcg_cut(measure, qrel_path, run_path):
    def transform(m):
        return 'trec_' + m.replace('_cut_', '@')

    aggregated, qno_results = trec_eval('ndcg_cut', qrel_path, run_path)
    aggregated = {transform(m): v for m, v in aggregated.items()}
    qno_results = {
        qno: {transform(m): v
              for m, v in ms.items()}
        for qno, ms in qno_results.items()
    }

    return aggregated, qno_results


def eval_trec_map_k(measure, qrel_path, run_path):
    k = int(measure.split('@')[1])
    trec_input = 'map_cut.{}'.format(k)
    trec_output = 'map_cut_{}'.format(k)
    aggregated, qno_results = trec_eval(trec_input, qrel_path, run_path)
    aggregated = {measure: aggregated[trec_output]}
    qno_results = {
        qno: {
            measure: ms[trec_output]
        }
        for qno, ms in qno_results.items()
    }

    return aggregated, qno_results


def eval_trec_map_cut(measure, qrel_path, run_path):
    def transform(m):
        return m.replace('_cut_', '@')

    aggregated, qno_results = trec_eval('map_cut', qrel_path, run_path)
    aggregated = {transform(m): v for m, v in aggregated.items()}
    qno_results = {
        qno: {transform(m): v
              for m, v in ms.items()}
        for qno, ms in qno_results.items()
    }

    return aggregated, qno_results


def eval_trec_default(measure, qrel_path, run_path):
    aggregated, qno_results = trec_eval(measure, qrel_path, run_path)
    return aggregated, qno_results


def eval_trec_general_k(measure, qrel_path, run_path):
    trec_input = measure.replace('@', '.')
    trec_output = measure.replace('@', '_')
    aggregated, qno_results = trec_eval(trec_input, qrel_path, run_path)
    aggregated = {measure: aggregated[trec_output]}
    qno_results = {
        qno: {
            measure: ms[trec_output]
        }
        for qno, ms in qno_results.items()
    }

    return aggregated, qno_results


def eval_trec_general_cut(measure, qrel_path, run_path):
    def transform(m):
        return m.replace('_', '@')

    trec_input = measure.rstrip('_cut')
    aggregated, qno_results = trec_eval(trec_input, qrel_path, run_path)
    aggregated = {transform(m): v for m, v in aggregated.items()}
    qno_results = {
        qno: {transform(m): v
              for m, v in ms.items()}
        for qno, ms in qno_results.items()
    }

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
    EvalEntry('', match_true, eval_trec_default),
]


def eval_run(measure, qrel_path, run_path):
    """Supported measure: All names supported by trec_eval. \"gdeval\" and
    \"gdeval@k\" are also supported but are not official names.
    """
    for entry in functions:
        if entry.match_function(measure, entry.match_str):
            aggregated, qno_results = entry.eval_func(measure, qrel_path,
                                                      run_path)
            return aggregated, qno_results

    raise ValueError('Unrecognizable measure {}'.format(measure))
