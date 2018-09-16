#!/usr/bin/env python3
import sys
import subprocess
from pathlib import Path


def eprint(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr, flush=True)


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
    for k in [5, 10, 15, 20, 30, 100, 200, 500, 1000]:
        agg, result = gdeval(k, qrel_path, run_path)
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


def normalize_name(n, measure):
    new_name = n.replace('_', '@') if n.startswith('P_') else n
    new_name = n.replace('_cut_', '@') if n.startswith('ndcg_cut_') else n
    new_name = 'GDEVAL-' + n if measure.startswith('gdeval') else n
    new_name = 'TREC-' + n if measure.startswith('ndcg') else n
    new_name = new_name.upper()
    return new_name


def eval_run(measure, qrel_path, run_path):
    """Supported measure: All names supported by trec_eval. \"gdeval\" and
    \"gdeval@k\" are also supported but are not official names.
    """
    if measure.startswith('gdeval'):
        if '@' in measure:
            k = measure.split('@')[1]
            aggregated, qno_results = gdeval(k, qrel_path, run_path)
        else:
            aggregated, qno_results = gdeval_all(qrel_path, run_path)
    else:
        aggregated, qno_results = trec_eval(measure, qrel_path, run_path)

    updated_name = {}
    for m in list(aggregated.keys()):
        new_name = normalize_name(m, measure)
        aggregated[new_name] = aggregated.pop(m)

    for q in list(qno_results.keys()):
        for m in qno_results[q].keys():
            new_name = normalize_name(m, measure)
            qno_results[q][new_name] = qno_results[q].pop(m)
    return updated_name, qno_results
