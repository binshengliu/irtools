#!/usr/bin/env python3
import argparse
import os
import logging
import configparser
from pathlib import Path
import itertools
from sklearn.model_selection import KFold
from operator import itemgetter
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from eval_run import eval_run
import numpy as np
import sys
import ast


def eprint(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr, flush=True)


def setup_logging():
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    logging.basicConfig(
        format='%(asctime)s %(message)s',
        handlers=[stream_handler],
        level=logging.DEBUG)


def split_and_convert(l, convert):
    return [convert(d) for d in l.split(',')]


def print_args(args):
    logging.info('# Settings')
    max_len = len(max(vars(args).keys(), key=len))
    for (k, v) in vars(args).items():
        logging.info('{0:{width}}: {1}'.format(k, v, width=max_len + 1))


def param_to_str(params):
    return ', '.join(['{:>4}'.format(v) for k, v in params])


def cv(queries, shuffle, fold, all_evals):
    per_fold = []
    per_query = {}
    kfold = KFold(n_splits=fold, shuffle=shuffle)
    for ith, (train_index, test_index) in enumerate(kfold.split(queries)):
        train_queries = [queries[i] for i in train_index]
        test_queries = [queries[i] for i in test_index]
        logging.debug('Fold {} test: {}'.format(ith, test_queries))

        param_result = []
        for param_setting, query_evals in all_evals.items():
            train_measure = np.mean([
                value for query, value in query_evals.items()
                if query in train_queries
            ])
            param_result.append((param_setting, train_measure))

        best_param, best_train = max(param_result, key=itemgetter(1))
        this_fold = {
            query: value
            for query, value in all_evals[best_param].items()
            if query in test_queries
        }
        per_query.update({
            query: (value, *best_param)
            for query, value in all_evals[best_param].items()
            if query in test_queries
        })
        this_fold = np.mean(list(this_fold.values()))
        per_fold.append((best_train, this_fold, best_param))
        logging.info('Fold {} {} {:.3f}, {:.3f}'.format(
            ith, best_param, best_train, this_fold))

    agg = np.mean([pq[0] for pq in per_query.values()])
    return agg, per_fold, per_query


def parse_args():
    parser = argparse.ArgumentParser(
        description='General cross validation framework.', add_help=False)

    def str_to_bool(s):
        return s.lower() in ['true', 'yes', 't', 'y']

    def parse_cv_params(s):
        params = ast.literal_eval(s)
        params = list(params.items())
        return params

    parser.add_argument('--params', type=parse_cv_params)
    parser.add_argument('--template', type=str, required=True)
    parser.add_argument('--metric', type=str, required=True)
    parser.add_argument('--shuffle', type=str_to_bool, default=False)
    parser.add_argument('--folds', type=int, default=5)

    args = parser.parse_args()

    return args


def load_eval(csv, metric):
    df = pd.read_csv(csv, index_col=0)
    per_query = df.to_dict()
    per_query = per_query[metric]
    return per_query


def load_all_evals(params, eval_template, metric):
    workers = len(os.sched_getaffinity(0))
    param_names, param_values = zip(*params)
    result = {}
    all_queries = set()
    with ProcessPoolExecutor(workers) as pool:
        future_to_param = {}
        for setting in itertools.product(*param_values):
            comb = list(zip(param_names, setting))
            eval_name = Path(eval_template.format(**dict(comb)))
            if not eval_name.exists():
                logging.warn('{} does not exist'.format(eval_name))
                continue
            future = pool.submit(load_eval, eval_name, metric)
            future_to_param[future] = setting

        for f in as_completed(future_to_param):
            param = future_to_param[f]
            per_query = f.result()
            all_queries.update(per_query.keys())
            result[param] = per_query
            agg = np.mean(list(per_query.values()))
            logging.info('{}: {:.3f}'.format(param, agg))

    try:
        all_queries = sorted(list(all_queries), key=int)
    except Exception:
        all_queries = sorted(list(all_queries))
    return result, all_queries


def main():
    args = parse_args()

    setup_logging()

    logging.info('# Start cross validation')
    print_args(args)

    all_evals, all_queries = load_all_evals(args.params, args.template,
                                            args.metric)
    test_measure, per_fold, per_query = cv(all_queries, args.shuffle,
                                           args.folds, all_evals)

    fields, _ = zip(*args.params)

    data = [(*params, train, test) for train, test, params in per_fold]
    df = pd.DataFrame(data, columns=fields + ('train', 'test'))
    logging.info('\n' + df.to_latex(float_format=lambda f: '{:.3f}'.format(f)))
    logging.info('Agg measure: {:.3f}'.format(test_measure))

    df = pd.DataFrame([(k, *v) for k, v in per_query.items()],
                      columns=('number', 'measure',
                               *[r[0] for r in args.params]))
    # args.cv_per_query.write_text(df.to_csv(index=False))


if __name__ == '__main__':
    main()
