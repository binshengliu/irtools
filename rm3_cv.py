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


def cv(queries, shuffle, fold, all_evals, metric):
    per_fold = []
    per_query = {}
    kfold = KFold(n_splits=fold, shuffle=shuffle)
    for ith, (train_index, test_index) in enumerate(kfold.split(queries)):
        train_queries = [queries[i] for i in train_index]
        test_queries = [queries[i] for i in test_index]
        logging.debug('Fold {} test: {}'.format(ith, test_queries))

        param_result = []
        for param_setting, query_evals in all_evals.items():
            train_measure = np.mean(
                [query_evals[metric][query] for query in train_queries])
            param_result.append((param_setting, train_measure))

        best_param, best_train = max(param_result, key=itemgetter(1))
        this_fold = {
            query: all_evals[best_param][metric][query]
            for query in test_queries
        }
        per_query.update({query: best_param for query in test_queries})
        best_test = np.mean(list(this_fold.values()))
        per_fold.append((best_train, best_test, best_param))
        logging.info('Fold {} {} {:.3f}, {:.3f}'.format(
            ith, best_param, best_train, best_test))

    return per_fold, per_query


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
            all_queries.update(per_query[metric].keys())
            result[param] = per_query
            agg = np.mean(list(per_query[metric].values()))
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
    per_fold, per_query = cv(all_queries, args.shuffle, args.folds, all_evals,
                             args.metric)

    fields, _ = zip(*args.params)

    data = [(*params, train, test) for train, test, params in per_fold]
    df = pd.DataFrame(data, columns=fields + ('train', 'test'))
    logging.info('Per fold optimizing {}:\n'.format(args.metric) +
                 df.to_latex(float_format=lambda f: '{:.3f}'.format(f)))

    avail_metrics = list(next(iter(all_evals.values())).keys())

    df = pd.DataFrame()
    df['number'] = all_queries
    for m in avail_metrics:
        df[m] = [all_evals[per_query[q]][m][q] for q in all_queries]
    logging.info('Agg optimizing {}:\n'.format(args.metric) +
                 df[avail_metrics].mean().to_latex(
                     float_format=lambda f: '{:.3f}'.format(f)))


if __name__ == '__main__':
    main()
