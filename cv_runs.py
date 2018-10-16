#!/usr/bin/env python3
import argparse
import os
import subprocess
import logging
import configparser
from pathlib import Path
import itertools
from sklearn.model_selection import KFold
from tempfile import NamedTemporaryFile
from operator import itemgetter
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from eval_run import eval_run


def setup_logging(log_file):
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    logging.basicConfig(
        format='%(asctime)s %(message)s',
        handlers=[file_handler, stream_handler],
        level=logging.DEBUG)


def split_and_convert(l, convert):
    return [convert(d) for d in l.split(',')]


def print_args(args):
    logging.info('# Settings')
    max_len = len(max(vars(args).keys(), key=len))
    for (k, v) in vars(args).items():
        logging.info('{0:{width}}: {1}'.format(k, v, width=max_len + 1))


def split_run(train_queries, test_queries, run_file, train_file, test_file):
    test_lines = []
    for line in run_file:
        try:
            qno, _, _, _, _, _ = line.split()
        except Exception:
            raise Exception('Error unpacking {}'.format(run_file.name))
        if qno in train_queries:
            train_file.write(line)
        elif qno in test_queries:
            test_file.write(line)
            test_lines.append(line)
        else:
            raise ValueError
    return test_lines


def param_to_str(params):
    return ', '.join(['{:>4}'.format(v) for k, v in params])


def trec_eval_mp_wrapper(name, measure, qrel, run_name, train_queries,
                         test_queries, comb):
    with open(run_name, 'r') as run, NamedTemporaryFile(
            mode='w', delete=True) as train_run, NamedTemporaryFile(
                mode='w', delete=True) as test_run:
        test_lines = split_run(train_queries, test_queries, run, train_run,
                               test_run)

        train_measure, _ = eval_run(measure, str(qrel), train_run.name)
        test_measure, _ = eval_run(measure, str(qrel), test_run.name)
        logging.info('{} {:.3f} {:.3f} {}'.format(name, train_measure[measure],
                                                  test_measure[measure], comb))
    return train_measure, test_measure, test_lines, comb


def cv_one_fold(name, train_queries, test_queries, measure, qrel, cv_params,
                cv_run_template, max_workers):
    cv_param_names, cv_param_values = zip(*cv_params)
    with ProcessPoolExecutor(max_workers) as pool:
        futures = []
        for param_values in itertools.product(*cv_param_values):
            comb = list(zip(cv_param_names, param_values))
            run_name = cv_run_template.format(**dict(comb))

            if not os.path.isfile(run_name):
                continue
            future = pool.submit(trec_eval_mp_wrapper, name, measure, qrel,
                                 run_name, train_queries, test_queries, comb)
            futures.append(future)

        train_measure, test_measure, test_lines, comb = max(
            [future.result() for future in futures], key=itemgetter(0))
        logging.info('{} Best: [{}, {:.3f}, {:.3f}]'.format(
            name, param_to_str(comb), train_measure, test_measure))

    return (train_measure, test_measure, test_lines, comb)


def cv(queries, shuffle, fold, measure, qrel, cv_params, cv_run_template,
       testset_output):
    testset = open(testset_output, 'w')
    kfold = KFold(n_splits=fold, shuffle=False)
    fold_info = []

    fold_workers, _ = divmod(len(os.sched_getaffinity(0)), fold)
    logging.info('Start {} processes'.format(fold_workers * fold))
    with ProcessPoolExecutor(max_workers=fold) as pool:
        futures = []
        for ith, (train_index, test_index) in enumerate(kfold.split(queries)):
            train_queries = [queries[i] for i in train_index]
            test_queries = [queries[i] for i in test_index]
            logging.info('Fold {} train: {}'.format(ith, train_queries))
            logging.info('Fold {} test: {}'.format(ith, test_queries))

            future = pool.submit(cv_one_fold, 'Fold {}'.format(ith),
                                 train_queries, test_queries, measure, qrel,
                                 cv_params, cv_run_template, fold_workers)
            futures.append(future)

        for i, future in enumerate(futures):
            train_measure, test_measure, test_lines, params = future.result()
            fold_info.append((train_measure, test_measure, test_lines, params))

            testset.writelines(test_lines)

    return fold_info


def str_to_bool(s):
    return s.lower() in ['true', 'yes', 't', 'y']


def parse_cv_params(s):
    cv_params = s.split(',')
    cv_params = [f.split(':') for f in cv_params]
    cv_params = [(field, values.split('|')) for field, values in cv_params]
    return cv_params


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate Indri parameter files.', add_help=False)
    parser.add_argument('-c', '--conf', default='conf.ini', type=Path)

    args, remaining_argv = parser.parse_known_args()
    directory = args.conf.parent
    defaults = {
        'log': directory.joinpath('log',
                                  Path(__file__).with_suffix('.log').name)
    }

    config = configparser.ConfigParser()
    config.read(args.conf)

    defaults.update(dict(config.items("CV")))

    parser = argparse.ArgumentParser(parents=[parser])
    parser.set_defaults(**defaults)

    def join_dir(s):
        return directory.joinpath(s)

    def join_dir_str(s):
        return str(directory.joinpath(s))

    def split_comma(s):
        return s.split(',')

    parser.add_argument('--cv-params', type=parse_cv_params)
    parser.add_argument('--cv-run-template', type=join_dir_str)
    parser.add_argument('--cv-measure')
    parser.add_argument('--cv-qrel', type=join_dir)
    parser.add_argument('--cv-queries', type=split_comma)
    parser.add_argument('--cv-shuffle', type=str_to_bool)
    parser.add_argument('--cv-folds', type=int)
    parser.add_argument('--cv-testset-name', type=join_dir)
    parser.add_argument('--log', type=join_dir)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    setup_logging(args.log)

    logging.info('# Start cross validation')
    print_args(args)

    Path(args.cv_testset_name).parent.mkdir(parents=True, exist_ok=True)
    fold_info = cv(args.cv_queries, args.cv_shuffle, args.cv_folds,
                   args.cv_measure, args.cv_qrel, args.cv_params,
                   args.cv_run_template, args.cv_testset_name)

    fields, _ = zip(*args.cv_params)
    measure, _ = eval_run(args.cv_measure, args.cv_qrel, args.cv_testset_name)

    data = [(*[p[1] for p in params], train, test)
            for train, test, _, params in fold_info]
    df = pd.DataFrame(data, columns=fields + ('train', 'test'))
    logging.info('\n' + df.to_latex(column_format='l' * len(df.columns)))
    logging.info('Testset measure: {}'.format(measure[args.cv_measure]))


if __name__ == '__main__':
    main()
