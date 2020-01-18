#!/usr/bin/env python3
from concurrent.futures import ProcessPoolExecutor as Pool
from tqdm import tqdm
import numpy as np
import argparse
import rouge


def prepare_results(m, p, r, f):
    return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(
        m, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)


def docalc(hyp, ref):
    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                            max_n=2,
                            apply_avg=True,
                            apply_best=False,
                            limit_length=False,
                            alpha=0.5,
                            weight_factor=1.2,
                            stemming=True)

    scores = evaluator.get_scores(hyp, ref)
    scores = {k: v['p'] for k, v in scores.items()}

    return scores


def parse_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--ref', required=True, type=argparse.FileType('r'))
    parser.add_argument('--hyp', required=True, type=argparse.FileType('r'))
    return parser.parse_args()


def main():
    args = parse_arguments()
    with Pool() as pool:
        result = pool.map(docalc, args.hyp, args.ref)
        result = list(tqdm(result))
    metrics = {}
    for x in result:
        for m, v in x.items():
            metrics.setdefault(m, [])
            metrics[m].append(v)
    metrics = [(k, np.mean(v)) for k, v in metrics.items()]
    print(metrics)


if __name__ == '__main__':
    main()
