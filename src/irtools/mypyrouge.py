#!/usr/bin/env python3
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import argparse
import rouge

# pip install py-rouge

# for rouge-s
# https://github.com/neural-dialogue-metrics/rouge


def prepare_results(m, p, r, f):
    return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(
        m, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)


def docalc(args):
    hyp, ref = args
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


def docalc2(args):
    hyp, ref = args
    hyp = hyp.split()
    ref = ref.split()
    scores = {'rouge-s': rouge.rouge_s_sentence_level(hyp, ref).precision,
              'rouge-w': rouge.rouge_w_sentence_level(hyp, ref).precision,
              'rouge-l': rouge.rouge_l_sentence_level(hyp, ref).precision,
              'rouge-1': rouge.rouge_n_sentence_level(hyp, ref, 1).precision,
              'rouge-2': rouge.rouge_n_sentence_level(hyp, ref, 2).precision,
              'rouge-3': rouge.rouge_n_sentence_level(hyp, ref, 3).precision,
              'rouge-4': rouge.rouge_n_sentence_level(hyp, ref, 4).precision}

    return scores


def parse_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--ref', required=True, type=argparse.FileType('r'))
    parser.add_argument('--hyp', required=True, type=argparse.FileType('r'))
    return parser.parse_args()


def rouge_pair(hyp, ref):
    hyp = list(hyp)
    ref = list(ref)
    with Pool() as pool:
        result = pool.imap(docalc, zip(hyp, ref))
        result = list(tqdm(result, total=len(hyp), desc='Rouge'))
    metrics = {}
    for x in result:
        for m, v in x.items():
            metrics.setdefault(m, [])
            metrics[m].append(v)
    return metrics


def main():
    args = parse_arguments()
    metrics = rouge_pair(args.hyp, args.ref)
    metrics = [(k, np.mean(v)) for k, v in metrics.items()]
    print(metrics)


if __name__ == '__main__':
    main()
