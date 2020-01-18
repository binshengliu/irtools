#!/usr/bin/env python3
from sumeval.metrics.rouge import RougeCalculator
from concurrent.futures import ProcessPoolExecutor as Pool
from itertools import product
import numpy as np
import argparse
from tqdm import tqdm


def calculate(summary, reference):
    rouge = RougeCalculator(lang="en")
    scores = []
    for n, alpha in product([1, 2], [1, 0, 0.5]):
        scores.append(
            rouge.rouge_n(summary=summary,
                          references=reference,
                          n=n,
                          alpha=alpha))

    for alpha in [1, 0, 0.5]:
        scores.append(
            rouge.rouge_l(summary=summary, references=reference, alpha=alpha))

    return scores


def parse_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--ref', required=True, type=argparse.FileType('r'))
    parser.add_argument('--hyp', required=True, type=argparse.FileType('r'))
    return parser.parse_args()


def main():
    args = parse_arguments()
    with Pool() as pool:
        result = pool.map(calculate, args.hyp, args.ref)
        result = list(tqdm(result))

    print("ROUGE-1 P: {}".format(np.mean([x[0] for x in result])))
    print("ROUGE-1 R: {}".format(np.mean([x[1] for x in result])))
    print("ROUGE-1 F: {}".format(np.mean([x[2] for x in result])))

    print("ROUGE-2 P: {}".format(np.mean([x[3] for x in result])))
    print("ROUGE-2 R: {}".format(np.mean([x[4] for x in result])))
    print("ROUGE-2 F: {}".format(np.mean([x[5] for x in result])))

    print("ROUGE-L P: {}".format(np.mean([x[6] for x in result])))
    print("ROUGE-L R: {}".format(np.mean([x[7] for x in result])))
    print("ROUGE-L F: {}".format(np.mean([x[8] for x in result])))

    # print("ROUGE-L P: {}".format(np.mean([x[9] for x in result])))
    # print("ROUGE-L R: {}".format(np.mean([x[10] for x in result])))
    # print("ROUGE-L F: {}".format(np.mean([x[11] for x in result])))


if __name__ == '__main__':
    main()
