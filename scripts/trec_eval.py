#!/usr/bin/env python3
import argparse

from irtools import TrecQrels, TrecRun


def parse_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-q', action='store_true')
    parser.add_argument(
        '-m', required=True, choices=['map', 'recip_rank', 'ndcg'])
    parser.add_argument('-k', type=int)
    parser.add_argument('-j', type=int)
    parser.add_argument('qrels')
    parser.add_argument('run')

    return parser.parse_args()


def main():
    args = parse_arguments()
    qrels = TrecQrels(args.qrels, depth=args.k, method=args.m)
    run = TrecRun.from_file(args.run)
    reduction = None if args.q else 'mean'
    result = run.eval(qrels, reduction=reduction, progress_bar=True)
    if args.q:
        result = sorted(result.items())
        for one in result:
            print('{}\t{}\t{:.4f}'.format(args.m, *one))
    else:
        print('{}\tall\t{:.4f}'.format(args.m, result))


if __name__ == '__main__':
    main()
