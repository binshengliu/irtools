#!/usr/bin/env python3
from eval_run import eval_run
import sys
import argparse
from operator import itemgetter


def eprint(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Sort run files based on measure.')

    parser.add_argument(
        '--measure',
        required=True,
        help='Measure. For trec ndcg, use ndcg_cut.20; '
        'for gdeval ndcg, use gdeval@20 ...')

    parser.add_argument('--threshold', default=0.1, type=float)

    parser.add_argument('qrel', metavar='QREL', help='qrel')

    parser.add_argument('run1')

    parser.add_argument('run2')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    _, result1 = eval_run('gdeval@20', args.qrel, args.run1)
    result1 = {qno: m['GDEVAL-NDCG@20'] for qno, m in result1.items()}

    sort = sorted(result1.items(), key=itemgetter(1))
    sort = [q for q, _ in sort[:50]]

    _, result2 = eval_run('gdeval@20', args.qrel, args.run2)
    result2 = {qno: m['GDEVAL-NDCG@20'] for qno, m in result2.items()}

    queries = sort
    # queries = set(result1.keys()) | set(result2.keys())
    win = 0
    tie = 0
    loss = 0
    for qno in queries:
        s1 = result1.get(qno, 0.0)
        s2 = result2.get(qno, 0.0)
        if s2 > s1 and (s2 - s1) > s1 * args.threshold:
            win += 1
        elif s2 < s1 and (s1 - s2) > s1 * args.threshold:
            loss += 1
        else:
            tie += 1

    print('win,tie,loss')
    print('{},{},{}'.format(win, tie, loss))


if __name__ == '__main__':
    main()
