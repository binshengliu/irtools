#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path
from operator import itemgetter


def eprint(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr, flush=True)


def parse_file_weight(s):
    if ':' in s:
        f, w = s.split(':')
    else:
        f = s
        w = 1
    return (f, float(w))


def float_comma_list(s):
    return [float(w) for w in s.split(',')]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Filter spams from run files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument

    parser.add_argument(
        'run',
        type=Path,
        nargs='+',
        metavar='RUN',
    )

    parser.add_argument(
        '--weight',
        '-w',
        type=float_comma_list,
        required=True,
        metavar='WEIGHT',
    )

    args = parser.parse_args()
    if len(args.weight) != len(args.run):
        parser.error('Please specify the same number of weights as runs')

    return args


def fuse(run_weight_list, output_fd):
    qno_scores = {}
    for (run, weight) in run_weight_list:
        for line in run.read_text().splitlines():
            qno, _, docno, _, score, _ = line.split()
            qno_scores.setdefault(qno, {}).setdefault(docno, 0)
            qno_scores[qno][docno] += weight * float(score)

    qno_scores = sorted(
        [(qno, sorted(doc_scores.items(), key=itemgetter(1)))
         for qno, doc_scores in qno_scores.items()],
        key=lambda qs: float(qs[0]))

    current_rank = {}
    lines = []
    for qno, rank_list in qno_scores:
        for docno, score in rank_list:
            current_rank.setdefault(qno, 1)
            lines.append('{qno} Q0 {docno} {rank} {score:.5f} linear\n'.format(
                qno=qno, docno=docno, rank=current_rank[qno], score=score))
            current_rank[qno] += 1

    output_fd.writelines(lines)


def main():
    args = parse_args()

    fuse(zip(args.run, args.weight), sys.stdout)


if __name__ == '__main__':
    main()
