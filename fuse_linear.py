#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path
from operator import itemgetter
from concurrent.futures import ProcessPoolExecutor
import os


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

    parser.add_argument(
        'run',
        type=Path,
        nargs='+',
        metavar='RUN',
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--weight',
        '-w',
        type=float_comma_list,
        metavar='WEIGHT',
    )

    group.add_argument('--sweep', '-s', action='store_true')

    args = parser.parse_args()
    if args.weight is not None and len(args.weight) != len(args.run):
        parser.error('Please specify the same number of weights as runs')

    return args


def fuse(rw_list, output):
    rw_list = list(rw_list)
    qno_scores = {}
    qno_min_scores = {}
    for run, _ in rw_list:
        for line in run.read_text().splitlines():
            run_str = str(run)
            qno, _, docno, _, score, _ = line.split()
            score = float(score)
            qno_scores.setdefault(qno, {}).setdefault(run_str, {})
            qno_scores[qno][run_str][docno] = score
            qno_min_scores.setdefault(qno, {}).setdefault(run_str, score)
            qno_min_scores[qno][run_str] = min(qno_min_scores[qno][run_str],
                                               score)

    weight = dict([(str(r), w) for r, w in rw_list])
    fused_scores = {}
    for qno, run_scores in qno_scores.items():
        # Find all the documents for qno
        all_docs = set()
        for run, doc_scores in run_scores.items():
            all_docs.update(doc_scores.keys())

        fused_scores.setdefault(qno, {})
        # Calculate a score for each doc
        for doc in all_docs:
            score = 0
            for run, doc_scores in run_scores.items():
                if doc in doc_scores:
                    score += doc_scores[doc] * weight[run]
                else:
                    score += qno_min_scores[qno][run] * weight[run]

            fused_scores[qno][doc] = score

    fused_scores = sorted(
        [(qno, sorted(doc_scores.items(), key=itemgetter(1), reverse=True))
         for qno, doc_scores in fused_scores.items()],
        key=lambda qs: float(qs[0]))

    current_rank = {}
    lines = []
    for qno, rank_list in fused_scores:
        for docno, score in rank_list:
            current_rank.setdefault(qno, 1)
            lines.append('{qno} Q0 {docno} {rank} {score:.5f} linear\n'.format(
                qno=qno, docno=docno, rank=current_rank[qno], score=score))
            current_rank[qno] += 1

    if output == '-':
        sys.stdout.writelines(lines)
    else:
        output.write_text(''.join(lines))
        eprint('{}'.format(output))


def sum_to_number(candidates, n, target):
    if n == 1:
        return [[target]] if target in candidates else []

    ans = []
    for current in candidates:
        sub_ans = sum_to_number(candidates, n - 1, target - current)
        ans.extend([[current] + a for a in sub_ans])
    return ans


def main():
    args = parse_args()

    if args.sweep:
        fuse_args = []
        fuse_output = []
        for wts in sum_to_number(range(0, 11), len(args.run), 10):
            wts = [float(w) / 10.0 for w in wts]
            fuse_args.append(list(zip(args.run, wts)))

            output = Path('_'.join(str(w) for w in wts) + '.run')
            fuse_output.append(output)

        processes = min(len(os.sched_getaffinity(0)) - 1, len(fuse_args))
        with ProcessPoolExecutor(max_workers=processes) as executor:
            executor.map(fuse, fuse_args, fuse_output)
    else:
        fuse(zip(args.run, args.weight), '-')


if __name__ == '__main__':
    main()
