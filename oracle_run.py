#!/home/sl8/S3676608/.anaconda3/bin/python3
from operator import itemgetter
from pathlib import Path
import argparse
import os


def oracle_run(qrel_path, min_rel=1, score_type='uniform'):
    filtered = []
    for line in qrel_path.read_text().splitlines():
        qno, _, docno, rel = line.split()
        if int(rel) >= min_rel:
            filtered.append((qno, docno, int(rel)))

    # Order by relevance
    filtered.sort(key=itemgetter(2), reverse=True)
    try:
        filtered.sort(key=lambda a: int(a[0]))
    except Exception:
        filtered.sort(key=itemgetter(0))

    formated = []
    ranking = {}
    for qno, docno, rel in filtered:
        ranking.setdefault(qno, 1)
        if score_type == 'uniform':
            score = 0
        elif score_type == 'relevance':
            score = rel
        line = '{} Q0 {} {} {} oracle'.format(qno, docno, ranking[qno], score)
        ranking[qno] += 1
        formated.append(line)

    return formated


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert a qrels file into an oracle run file.')

    parser.add_argument(
        'qrels',
        type=Path,
        help='Documents with a score lower than the value will be removed.')

    parser.add_argument(
        '--min-relevance',
        '-r',
        type=int,
        default=1,
        help='Minimum relevance score.')

    parser.add_argument(
        '--score-type',
        '-s',
        default='uniform',
        choices=['uniform', 'relevance'],
        help='How to score documents.')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    formated = oracle_run(args.qrels, args.min_relevance, args.score_type)
    print(os.linesep.join(formated))


if __name__ == '__main__':
    main()
