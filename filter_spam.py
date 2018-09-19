#!/home/sl8/S3676608/.anaconda3/bin/python3
import sys
import argparse
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Filter spams from run files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--score',
        '-s',
        type=int,
        metavar='[1-100]',
        choices=range(1, 101),
        required=True,
        help='Documents with a score lower than the value will be removed.')

    parser.add_argument(
        '--count',
        '-c',
        type=int,
        help='Number of records per query to keep after filtering. '
        'Default to keep all.')

    parser.add_argument(
        '--output',
        '-o',
        metavar='DIRECTORY',
        type=Path,
        help='Directory to save filtered run files. '
        'Default to the same directory as each input run.')

    parser.add_argument(
        '--force',
        '-f',
        action='store_true',
        help='Overwrite existing filtered run files.')

    parser.add_argument(
        'score_file',
        type=Path,
        metavar='SCORE-FILE',
        help='File containing document scores '
        'in the format of \'percentile-score clueweb-docid\'. '
        'Check https://plg.uwaterloo.ca/~gvcormac/clueweb09spam/ '
        'and https://www.mansci.uwaterloo.ca/~msmucker/cw12spam/')

    parser.add_argument(
        'run', metavar='RUN', nargs='+', type=Path, help='Run files')

    args = parser.parse_args()

    return args


def filter_with_score(run_path, score_dict, threshold, count):
    filtered = []
    qno_count = {}
    for line in run_path.read_text().splitlines():
        qno, _, docno, _, _, _ = line.split()
        if docno not in score_dict:
            continue

        if score_dict[docno] < threshold:
            continue

        if count and qno_count.get(qno, 0) >= count:
            continue

        filtered.append(line)
        qno_count.setdefault(qno, 0)
        qno_count[qno] += 1

    return filtered


def eprint(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr, flush=True)


def output_path(directory, orig):
    if directory:
        return directory.joinpath(
            orig.with_suffix('.filtered' + orig.suffix).name)
    else:
        return orig.with_suffix('.filtered' + orig.suffix)


def filtered_path(directory, orig):
    return directory.joinpath(orig.with_suffix('.filtered' + orig.suffix).name)


def main():
    args = parse_args()

    if not args.force:
        if all([output_path(args.output, run).exists() for run in args.run]):
            eprint('All the files have been filtered before. Do nothing.')
            return

    eprint('Reading {}'.format(args.score_file))
    scores_dict = {}
    for line in args.score_file.read_text().splitlines():
        score, docno = line.split()
        scores_dict[docno] = int(score)

    for run in args.run:
        output_name = output_path(args.output, run)
        if not args.force and output_name.exists():
            eprint('{} already exists.'.format(output_name))
            continue

        filtered = filter_with_score(run, scores_dict, args.score, args.count)

        # A friendly reminder
        if not filtered:
            eprint('Nothing left after filtering {}'.format(run))

        output_name.parent.mkdir(exist_ok=True)
        output_name.write_text('\n'.join(filtered))
        print(output_name)

    eprint('Cleaning up memory.')


if __name__ == '__main__':
    main()
