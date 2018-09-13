#!/home/sl8/S3676608/.anaconda3/bin/python3
import sys
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Filter spams from run files')
    parser.add_argument(
        '--score',
        '-s',
        type=int,
        metavar='[1-100]',
        choices=range(1, 101),
        required=True,
        help='Score of spam to filter out')

    parser.add_argument(
        '--run',
        '-r',
        metavar='FILE',
        nargs='+',
        type=argparse.FileType('r'),
        help='Run files')

    parser.add_argument(
        '--count',
        '-c',
        type=int,
        required=True,
        help='Number of results to keep after filtering')

    parser.add_argument(
        '--suffix',
        default='s{score}_c{count}_filtered',
        help='init.run -> init.s50_c1000_filtered.run')

    args = parser.parse_args()

    return args


def filter_with_score(run_lines, score_dict, threshold, count):
    filtered = []
    for line in run_lines:
        qno, _, docno, _, _, _ = line.split()
        if docno not in score_dict:
            eprint('Warning: {} score not found.'.format(docno))
            continue

        if score_dict[docno] >= threshold:
            filtered.append(line)

    trimmed = []
    if count:
        qno_count = {}
        for line in filtered:
            qno, _, docno, _, _, _ = line.split()

            if qno_count.get(qno, 0) >= count:
                continue

            trimmed.append(line)
            qno_count.setdefault(qno, 0)
            qno_count[qno] += 1
    else:
        trimmed = filtered

    return trimmed


def eprint(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr)


def main():
    args = parse_args()
    spam_filter = '/research/remote/petabyte/users/' \
        'binsheng/CW09B_Spam_Score/ClueWeb09B_Spam_Fusion.txt'

    eprint('Reading CW09B scores {}'.format(spam_filter))
    scores_dict = {}
    with open(spam_filter, 'r') as f:
        for line in f:
            score, docno = line.split()
            scores_dict[docno] = int(score)

    for run in args.run:
        run_lines = run.readlines()
        eprint('Filtering {} with score {}'.format(run.name, args.score))
        filtered = filter_with_score(run_lines, scores_dict, args.score,
                                     args.count)
        if not filtered:
            eprint('{} nothing left'.format(args.score))
            continue

        suffix = args.suffix.format(score=args.score, count=args.count)
        output_name = '{0}.{2}{1}'.format(*os.path.splitext(run.name), suffix)

        with open(output_name, 'w') as f:
            f.writelines(filtered)
        eprint('Saved at {}'.format(output_name))

    eprint('Cleaning up memory.')


if __name__ == '__main__':
    main()
