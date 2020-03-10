#!/usr/bin/env python3
import sys
import argparse
import os
from pathlib import Path
from multiprocessing import Pool, Manager, current_process


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


def prev_boundary(filename, pos):
    if pos == 0:
        return 0

    with open(filename, 'r') as f:
        while pos > 0:
            pos -= 1
            f.seek(pos)
            if f.read(1) == os.linesep:
                return f.tell()

    return 0


def next_boundary(filename, pos):
    with open(filename, 'r') as f:
        while True:
            f.seek(pos)
            c = f.read(1)
            if not c or c == os.linesep:
                return f.tell()
            pos += 1

    return 0


def read_at_pos(filename, start, length, queue, docnos):
    filesize = os.stat(filename).st_size
    last = False
    if start + length >= filesize:
        last = True

    start = prev_boundary(filename, start)
    length = prev_boundary(filename,
                           start + length) - start if not last else None
    pname = current_process().name

    with open(filename, 'r') as f:
        f.seek(start)
        if length:
            content = f.read(length)
        else:
            content = f.read()

    scores_dict = {}
    for line in content.splitlines():
        try:
            score, docno = line.split()
        except Exception:
            eprint(line)
            raise
        if docno in docnos:
            scores_dict[docno] = int(score)

    queue.put((pname, start, scores_dict))
    return


def get_docnos(runlist):
    s = set()
    for run in runlist:
        with open(run, 'r') as f:
            for line in f:
                s.add(line.split()[2])
    return s


def read_file_mp(filename, runlist):
    size = os.stat(filename).st_size
    processes = len(os.sched_getaffinity(0)) - 1
    chunk_size, residual = divmod(size, processes)
    if residual:
        chunk_size += 1
    start = 0
    param = []
    docnos = get_docnos(runlist)
    manager = Manager()
    queue = manager.Queue()
    while start < size:
        if start + chunk_size <= size:
            param.append((filename, start, chunk_size, queue, docnos))
            start += chunk_size
        else:
            param.append((filename, start, size - start, queue, docnos))
            start = size

    scores_dict = {}
    eprint('Start {} processes.'.format(processes))
    eprint('Read {}'.format(filename), end='\r')
    with Pool(processes=processes) as pool:
        ret = pool.starmap_async(read_at_pos, param, 1)
        for i, _ in enumerate(range(len(param))):
            _, _, d = queue.get()
            scores_dict.update(d)
            eprint(
                'Read {} {}/{} chunks'.format(filename, i + 1, len(param)),
                end='\r')
        ret.wait()
        eprint()

    return scores_dict


def main():
    args = parse_args()

    if not args.force:
        if all([output_path(args.output, run).exists() for run in args.run]):
            eprint('All the files have been filtered before. Do nothing.')
            return

    scores_dict = read_file_mp(args.score_file, args.run)

    eprint('Filtering')
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
