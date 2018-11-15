#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools


def eprint(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr)


def generate_rmodel(runfile, field, index, docs):
    rmodel = '/research/remote/petabyte/users/'\
        'binsheng/indri-5.12/rmodel/rmodel'
    args = [
        rmodel, '-field={}'.format(field), '-trecrun={}'.format(runfile),
        '-index={}'.format(str(index)), '-documents={}'.format(docs),
        '-format=xml'
    ]

    eprint(' '.join(args))
    proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc.stdout


def rmodels_mp(pool, index, runfile, docs, fields, template):
    future_to_param = {}
    for d, f in itertools.product(docs, fields):
        path = Path(template.format(field=f, docs=d))
        if path.exists():
            continue
        future = pool.submit(generate_rmodel, runfile, f, index, d)
        future_to_param[future] = path

    for f in as_completed(future_to_param):
        path = future_to_param[f]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f.result())
        eprint('{}'.format(path))


def parse_args():
    parser = argparse.ArgumentParser(description='Generate relevance models.')

    def comma_int(s):
        return [int(t) for t in s.split(',')]

    def comma_str(s):
        return [str(t) for t in s.split(',')]

    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--fields', type=comma_str, default=['all'])
    parser.add_argument('--docs', type=comma_int, default=[50])
    parser.add_argument('--template', default='rms/{field}_{docs}.model')

    args, _ = parser.parse_known_args()

    return args


def main_mp(pool):
    args = parse_args()

    rmodels_mp(pool, args.index, args.run, args.docs, args.fields,
               args.template)


def main():
    with ProcessPoolExecutor() as pool:
        main_mp(pool)


if __name__ == '__main__':
    main()
