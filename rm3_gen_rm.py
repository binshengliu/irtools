#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import lxml.etree as ET
from tempfile import NamedTemporaryFile


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


def load_param(path):
    tree = ET.parse(str(path), ET.XMLParser(remove_blank_text=True))
    root = tree.getroot()
    length_dict = {}
    for query in root.findall('query'):
        text = query.find('text').text
        number = query.find('number').text
        length = len(text.split())
        length_dict[number] = length
    return length_dict


def convert_ce_to_ql(param_file, run_file):
    length_dict = load_param(param_file)
    converted = []
    with open(run_file, 'r') as f:
        for line in f:
            # qno, q0, docno, rank, score, runid = line.split()
            splits = line.split()
            score = float(splits[4]) * length_dict[splits[0]]
            splits[4] = '{}'.format(score)
            converted.append(' '.join(splits) + '\n')

    return converted


def parse_args():
    parser = argparse.ArgumentParser(description='Generate relevance models.')

    def comma_int(s):
        return [int(t) for t in s.split(',')]

    def comma_str(s):
        return [str(t) for t in s.split(',')]

    parser.add_argument('--param', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--fields', type=comma_str, default=['all'])
    parser.add_argument('--docs', type=comma_int, default=[50])
    parser.add_argument('--template', default='rms/{field}_{docs}.model')

    args, _ = parser.parse_known_args()

    return args


def main():
    args = parse_args()

    with ProcessPoolExecutor() as pool, NamedTemporaryFile(mode='w') as f:
        converted = convert_ce_to_ql(args.param, args.run)
        f.writelines(converted)
        rmodels_mp(pool, args.index, f.name, args.docs, args.fields,
                   args.template)


if __name__ == '__main__':
    main()
