#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor
import itertools
import lxml.etree as ET
from tempfile import NamedTemporaryFile
from collections import OrderedDict
from copy import deepcopy


def eprint(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr)


def run_rmodel(runfile, index, docs):
    rmodel = '/research/remote/petabyte/users/'\
        'binsheng/indri-5.12/rmodel/rmodel'
    args = [
        rmodel, '-trecrun={}'.format(runfile), '-index={}'.format(str(index)),
        '-documents={}'.format(docs)
    ]

    eprint(' '.join(args))
    proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc.stdout.decode('utf-8')


def pairwise(iterable):
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)


def parse_rmodel(output):
    output = [line.strip() for line in output.splitlines()]
    byquery = [i for i, l in enumerate(output) if l.startswith('#')]
    rms = OrderedDict()
    for begin, end in pairwise(byquery):
        _, _, qno, *_ = output[begin].split()
        rm = output[begin + 1:end]
        rms[qno] = rm
    return rms


def generate_rmodel(runfile, index, docs):
    output = run_rmodel(runfile, index, docs)
    rms = parse_rmodel(output)
    return rms


def rmodels_mp(pool, index, runfile, docs, prefix):
    results = pool.map(generate_rmodel, [runfile] * len(docs),
                       [index] * len(docs), docs)
    allrms = {}
    for doc, rms in zip(docs, results):
        for qno, rm in rms.items():
            allrms.setdefault(doc, {})[qno] = rm
            path = prefix.joinpath('docs_{}'.format(doc), 'qno_{}'.format(qno),
                                   'model.txt')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('\n'.join(rm))
    return allrms


def generate_param(root, rms, docs, terms, origs, prefix):
    for ndoc, nterm, orig in itertools.product(docs, terms, origs):
        copy = deepcopy(root)
        sweep = prefix.joinpath('docs_{}'.format(ndoc),
                                'terms_{}'.format(nterm),
                                'orig_{}'.format(orig), 'query.param')
        for node in copy.findall('query'):
            qno = node.find('number').text
            title = node.find('text')
            rm1 = '#weight({})'.format(' '.join(rms[ndoc][qno][:nterm]))
            if orig == 0.0:
                title.text = rm1
            elif orig == 1.0:
                title.text = title.text
            else:
                title.text = '#weight({} #combine({}) {} {})'.format(
                    orig, title.text, 1 - orig, rm1)
        sweep.parent.mkdir(parents=True, exist_ok=True)
        eprint('{}'.format(sweep))
        sweep.write_text(
            ET.tostring(copy, pretty_print=True, encoding='unicode'))


def load_param(path):
    tree = ET.parse(str(path), ET.XMLParser(remove_blank_text=True))
    root = tree.getroot()
    queries = {}
    for query in root.findall('query'):
        text = query.find('text').text
        number = query.find('number').text
        queries[number] = text.split()
    return queries


def convert_ce_to_ql(param_file, run_file, fp):
    queries = load_param(param_file)
    converted = []
    with open(run_file, 'r') as f:
        for line in f:
            # qno, q0, docno, rank, score, runid = line.split()
            splits = line.split()
            score = float(splits[4]) * len(queries[splits[0]])
            splits[4] = '{}'.format(score)
            converted.append(' '.join(splits) + '\n')

    fp.writelines(converted)


def parse_args():
    parser = argparse.ArgumentParser(description='Generate relevance models.')

    def comma(conv):
        return lambda s: [conv(t) for t in s.split(',')]

    parser.add_argument('--param', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--docs', type=comma(int), default=[50])
    parser.add_argument('--terms', type=comma(int), default=[50])
    parser.add_argument(
        '--origs',
        type=comma(float),
        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    parser.add_argument('--rerank', action='store_true')
    parser.add_argument('--output', default='.', type=Path)

    args, _ = parser.parse_known_args()

    return args


def prepare_param(param, run, rerank):
    root = ET.parse(str(param), ET.XMLParser(remove_blank_text=True)).getroot()
    workingset = {}
    if rerank:
        with open(run, 'r') as f:
            for line in f:
                qno, _, docno, *_ = line.split()
                workingset.setdefault(qno, []).append(docno)
        for node in root.findall('query'):
            qno = node.find('number').text
            for docno in workingset.get(qno, []):
                wsd = ET.SubElement(node, 'workingSetDocno')
                wsd.text = docno
    return root


def main():
    args = parse_args()

    with ProcessPoolExecutor() as pool, NamedTemporaryFile(mode='w') as f:
        convert_ce_to_ql(args.param, args.run, f)
        rms = rmodels_mp(pool, args.index, f.name, args.docs,
                         args.output.joinpath('rms'))
        root = prepare_param(args.param, args.run, args.rerank)
        generate_param(root, rms, args.docs, args.terms, args.origs,
                       args.output.joinpath('sweep'))


if __name__ == '__main__':
    main()
