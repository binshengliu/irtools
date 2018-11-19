#!/usr/bin/env python3
import argparse
import subprocess
import logging
import configparser
from pathlib import Path
import pytrec_eval
import lxml.etree as ET
import itertools
import sys
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed


def eprint(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr)


def load_model(path, normalize=False):
    root = ET.parse(str(path)).getroot()
    model = {}
    for node in root.findall('model'):
        qno = node.get('query')
        expansions = node.text.strip().splitlines()
        expansions = [plain.split() for plain in expansions]
        expansions = [(term, float(prob)) for prob, term in expansions]
        if normalize:
            total = sum([prob for _, prob in expansions])
            expansions = [(term, prob / total) for term, prob in expansions]
        if expansions:
            model[qno] = expansions

    return model


def sum_to_number(candidates, n, target):
    if n == 1:
        return [[target]] if target in candidates else []

    ans = []
    for current in candidates:
        sub_ans = sum_to_number(candidates, n - 1, target - current)
        ans.extend([[current] + a for a in sub_ans])
    return ans


def weight_combinations(fields):
    combinations = sum_to_number(range(11), len(fields), 10)
    for i, field in enumerate(fields):
        if ':' in field:
            field, selections = field.split(':')
            selections = [int(float(s) * 10) for s in selections.split('|')]
        else:
            field = field
            selections = None
        if selections:
            combinations = [c for c in combinations if c[i] in selections]
    combinations = [[float(w) / 10.0 for w in c] for c in combinations]
    return combinations


def expand_query(original_query, orig_field, wt_orig, expansion):
    origin = original_query
    if orig_field and orig_field != 'all':
        origin = ['{}.({})'.format(w, orig_field) for w in origin.split()]
        origin = ' '.join(origin)
    template = '#weight({:.1f} #combine({}) '\
        '{:.1f} {})'
    return template.format(wt_orig, origin, 1 - wt_orig, expansion)


def add_working_set(tree, runfile):
    with open(runfile, 'r') as f:
        run = pytrec_eval.parse_run(f.readlines())

    root = tree.getroot()
    for query in root.findall('query'):
        number = query.find('number').text
        for docno in run[number].keys():
            working_set_node = ET.SubElement(query, 'workingSetDocno')
            working_set_node.text = docno

    return tree


def generate_rmodel(runfile, field, index, docs):
    rmodel = '/research/remote/petabyte/users/'\
        'binsheng/indri-5.12/rmodel/rmodel'
    args = [
        rmodel, '-field={}'.format(field), '-trecrun={}'.format(runfile),
        '-index={}'.format(str(index)), '-documents={}'.format(docs),
        '-format=xml'
    ]

    logging.info(' '.join(args))
    proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc.stdout


def prepare_base_param(ql_param, condense_run):
    tree = ET.parse(str(ql_param), ET.XMLParser(remove_blank_text=True))
    if condense_run and condense_run.is_file():
        add_working_set(tree, condense_run)
    return tree.getroot()


def get_expansion_terms(rm, score_field, qno, terms):
    if qno not in rm:
        return ''
    if score_field == 'all':
        expansion = ['{:f} {}'.format(prob, t) for t, prob in rm[qno][:terms]]
    else:
        expansion = [
            '{:f} {}.({})'.format(prob, t, score_field)
            for t, prob in rm[qno][:terms]
        ]

    expansion = ' '.join(expansion)
    expansion = '#weight({})'.format(expansion)

    return expansion


def update_or_add_node(root, name, value):
    node = root.find(name)
    if node is None:
        node = ET.Element(name)
        root.find('query').addprevious(node)
    node.text = value


def remove_all_node(root, name):
    for node in root.findall(name):
        root.remove(node)


def sweep_params(baseparam, rm_induce_field, rm_score_field, rm_docs, rm_terms,
                 rm_orig_weights, rm_param_template, rerank_count, rerank_mu,
                 rmodels):

    for (d, t, o, mu) in itertools.product(rm_docs, rm_terms, rm_orig_weights,
                                           rerank_mu):
        rerank_param_name = rm_param_template.format(
            docs=d,
            terms=t,
            orig=o,
            rm_induce_field=rm_induce_field,
            rm_score_field=rm_score_field,
            rerank_mu=mu)
        rerank_param_name = Path(rerank_param_name)
        if rerank_param_name.exists():
            logging.info('{} exists'.format(rerank_param_name))
            continue

        root = deepcopy(baseparam)
        remove_all_node(root, 'threads')
        update_or_add_node(root, 'rule', 'method:dirichlet,mu:{}'.format(mu))
        update_or_add_node(root, 'count', str(rerank_count))

        for node in root.findall('query'):
            qno = node.find('number').text
            expansion = get_expansion_terms(rmodels[d][rm_induce_field],
                                            rm_score_field, qno, t)
            if not expansion.strip():
                continue
            original_query = node.find('text').text
            node.find('text').text = expand_query(original_query,
                                                  rm_score_field, o, expansion)
        rerank_param_name.parent.mkdir(parents=True, exist_ok=True)
        rerank_param_name.write_text(
            ET.tostring(root, pretty_print=True, encoding='unicode'))

        logging.info('{}'.format(rerank_param_name))


def setup_logging(log_file):
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    logging.basicConfig(
        format='%(asctime)s %(message)s',
        handlers=[file_handler, stream_handler],
        level=logging.DEBUG)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Sweep relevance modeling.', add_help=False)
    parser.add_argument('-c', '--conf', default='conf.ini', type=Path)

    args, remaining_argv = parser.parse_known_args()
    directory = args.conf.parent
    defaults = {
        'rm_rerank_mu': 2500,
        'rm_rerank_count': 1000,
        'log': directory.joinpath('log',
                                  Path(__file__).with_suffix('.log').name)
    }

    config = configparser.ConfigParser()
    config.read([directory.joinpath(args.conf).resolve()])
    defaults.update(dict(config.items("RUN")))

    parser = argparse.ArgumentParser(parents=[parser])
    parser.set_defaults(**defaults)

    def join_dir(s):
        return directory.joinpath(s)

    def join_dir_str(s):
        return str(directory.joinpath(s))

    def comma_int(s):
        return [int(t) for t in s.split(',')]

    def comma_float(s):
        return [float(t) for t in s.split(',')]

    def string_to_bool(s):
        return True if s.lower() in ['true', 'yes', 't', 'y'] else False

    parser.add_argument('--ql-param', type=join_dir)
    parser.add_argument('--condense-run', type=Path)
    parser.add_argument('--rm-rerank-count', type=int)
    parser.add_argument('--rm-rerank-mu', type=comma_int)
    parser.add_argument('--rm-induce-field')
    parser.add_argument('--rm-score-field')
    parser.add_argument('--rm-docs', type=comma_int)
    parser.add_argument('--rm-terms', type=comma_int)
    parser.add_argument('--rm-orig-weights', type=comma_float)
    parser.add_argument('--rm-template', type=join_dir_str)
    parser.add_argument('--log', type=join_dir)

    args, _ = parser.parse_known_args()

    return args


def print_args(args):
    max_len = len(max(vars(args).keys(), key=len))
    for (k, v) in vars(args).items():
        logging.info('{0:{width}}: {1}'.format(k, v, width=max_len + 1))


def load_rmodels_mp(pool, docs, field, template):
    future_to_param = {}
    for d in docs:
        path = Path(template.format(field=field, docs=d))
        future = pool.submit(load_model, path)
        future_to_param[future] = d

    rmodels = {}
    for f in as_completed(future_to_param):
        d = future_to_param[f]
        rmodels.setdefault(d, {})
        rmodels[d][field] = f.result()

    return rmodels


def main():
    with ProcessPoolExecutor() as pool:
        main_mp(pool)


def main_mp(pool):
    args = parse_args()

    setup_logging(args.log)

    print_args(args)

    rmodels = load_rmodels_mp(pool, args.rm_docs, args.rm_induce_field,
                              args.rm_template)
    base_param = prepare_base_param(args.ql_param, args.condense_run)

    sweep_params(base_param, args.rm_induce_field, args.rm_score_field,
                 args.rm_docs, args.rm_terms, args.rm_orig_weights,
                 args.rm_param_template, args.rm_rerank_count,
                 args.rm_rerank_mu, rmodels)


if __name__ == '__main__':
    main()
