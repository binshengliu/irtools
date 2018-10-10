#!/usr/bin/env python3
from pathlib import Path
import seaborn as sns
import pandas as pd
from eval_run import eval_run
import sys
import argparse
from matplotlib import pyplot as plt
plt.switch_backend('agg')

qrels = Path(
    '/research/remote/petabyte/users/binsheng/clueweb09b-rm/cw09b.qrels')


def eprint(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr, flush=True)


def split_comma(s):
    return s.split(',')


def eval_wrapper(measure, qrel, run):
    try:
        return eval_run(measure, qrel, run)
    except Exception:
        eprint('Error: {}'.format(run))


def point_vs_box(pointdata, boxdata, name):
    pointdf = pd.DataFrame(pointdata, columns=['Query', 'Measure'])
    order = pointdf.sort_values(by='Measure', ascending=False)['Query']
    print(pointdf.head())
    g = sns.catplot(
        kind='point',
        data=pointdf,
        x='Query',
        y='Measure',
        order=order,
        markers='.',
        join=False,
        scale=0.3,
    )

    boxdf = pd.DataFrame(boxdata, columns=['Query', 'Measure'])
    sns.boxplot(
        x='Query',
        y='Measure',
        data=boxdf,
        order=order,
        ax=g.ax,
        # scale=0.2,
        # join=False,
        linewidth=0.3,
        fliersize=0.3,
        color='#bdc3c7')

    g = g.set_titles('Regular RM3 vs Shard RM3 Range')
    g.set_xticklabels([])
    g.set_xlabels('Query')
    g.set(xticks=[])
    g.despine()
    g.savefig(name)


def point_vs_point(title, point1data, names, markers, filename):
    dfs = [pd.DataFrame(d, columns=['Query', 'Measure']) for d in point1data]
    order = dfs[0].sort_values(by='Measure', ascending=False)['Query']
    df = pd.concat(
        dfs, keys=names, names=['System']).reset_index(level='System')
    g = sns.catplot(
        kind='point',
        data=df,
        x='Query',
        y='Measure',
        hue='System',
        order=order,
        markers=markers,
        palette=sns.color_palette('Set1'),
        join=False,
        scale=0.4,
        legend_out=False,
    )

    # g.set_titles(title)
    g.set_xticklabels([])
    g.set_xlabels('Query')
    g.set(xticks=[])
    g.despine()
    g.savefig(filename)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Sort run files based on measure.')

    parser.add_argument(
        '--measure',
        required=True,
        help='Measure. For trec ndcg, use ndcg_cut.20; '
        'for gdeval ndcg, use gdeval@20 ...')

    parser.add_argument('--names', required=True, type=split_comma)
    parser.add_argument('--output', default='output.pdf')

    parser.add_argument('qrel', metavar='QREL', help='qrel')

    parser.add_argument('run', nargs='*', metavar='RUN', help='run files')

    args = parser.parse_args()

    return args


def main():
    sns.set_style({'font.family': 'serif', 'font.serif': 'Latin Modern Roman'})
    args = parse_args()

    evals = []
    for run in args.run:
        _, result = eval_run('gdeval@20', args.qrel, run)
        result = [(qno, m['GDEVAL-NDCG@20']) for qno, m in result.items()]
        evals.append(result)

    count = len(args.run)
    markers = ['.', '1', '2', '3', '4', '+', 'x']
    point_vs_point('', evals, args.names, markers[:count], args.output)


if __name__ == '__main__':
    main()
