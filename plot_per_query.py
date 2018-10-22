#!/usr/bin/env python3
from pathlib import Path
import seaborn as sns
import pandas as pd
from eval_run import eval_run
import sys
import argparse
from matplotlib import pyplot as plt

# plt.switch_backend('agg')


def eprint(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr, flush=True)


def split_comma(s):
    return s.split(',')


def eval_wrapper(measure, qrel, run):
    try:
        return eval_run(measure, qrel, run)
    except Exception:
        eprint('Error: {}'.format(run))


def point_vs_point(title, point1data, names, markers, filename):
    dfs = [pd.DataFrame(d, columns=['Query', 'Measure']) for d in point1data]
    order = dfs[0].sort_values(by='Measure', ascending=False)['Query']
    df = pd.concat(
        dfs, keys=names, names=['System']).reset_index(level='System')
    g = sns.relplot(
        kind='scatter',
        data=df,
        x='Query',
        y='Measure',
        hue='System',
        style='System',
        markers=markers,
        # palette='Set1',
        # join=False,
        # scale=0.4,
        # legend_out=False,
        palette=['#95a5a6', '#f39c12', '#2980b9', '#8e44ad'],
        # alpha=0.8,
        s=30,
        # mew=0.5,
        # scatter_kws={'alpha': 0.5}
    )

    # for patch in g.ax.artists:
    #     r, g, b, a = patch.get_facecolor()
    #     patch.set_facecolor((r, g, b, .3))
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
    # sns.set_style({'font.family': 'serif', 'font.serif': 'Latin Modern Roman'})
    plt.rcParams.update({
        "backend": "cairo",
        # "text.usetex": True,
        "ps.fonttype": 42,
        "pdf.fonttype": 42,
        "lines.linewidth": 2.0,
        "patch.linewidth": 0.5,
        "axes.facecolor": "#ffffff",
        "axes.labelsize": "large",
        "axes.axisbelow": True,
        "axes.grid": True,
        "patch.edgecolor": "#f0f0f0",
        "axes.titlesize": "x-large",
        "examples.directory": "",
        "figure.facecolor": "#f0f0f0",
        "grid.linestyle": "-",
        "grid.linewidth": 0.5,
        "grid.color": "#cbcbcb",
        "axes.edgecolor": "#000000",
        "xtick.major.size": 0,
        "xtick.minor.size": 0,
        "ytick.major.size": 0,
        "ytick.minor.size": 0,
        "axes.linewidth": 1.0,
        "font.size": 18.0,
        "font.family": "sans-serif",
        "lines.linewidth": 2,
        "lines.solid_capstyle": "butt",
        "savefig.edgecolor": "#000000",
        "savefig.facecolor": "#ffffff",
        "legend.fancybox": False,
        # 'legend.handlelength': 2,
        'legend.fontsize': 'x-small',
        'legend.borderpad': 0.1,
        'legend.labelspacing': 0.1,
    })
    args = parse_args()

    evals = []
    base = None
    for run in args.run:
        _, result = eval_run('gdeval@20', args.qrel, run)
        if base is None:
            base = {qno: m['GDEVAL-NDCG@20'] for qno, m in result.items()}
        result = sorted(
            [(qno, m['GDEVAL-NDCG@20']) for qno, m in result.items()],
            key=lambda r: base[r[0]],
            reverse=True)
        evals.append(result)

    count = len(args.run)
    markers = [
        'o',
        'p',
        'P',
        'X',
        'P',
        '>',
        '.',
        '.',
    ]
    point_vs_point('', evals, args.names, markers[:count], args.output)


if __name__ == '__main__':
    main()
