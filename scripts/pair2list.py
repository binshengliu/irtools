#!/usr/bin/env python3
import argparse
import sys
from collections import OrderedDict

import numpy as np
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument(
        '-i',
        '--input',
        type=argparse.FileType('r'),
        default=sys.stdin,
        help='default stdin')

    parser.add_argument(
        '-o',
        '--output',
        type=argparse.FileType('w'),
        default=sys.stdout,
        help='default stdout')

    parser.add_argument('--pos-label', default='1')
    parser.add_argument('--neg-label', default='0')
    parser.add_argument(
        '--sample-neg',
        type=float,
        help='if < 1.0, used as ratio; if >= 1 used as number')

    return parser.parse_args()


def main():
    args = parse_arguments()
    pos_label = args.pos_label
    neg_label = args.neg_label
    sample = args.sample_neg

    id_map = OrderedDict()
    delim = ' '
    for line in args.input:
        delim = ' ' if ' ' in line else '\t'
        id, pos, neg = line.split()
        id_map.setdefault(id, {'pos': set(), 'neg': set()})
        id_map[id]['pos'].add(pos)
        id_map[id]['neg'].add(neg)

    for id, value in tqdm(id_map.items(), total=len(id_map), desc='pair2list'):
        pos = value['pos']
        neg = value['neg']
        if sample is not None:
            if sample < 1:
                neg = np.random.choice(neg, round(len(neg) * sample))
            else:
                neg = np.random.choice(neg, round(sample))

        for one in pos:
            args.output.write(delim.join([id, one, pos_label]))

        for one in neg:
            args.output.write(delim.join([id, one, neg_label]))


if __name__ == '__main__':
    main()
