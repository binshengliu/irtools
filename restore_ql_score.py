#!/usr/bin/env python3
import argparse
from pathlib import Path
import lxml.etree as ET


def parse_args():
    parser = argparse.ArgumentParser(
        description='Restore QL score by '
        'multiplying query length with Indri scores.')

    parser.add_argument('--param', '-p', required=True, type=Path)

    parser.add_argument('--run', '-r', required=True, type=Path)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    qno_len = {}
    for node in ET.parse(str(args.param)).getroot().findall('query'):
        number = node.find('number').text
        text = node.find('text').text
        qno_len[number] = len(text.split())

    for line in args.run.read_text().splitlines():
        qno, q0, docno, ranking, score, indri = line.split()
        score = float(score) * qno_len[qno]
        print(qno, q0, docno, ranking, score, indri)


if __name__ == '__main__':
    main()
