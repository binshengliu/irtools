#!/usr/bin/env python3
from concurrent.futures import ThreadPoolExecutor
import sys


def parse(r):
    with open(r, 'r') as f:
        for line in f:
            try:
                _, _, _, rank, score, _ = line.split()
                rank = int(rank)
                score = float(score)
            except ValueError:
                print(r)
                return


def main():
    with ThreadPoolExecutor() as executor:
        files = sys.argv[1:]
        if not files:
            files = [f.strip('\n') for f in sys.stdin]
        executor.map(parse, files)


if __name__ == '__main__':
    main()
