#!/usr/bin/env python3
import sys


def main():
    for line in sys.stdin:
        qno, dno, rank = line.strip().split()
        score = 1.0 / int(rank)
        print(f'{qno} Q0 {dno} {rank} {score} ANSERINI')


if __name__ == '__main__':
    main()
