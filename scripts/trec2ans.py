#!/usr/bin/env python3
import sys


def main():
    for line in sys.stdin:
        qno, _, dno, rank, _, _ = line.strip().split()
        print(f'{qno} {dno} {rank}')


if __name__ == '__main__':
    main()
