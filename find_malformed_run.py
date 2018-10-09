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
        executor.map(parse, sys.argv[1:])


if __name__ == '__main__':
    main()
