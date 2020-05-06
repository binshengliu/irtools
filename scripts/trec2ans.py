#!/usr/bin/env python3
import argparse
import sys


def parse_arguments():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-i", "--input", type=argparse.FileType("r"), default=sys.stdin, help=""
    )
    parser.add_argument(
        "-o", "--output", type=argparse.FileType("w"), default=sys.stdout, help=""
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    for line in args.input:
        qno, _, dno, rank, _, _ = line.strip().split()
        args.output.write(f"{qno}\t{dno}\t{rank}\n")


if __name__ == "__main__":
    main()
