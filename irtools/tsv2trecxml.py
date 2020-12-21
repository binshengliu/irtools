#!/usr/bin/env python3
import argparse
import sys


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-i", "--input", type=argparse.FileType("r"), default=sys.stdin, help=""
    )
    parser.add_argument(
        "-o", "--output", type=argparse.FileType("w"), default=sys.stdout, help=""
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    lines = list(args.input)
    if not lines:
        return
    args.output.write("<parameters>\n")
    for line in lines:
        qid, text = line.rstrip("\n").split("\t")
        args.output.write("  <query>\n")
        args.output.write(f"    <number>{qid}</number>\n")
        args.output.write(f"    <text>{text}</text>\n")
        args.output.write("  </query>\n")

    args.output.write("</parameters>\n")


if __name__ == "__main__":
    main()
