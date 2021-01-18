#!/usr/bin/env python3
import argparse
import sys


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-i",
        "--input",
        type=argparse.FileType("r"),
        default=sys.stdin,
        help="default stdin",
    )

    def field_type(x: str) -> int:
        value = int(x)
        if value < 1:
            raise ValueError("--field needs to be 1, 2, 3 ...")
        value -= 1
        return value

    parser.add_argument("--field", type=field_type)

    parser.add_argument(
        "-o",
        "--output",
        type=argparse.FileType("w"),
        default=sys.stdout,
        help="default stdout",
    )

    parser.add_argument("--keys", type=argparse.FileType("r"))

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    keys = {x.rstrip("\n") for x in args.keys}
    for line in args.input:
        if line.split()[args.field] in keys:
            args.output.write(line)


if __name__ == "__main__":
    main()
