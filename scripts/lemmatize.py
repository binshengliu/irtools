#!/usr/bin/env python3
import argparse
import os
import sys

import spacy


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-d", "--delimiter", default="\t", help="default to \\t")

    cpu_count = os.cpu_count()
    assert isinstance(cpu_count, int)
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=cpu_count // 2,
        help=f"default to half of cpu count {cpu_count // 2}",
    )

    parser.add_argument(
        "-f",
        "--field",
        type=lambda x: int(x) - 1,
        help="one-based field index to process.",
    )

    parser.add_argument("--remove-stopwords", action="store_true")

    parser.add_argument(
        "-i",
        "--input",
        type=argparse.FileType("r"),
        default=sys.stdin,
        help="default stdin",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=argparse.FileType("w"),
        default=sys.stdout,
        help="default stdout",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    field = args.field
    inputs = [x.split(args.delimiter) for x in args.input]
    texts = [x[field] for x in inputs]

    nlp = spacy.load("en", disable=["parser", "ner"])
    processed = list(nlp.pipe(texts, n_process=args.workers))
    if args.remove_stopwords:
        processed = [[x for x in one if not x.is_stop] for one in processed]
    processed = [" ".join(x.lemma_ for x in one) for one in processed]
    outputs = [
        args.delimiter.join(x[:field] + [y] + x[args.field + 1 :])
        for x, y in zip(inputs, processed)
    ]

    args.output.writelines(outputs)


if __name__ == "__main__":
    main()
