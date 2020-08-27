#!/usr/bin/env python3
import argparse
import sys
from typing import Generator, List, TextIO, Tuple

from more_itertools import first_true
from tqdm import tqdm


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-i", "--input", type=argparse.FileType("r"), default=sys.stdin, help=""
    )
    parser.add_argument(
        "-o", "--output", type=argparse.FileType("w"), default=sys.stdout, help=""
    )

    return parser.parse_args()


def parse_one_record(buffers: List[str]) -> Tuple[str, str]:
    dno = first_true(buffers, pred=lambda x: x.startswith("<DOCNO>"))
    assert dno is not None
    dno = dno.replace("<DOCNO>", "").replace("</DOCNO>", "")

    indexes = list(range(len(buffers)))
    text_start_idx = first_true(indexes, pred=lambda x: buffers[x].startswith("<BODY>"))
    assert text_start_idx is not None

    text_end_idx = first_true(indexes, pred=lambda x: buffers[x].startswith("</BODY>"))
    assert text_end_idx is not None

    text = " ".join(buffers[text_start_idx + 1 : text_end_idx])
    text = text.replace("\t", " ").replace("\r", " ")

    return dno, text


def trec_text(input_: TextIO) -> Generator[Tuple[str, str], None, None]:
    buffers = []
    for line in input_:
        if not line:
            continue
        buffers.append(line.rstrip("\n"))
        if line.startswith("</DOC>"):
            dno, text = parse_one_record(buffers)
            yield dno, text
            buffers = []


def main() -> None:
    args = parse_arguments()
    for dno, text in tqdm(trec_text(args.input), unit=" docs"):
        args.output.write(f"{dno}\t{text}\n")


if __name__ == "__main__":
    main()
