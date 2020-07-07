#!/usr/bin/env python3
import argparse
import logging
import multiprocessing as mp
import sys

import spacy
from spacy.tokens import DocBin
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="[%(asctime)s][%(levelname)s] - %(message)s",
)
logger = logging.getLogger("spacify")

nlp = None
count = 0
args = None


def init_nlp(args_: argparse.Namespace) -> None:
    global nlp, args
    args = args_
    nlp = spacy.load(args.lang, disable=["parser", "ner"])
    # logger = mp.log_to_stderr()
    # logger.info("Worker initialized")


def process(line: str) -> DocBin:
    global nlp
    assert nlp is not None
    doc_bin = DocBin(attrs=["LEMMA", "IS_STOP"])
    doc_bin.add(nlp(line))
    return doc_bin


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="""
Deserialize with:

path = "path.spacy""
nlp = spacy.blank("en")
with open(path, "rb") as f:
    docs = list(DocBin().from_bytes(f.read()).get_docs(nlp.vocab))""",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input", type=argparse.FileType("r"), default=sys.stdin, help=""
    )
    parser.add_argument("-o", "--output", type=argparse.FileType("wb"), required=True)
    parser.add_argument("--lang", default="en", help="", choices=["en"])

    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    # logger = mp.log_to_stderr(level=logging.INFO)
    args = parse_arguments()
    logger.info(f"Receive input from {args.input.name}")
    texts = list(args.input)
    logger.info(f"Received {len(texts)} lines")
    doc_bin = DocBin(attrs=["LEMMA", "IS_STOP"])
    logger.info(f"Initialize {args.workers} processes")
    with mp.Pool(args.workers, initializer=init_nlp, initargs=(args,),) as pool:
        pool_iter = pool.imap(process, texts, chunksize=args.batch_size)
        for doc in tqdm(pool_iter, total=len(texts)):
            doc_bin.merge(doc)

    args.output.write(doc_bin.to_bytes())
    logger.info(f"Saved at {args.output.name}")


if __name__ == "__main__":
    main()
