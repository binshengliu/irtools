import argparse
import multiprocessing as mp
import os
from collections import Counter
from typing import Counter as TCounter
from typing import Iterable, List, Tuple

from irtools.log import get_logger
from irtools.tqdmf import tqdmf
from more_itertools import chunked
from transformers import AutoTokenizer

log = get_logger(__name__)

g_tok = None


def init(arch: str) -> None:
    global g_tok
    g_tok = AutoTokenizer.from_pretrained(arch)


def analyze(lines: Iterable[str]) -> Tuple[TCounter[str], TCounter[str]]:
    global g_tok
    assert g_tok is not None
    tf: TCounter[str] = Counter()
    df: TCounter[str] = Counter()
    for line in lines:
        tokens: List[str] = g_tok.tokenize(line.split("\t")[1])
        tf.update(tokens)
        df.update(set(tokens))
    return tf, df


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--arch", required=True)
    parser.add_argument("--tsv", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    log.info(args)
    tf_counter: TCounter[str] = Counter()
    df_counter: TCounter[str] = Counter()
    with mp.Pool(args.workers, initializer=init, initargs=(args.arch,)) as pool:
        results = pool.imap(analyze, chunked(tqdmf(args.tsv), args.batch_size))
        for tf, df in results:
            tf_counter += tf
            df_counter += df

    vocab = sorted(tf_counter.keys(), key=lambda x: tf_counter[x], reverse=True)
    for word in vocab:
        print(f"{word}\t{tf_counter[word]}\t{df_counter[word]}")


if __name__ == "__main__":
    main()
