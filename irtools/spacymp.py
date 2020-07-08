#!/usr/bin/env python3
import multiprocessing as mp
from itertools import repeat
from typing import Any, Callable, Iterable, Iterator, List, Optional, Union

import spacy
from irtools.log import get_logger
from tqdm import tqdm

logger = get_logger("spacymp")

g_nlp = None


def init_nlp(lang: str, disable: List[str]) -> None:
    global g_nlp
    g_nlp = spacy.load(lang, disable=disable)


def process(args: Any) -> Union[List[spacy.tokens.Doc], Any]:
    global g_nlp
    assert g_nlp is not None

    lines: List[str] = args[:-1]
    callback: Optional[Callable[..., Any]] = args[-1]

    tokenized = [g_nlp(line) for line in lines]

    if callback is not None:
        return callback(*tokenized)
    return tokenized


def spacymp(
    *input_: Iterable[str],
    callback: Optional[Callable[..., Any]] = None,
    lang: str = "en",
    disable: List[str] = ["parser", "ner"],
    n_process: int = 1,
    batch_size: int = 1000,
) -> Iterator[Any]:
    texts = [list(x) for x in input_]
    logger.info(f"Received {len(texts[0])} lines")
    logger.info(f"Initialize {n_process} processes")
    if n_process > 1:
        with mp.Pool(n_process, initializer=init_nlp, initargs=(lang, disable)) as pool:
            out_iter: Iterator[Any] = pool.imap(
                process, zip(*texts, repeat(callback)), chunksize=batch_size
            )
            yield from tqdm(out_iter, total=len(texts[0]))
    else:
        init_nlp(lang, disable)
        out_iter = map(process, zip(*texts, repeat(callback)))
        yield from tqdm(out_iter, total=len(texts[0]))
