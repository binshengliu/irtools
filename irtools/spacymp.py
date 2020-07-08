#!/usr/bin/env python3
import multiprocessing as mp
from typing import Iterable, Iterator, List

import spacy
from irtools.log import get_logger
from tqdm import tqdm

logger = get_logger("spacymp")

g_nlp = None
g_lemmatize = False
g_remove_stop = False
g_remove_punct = False


def init_nlp(
    lang: str,
    disable: List[str],
    lemmatize: bool,
    remove_stop: bool,
    remove_punct: bool,
) -> None:
    global g_nlp, g_lemmatize, g_remove_stop, g_remove_punct
    g_nlp = spacy.load(lang, disable=disable)
    g_lemmatize = lemmatize
    g_remove_stop = remove_stop
    g_remove_punct = remove_punct


def process(line: str) -> List[str]:
    global g_nlp, g_lemmatize, g_remove_stop, g_remove_punct
    assert g_nlp is not None
    tokenized = g_nlp(line)

    def keep(token: spacy.tokens.Token) -> bool:
        if g_remove_stop and token.is_stop:
            return False
        if g_remove_punct and token.is_punct:
            return False
        return True

    def transform(token: spacy.tokens.Token) -> str:
        return token.lemma_ if g_lemmatize else token.text  # type: ignore

    return [transform(x) for x in tokenized if keep(x)]


def spacymp(
    input_: Iterable[str],
    lang: str = "en",
    disable: List[str] = ["parser", "ner"],
    n_process: int = 1,
    batch_size: int = 1000,
    lemmatize: bool = False,
    remove_stop: bool = False,
    remove_punct: bool = False,
) -> Iterator[List[str]]:
    texts = list(input_)
    logger.info(f"Received {len(texts)} lines")
    logger.info(f"Initialize {n_process} processes")
    with mp.Pool(
        n_process,
        initializer=init_nlp,
        initargs=(lang, disable, lemmatize, remove_stop, remove_punct),
    ) as pool:
        pool_iter = pool.imap(process, texts, chunksize=batch_size)
        for doc_tokens in tqdm(pool_iter, total=len(texts)):
            yield doc_tokens
            # yield spacy.tokens.Doc(vocab).from_bytes(doc_tokens)
