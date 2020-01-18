#!/usr/bin/env python3

import os
from concurrent.futures import ProcessPoolExecutor as Pool
import subprocess
from more_itertools import chunked
from itertools import chain
from pathlib import Path
import argparse
import sys
import io


def _tokenize(text):
    args = [
        'java',
        '-cp',
        Path(__file__).resolve().with_name('stanford-corenlp-3.9.2.jar'),
        'edu.stanford.nlp.process.PTBTokenizer',
        '-encoding',
        'ascii',
        '-lowerCase',
        '-preserveLines',
        # '-options',
        # 'ptb3Escaping=false,normalizeOtherBrackets=false,asciiQuotes=true,latexQuotes=false',
    ]
    proc = subprocess.run(args,
                          input=text,
                          stdout=subprocess.PIPE,
                          encoding='utf-8')
    processed = proc.stdout

    return processed


def _tokenize_mp(lines):
    length = len(lines)
    cpus = os.cpu_count()
    chunks = list(map('\n'.join, chunked(lines, 10000)))
    with Pool(cpus) as pool:
        output = pool.map(_tokenize, chunks)

    output = list(chain.from_iterable(i.split('\n') for i in output))
    assert length == len(output), 'Input: {} lines, output: {} lines'.format(
        length, len(output))
    return output


def tokenize(content):
    if isinstance(content, str):
        return '\n'.join(_tokenize_mp(content.split('\n')))
    else:
        return _tokenize_mp(content)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Multi-processing tokenizer for huge text')

    parser.add_argument('-i', '--input', default=sys.stdin)
    parser.add_argument('-o', '--output', default=sys.stdout)

    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.input == sys.stdin:
        input_stream = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
        texts = input_stream.read()
    else:
        texts = Path(args.input).read_text()

    texts = tokenize(texts)

    if args.output == sys.stdout:
        output_stream = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        output_stream.write(texts)
    else:
        Path(args.output).write_text(texts)


if __name__ == '__main__':
    main()
