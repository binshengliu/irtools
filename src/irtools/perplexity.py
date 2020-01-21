#!/usr/bin/env python3
from transformers import OpenAIGPTLMHeadModel as Model, OpenAIGPTTokenizer as Tokenizer
from concurrent.futures import ProcessPoolExecutor as Pool
from more_itertools import chunked
from itertools import chain, count
from tqdm import tqdm
import argparse
import torch
import math
import sys


def eprint(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr, flush=True)


def score(sentence, gpu=0):
    # Load pre-trained model (weights)
    model = Model.from_pretrained('openai-gpt').cuda(gpu)
    model.eval()
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = Tokenizer.from_pretrained('openai-gpt')

    results = []
    for sent in tqdm(
            sentence,
            total=len(sentence),
            position=gpu,
            leave=False,
            desc=str(gpu)):
        try:
            if len(sent.split()) == 1:
                sent = sent + ' ' + sent
            nwords = len(sent.split())
            tensor_input = torch.tensor([tokenizer.encode(sent)]).cuda(gpu)
            loss = model(tensor_input, labels=tensor_input)
            ppl = math.exp(loss[0].item())
            results.append((ppl, ppl * nwords))
        except Exception:
            print('ERROR {}'.format(sent), flush=True)
            eprint('ERROR {}'.format(sent))
    return results


def parse_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--query', required=True, type=argparse.FileType('r'))
    return parser.parse_args()


def perplexity(queries):
    total = len(queries)
    chunksize = math.ceil(total / 2)
    queries = chunked(queries, chunksize)
    with Pool(2) as pool:
        result = pool.map(score, queries, count())
        result = list(chain.from_iterable(result))
    return result


def main():
    args = parse_arguments()
    queries = args.query.read().splitlines()
    result = perplexity(queries)
    print('\t'.join(map(str, result)), flush=True)


if __name__ == '__main__':
    main()
