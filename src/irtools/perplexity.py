#!/usr/bin/env python3
from transformers import (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          GPT2LMHeadModel, GPT2Tokenizer)
from multiprocessing import Process, Queue
from more_itertools import chunked
from tqdm import trange
import argparse
import GPUtil
import torch
import math
import sys

MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
    "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
}


def eprint(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr, flush=True)


def worker(batch_sent, model_type, model_name, gpu, q):
    # Load pre-trained model (weights)
    model_class, tokenizer_class = MODEL_CLASSES[model_type]
    model = model_class.from_pretrained(model_name).cuda(gpu)
    model.eval()
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = tokenizer_class.from_pretrained(model_name)

    for id, sent in batch_sent:
        try:
            nwords = len(sent.split())
            if nwords == 1:
                sent += ' .'
            tensor_input = torch.tensor([tokenizer.encode(sent)]).cuda(gpu)
            loss = model(input_ids=tensor_input, labels=tensor_input)
            ppl = math.exp(loss[0].item())
            q.put((id, ppl, ppl * nwords, ppl / nwords))
        except Exception:
            q.put((id, None, None, None))
            eprint('ERROR {}'.format(sent))
    return


def parse_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--query', required=True, type=argparse.FileType('r'))
    parser.add_argument('--model-type', default='openai-gpt')
    parser.add_argument('--model-name', default='openai-gpt')
    return parser.parse_args()


def perplexity(queries, model_type='openai-gpt', model_name='openai-gpt'):
    deviceIDs = GPUtil.getAvailable(
        order='first', limit=8, maxLoad=0.5, maxMemory=0.5)
    nproc = len(deviceIDs)
    if nproc == 0:
        raise ValueError('No available GPU')

    queries = list(enumerate(queries))
    total = len(queries)
    queries = list(chunked(queries, math.ceil(total / nproc)))
    procs = []
    q = Queue()
    for gpu, batch_sent in zip(deviceIDs, queries):
        proc = Process(
            target=worker, args=(batch_sent, model_type, model_name, gpu, q))
        proc.start()
        procs.append(proc)
    results = [None] * total
    for _ in trange(total):
        id, *ppls = q.get()
        results[id] = ppls
    for proc in procs:
        proc.join()

    return results


def main():
    args = parse_arguments()

    queries = args.query.read().splitlines()
    result = perplexity(queries, args.model_type, args.model_name)
    print(''.join(['{}\t{}\t{}\n'.format(*x) for x in result]), end='')


if __name__ == '__main__':
    main()
