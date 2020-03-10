#!/usr/bin/env python3
from transformers import (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          GPT2LMHeadModel, GPT2Tokenizer)
from multiprocessing import Pool, Queue
from tqdm import tqdm
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


def worker(sent):
    # Load pre-trained model (weights)
    model, tokenizer, gpu = worker.model, worker.tokenizer, worker.gpu
    try:
        nwords = len(sent.split())
        if nwords == 1:
            sent += ' .'
        tensor_input = torch.tensor([tokenizer.encode(sent)]).cuda(gpu)
        loss = model(input_ids=tensor_input, labels=tensor_input)
        ppl = math.exp(loss[0].item())
        return ppl, ppl * nwords, ppl / nwords
    except Exception:
        eprint('ERROR {}'.format(sent))
        return None, None, None


def init_worker(model_type, model_name, available_ids):
    gpu = available_ids.get()
    eprint(f'Initialize on GPU {gpu}')
    model_class, tokenizer_class = MODEL_CLASSES[model_type]
    model = model_class.from_pretrained(model_name).cuda(gpu)
    model.eval()
    tokenizer = tokenizer_class.from_pretrained(model_name)
    worker.model = model
    worker.tokenizer = tokenizer
    worker.gpu = gpu


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

    if not hasattr(queries, '__len__'):
        queries = list(queries)
    total = len(queries)
    available_ids = Queue()
    for gpu in deviceIDs:
        available_ids.put(gpu)
    with Pool(
            len(deviceIDs),
            initializer=init_worker,
            initargs=(model_type, model_name, available_ids)) as pool:
        results = pool.imap(worker, queries)
        results = list(tqdm(results, total=total))

    return results


def main():
    args = parse_arguments()
    queries = args.query.read().splitlines()
    result = perplexity(queries, args.model_type, args.model_name)
    print(''.join(['\t'.join(map(str, x)) + '\n' for x in result]), end='')


if __name__ == '__main__':
    main()
