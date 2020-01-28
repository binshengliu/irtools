from multiprocessing import Process, Queue
from more_itertools import divide, flatten
from unidecode import unidecode
from tqdm import trange
import threading
import argparse
import spacy
import ftfy
import sys
import re
import os

skeleton_set = {
    'ADJ', 'ADV', 'INTJ', 'NOUN', 'NUM', 'PRON', 'PROPN', 'VERB', 'X'
}


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def remove_non_printable(string):
    reg = re.compile('[^\x00-\x7f]')
    return reg.sub('', string)


def process_line(nlp, line, delimiter, field, lower, alnum, eol, form,
                 remove_stop, skeleton):
    fields = line.split(delimiter)

    if field is None:
        field = range(len(fields))
    for f in field:
        s = fields[f]
        s = unidecode(ftfy.fix_text(s))
        s = remove_non_printable(s)
        s = nlp(s)
        if remove_stop:
            s = [x for x in s if not x.is_stop]
        if skeleton:
            s = [x for x in s if x.pos_ in skeleton_set]
        s = [getattr(x, form + '_') for x in s if not x.norm_.isspace()]
        if alnum:
            s = [''.join(filter(str.isalnum, x)) for x in s]
            s = [x for x in s if x]
        s = ' '.join(s)
        if lower:
            s = s.lower()
        fields[f] = s

    result = delimiter.join(fields) + eol
    return result


def worker(job_id, batch, delimiter, field, lower, alnum, eol, form,
           remove_stop, skeleton, out_q, ui_q):
    if skeleton:
        nlp = spacy.load('en_core_web_lg', disable=['parser', 'ner'])
    else:
        nlp = spacy.load('en_core_web_lg', disable=['parser', 'ner', 'tagger'])
    nlp.max_length = 100000000

    results = []
    processed = 0
    for line in batch:
        line = process_line(nlp, line, delimiter, field, lower, alnum, eol,
                            form, remove_stop, skeleton)
        results.append(line)
        processed += 1
        if processed % 1024 == 0:
            ui_q.put(processed)
            processed = 0
    ui_q.put(processed)
    out_q.put((job_id, results))
    return


def thread_progress_bar(total, q):
    with trange(total, desc='Tokenize') as bar:
        while True:
            msg = q.get()
            if msg is None:
                return
            else:
                bar.update(msg)


def spacit(lines,
           workers=1,
           delimiter='\t',
           field=None,
           lower=True,
           alnum=True,
           eol='\n',
           form='norm',
           remove_stop=False,
           skeleton=False):
    if not hasattr(lines, '__len__'):
        lines = list(lines)
    if isinstance(field, int):
        field = [field]

    assert form in ['norm', 'lemma', 'text']

    ui_q = Queue()
    ui = threading.Thread(target=thread_progress_bar, args=(len(lines), ui_q))
    ui.start()

    out_q = Queue()
    procs = []
    for job_id, batch in enumerate(divide(workers, lines)):
        proc = Process(
            target=worker,
            args=(job_id, batch, delimiter, field, lower, alnum, eol, form,
                  remove_stop, skeleton, out_q, ui_q))
        proc.start()
        procs.append(proc)
    results = [None] * workers
    for _ in range(workers):
        job_id, lines = out_q.get()
        results[job_id] = lines
    results = list(flatten(results))
    for proc in procs:
        proc.join()
    ui_q.put(None)
    ui.join()

    return results


def parse_arguments():
    def int_comma(line):
        return [int(x) for x in str(line).split(',')]

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input', type=argparse.FileType('r'))
    parser.add_argument(
        '-d', '--delimiter', default='\t', help='default to \\t')
    parser.add_argument(
        '-j',
        '--workers',
        type=int,
        default=os.cpu_count() // 2,
        help='number of workers, default to half of cpu count')
    parser.add_argument(
        '-f',
        '--field',
        type=int_comma,
        help='zero-based field index to process, e.g. 0,1,2,3.')
    parser.add_argument(
        '--no-lower',
        action='store_true',
        help='not convert to lower case if specified')
    parser.add_argument(
        '--keep-nonalnum',
        action='store_true',
        help=('override the default behavior of '
              'keeping only English alphabet and numbers'))

    return parser.parse_args()


def main():
    args = parse_arguments()
    # from itertools import islice
    # with open('data/msmarco-pass.tsv', 'r') as f:
    #     test = list(islice(f, 100))
    lines = spacit(args.input, args.workers, args.delimiter, args.field,
                   not args.no_lower, not args.keep_nonalnum, '\n')
    sys.stdout.writelines(lines)


if __name__ == '__main__':
    main()
