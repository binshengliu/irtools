from more_itertools import first_true
from .tqdmf import tqdmf


def _parse_one_record(buffers):
    dno = first_true(buffers, pred=lambda x: x.startswith('<DOCNO>'))
    dno = dno.replace('<DOCNO>', '').replace('</DOCNO>\n', '')

    indexes = list(range(len(buffers)))
    text_start_idx = first_true(
        indexes, pred=lambda x: buffers[x].startswith('<TEXT>'))

    text_end_idx = first_true(
        indexes, pred=lambda x: buffers[x].startswith('</TEXT>'))

    text = buffers[text_start_idx + 1:text_end_idx]

    return dno, text


def trec_text(path):
    buffers = []
    for line in tqdmf(path):
        if not line:
            continue
        buffers.append(line)
        if line.startswith('</DOC>'):
            dno, text = _parse_one_record(buffers)
            yield dno, text
            buffers = []
