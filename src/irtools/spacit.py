from multiprocessing import Pool
from unidecode import unidecode
from tqdm import tqdm
import spacy
import ftfy
import re

skeleton_set = {
    'ADJ', 'ADV', 'INTJ', 'NOUN', 'NUM', 'PRON', 'PROPN', 'VERB', 'X'
}


def remove_non_printable(string):
    reg = re.compile('[^\x00-\x7f]')
    return reg.sub('', string)


def proc_line(line):
    nlp = proc_line.nlp
    delimiter, field, lower, alnum, eol, form, remove_stop, skeleton \
        = proc_line.args

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


def init_nlp(delimiter, field, lower, alnum, eol, form, remove_stop, skeleton):
    if skeleton:
        nlp = spacy.load('en_core_web_lg', disable=['parser', 'ner'])
    else:
        nlp = spacy.load('en_core_web_lg', disable=['parser', 'ner', 'tagger'])
    nlp.max_length = 100000000
    proc_line.nlp = nlp
    proc_line.args = (delimiter, field, lower, alnum, eol, form, remove_stop,
                      skeleton)


def spacit(lines,
           nworkers=1,
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

    with Pool(
            nworkers,
            initializer=init_nlp,
            initargs=(delimiter, field, lower, alnum, eol, form, remove_stop,
                      skeleton)) as pool:
        results = pool.imap(proc_line, lines, chunksize=1024)
        results = list(tqdm(results, total=len(lines)))

    return results
