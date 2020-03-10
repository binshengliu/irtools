from multiprocessing import Pool
from itertools import repeat
from unidecode import unidecode
from tqdm import tqdm
import ftfy


def process_line(args):
    line, delimiter, field, lower, alnum, eol = args

    fields = line.rstrip('\n').split(delimiter)

    if field is None:
        field = range(len(fields))
    for f in field:
        s = fields[f]
        s = unidecode(ftfy.fix_text(s))
        if alnum:
            s = [''.join(filter(str.isalnum, x)) for x in s.split()]
            s = ' '.join(s)
        if lower:
            s = s.lower()
        fields[f] = ' '.join(s.split())

    result = delimiter.join(fields) + eol
    return result


def cleanit(lines,
            threads=1,
            delimiter='\t',
            field=None,
            lower=True,
            alnum=True,
            eol='\n'):
    if not hasattr(lines, '__len__'):
        lines = list(lines)
    total = len(lines)

    with Pool(threads) as pool:
        lines = pool.imap(
            process_line,
            zip(lines, repeat(delimiter), repeat(field), repeat(lower),
                repeat(alnum), repeat(eol)),
            chunksize=2048)
        lines = list(tqdm(lines, total=total))

    return lines
