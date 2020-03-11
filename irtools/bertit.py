from transformers import BertTokenizer
from multiprocessing import Pool
from itertools import repeat
from tqdm import tqdm


def process_line(args):
    line, tokenizer, delimiter, field, eol, \
        add_special_tokens, max_length, pad_to_max_length = args

    fields = line.rstrip('\n').split(delimiter)

    if field is None:
        field = range(len(fields))
    for f in field:
        s = fields[f]
        s = tokenizer.encode(
            s,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            pad_to_max_length=pad_to_max_length)
        fields[f] = ' '.join([str(x) for x in s])

    result = delimiter.join(fields) + eol
    return result


def bertit(lines,
           threads=1,
           delimiter='\t',
           field=None,
           eol='\n',
           add_special_tokens=True,
           max_length=None,
           pad_to_max_length=False):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    if not hasattr(lines, '__len__'):
        lines = list(lines)
    total = len(lines)

    with Pool(threads) as pool:
        lines = pool.imap(
            process_line,
            zip(lines, repeat(tokenizer), repeat(delimiter), repeat(field),
                repeat(eol), repeat(add_special_tokens), repeat(max_length),
                repeat(pad_to_max_length)),
            chunksize=2048)
        lines = list(tqdm(lines, total=total))

    return lines
