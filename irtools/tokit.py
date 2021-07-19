from itertools import repeat
from multiprocessing import Pool

from tqdm import tqdm
from transformers import ALL_PRETRAINED_CONFIG_ARCHIVE_MAP, AutoTokenizer


def process_line(args):
    line, tokenizer, delimiter, field, eol, text, \
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
            pad_to_max_length=pad_to_max_length,
            truncation=True if max_length else False)
        if text:
            fields[f] = ' '.join([str(x) for x in s])
        else:
            fields[f] = s

    if text:
        result = delimiter.join(fields) + eol
    else:
        result = fields
    return result


def get_all_models():
    return list(ALL_PRETRAINED_CONFIG_ARCHIVE_MAP.keys())


def tokit(model,
          lines,
          threads=1,
          delimiter='\t',
          field=None,
          eol='\n',
          text=False,
          add_special_tokens=True,
          max_length=None,
          pad_to_max_length=False):
    tokenizer = AutoTokenizer.from_pretrained(model)
    if not hasattr(lines, '__len__'):
        lines = list(lines)
    total = len(lines)

    with Pool(threads) as pool:
        lines = pool.imap(
            process_line,
            zip(lines, repeat(tokenizer), repeat(delimiter), repeat(field),
                repeat(eol), repeat(text), repeat(add_special_tokens),
                repeat(max_length), repeat(pad_to_max_length)),
            chunksize=2048)
        lines = list(tqdm(lines, total=total))

    return lines
