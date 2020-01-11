from multiprocessing import Pool
from itertools import repeat
from unidecode import unidecode
from tqdm import tqdm
import argparse
import ftfy
import sys
import os


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def process_line(args):
    line, delimiter, field, lower, alnum, eol = args

    fields = line.rstrip('\n').split(delimiter)

    if field is None:
        field = range(len(fields))
    for f in field:
        s = fields[f]
        s = unidecode(ftfy.fix_text(s))
        if alnum:
            s = [''.join(filter(str.isalnum, x)) for x in s]
            s = [x for x in s if x]
            s = ''.join(s)
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


def parse_arguments():
    def int_comma(line):
        return [int(x) for x in str(line).split(',')]

    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '-d', '--delimiter', default='\t', help='default to \\t')
    parser.add_argument(
        '-j',
        '--threads',
        type=int,
        default=os.cpu_count() // 2,
        help='number of threads, default to half of cpu count')
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
    lines = cleanit(sys.stdin, args.threads, args.delimiter, args.field,
                    not args.no_lower, not args.keep_nonalnum, '\n')
    sys.stdout.writelines(lines)


if __name__ == '__main__':
    main()
