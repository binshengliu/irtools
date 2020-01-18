import argparse
import string
import spacy
import ftfy
import sys


def convert(index, line):
    try:
        number, content = line.strip().split(maxsplit=1)
    except Exception:
        number = line.strip()
        content = 'this is an empty query'
    content = ftfy.fix_text(content)
    content = ' '.join(content.split()).lower()
    template = '  <query>\n    <number>{}</number>\n    <text>{}</text>\n  </query>\n'
    output = template.format(number, content)
    return output


def parse_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data', type=argparse.FileType('r'))
    parser.add_argument('--key', type=argparse.FileType('r'))

    return parser.parse_args()


def main():
    args = parse_arguments()

    for index, query_line in enumerate(text):
        query_line = convert(index, query_line)
        sys.stdout.write(query_line)
    sys.stdout.write('</parameters>\n')


if __name__ == '__main__':
    main()
