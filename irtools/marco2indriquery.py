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
    parser.add_argument('--no-qno', action='store_true')

    return parser.parse_args()


def main():
    args = parse_arguments()
    sys.stdout.write('<parameters>\n')
    #     dummy = """1738    <span> meaning 134.22
    # 659438  what female sings just one look at you & i know it's going to be a lovely day
    # 29180   at&t gophone customer service phone number
    # """.splitlines()
    text = sys.stdin
    for index, query_line in enumerate(text):
        query_line = convert(index, query_line)
        sys.stdout.write(query_line)
    sys.stdout.write('</parameters>\n')


if __name__ == '__main__':
    main()
