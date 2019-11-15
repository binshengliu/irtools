import argparse
import string
import spacy
import ftfy
import sys

nlp = spacy.load('en_core_web_lg', disable=['tagger', 'parser', 'ner'])
trans_table = str.maketrans('', '', string.punctuation)


def convert_query(index, query):
    qno, content = query.strip().split(maxsplit=1)
    content = ftfy.fix_text(content)
    content = ''.join(filter(lambda x: x in string.printable, content))
    content = nlp(content)
    content = [x.text.translate(trans_table) for x in content]
    content = [x for x in content if x]
    content = ' '.join(content).lower()
    template = '  <query>\n    <number>{}</number>\n    <text>{}</text>\n  </query>\n'
    output = template.format(qno, content)
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
        query_line = convert_query(index, query_line)
        sys.stdout.write(query_line)
    sys.stdout.write('</parameters>\n')


if __name__ == '__main__':
    main()
