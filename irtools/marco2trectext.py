# import string
# import spacy
import ftfy
import sys

# nlp = spacy.load('en_core_web_lg', disable=['tagger', 'parser', 'ner'])
# punc2none = str.maketrans('', '', string.punctuation)
# punc2space = str.maketrans({x: ' ' for x in string.punctuation})


def convert(line):
    number, content = line.strip().split(maxsplit=1)
    content = ftfy.fix_text(content)
    content = ' '.join(content.split()).lower()
    template = '<DOC>\n<DOCNO>{}</DOCNO>\n<TEXT>\n{}\n</TEXT>\n</DOC>\n'
    output = template.format(number, content)
    return output


def main():
    for line in sys.stdin:
        line = convert(line)
        sys.stdout.write(line)


if __name__ == '__main__':
    main()
