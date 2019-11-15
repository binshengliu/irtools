import string
import spacy
import ftfy
import sys

nlp = spacy.load('en_core_web_lg', disable=['tagger', 'parser', 'ner'])
trans_table = str.maketrans('', '', string.punctuation)


def convert_doc(doc):
    docno, content = doc.strip().split(maxsplit=1)
    content = ftfy.fix_text(content)
    content = ''.join(filter(lambda x: x in string.printable, content))
    content = nlp(content)
    content = [x.text.translate(trans_table) for x in content]
    content = [x for x in content if x]
    content = ' '.join(content).lower()
    template = '<DOC>\n<DOCNO>{}</DOCNO>\n<TEXT>\n{}\n</TEXT>\n</DOC>\n'
    output = template.format(docno, content)
    return output


def main():
    for doc in sys.stdin:
        doc = convert_doc(doc)
        sys.stdout.write(doc)


if __name__ == '__main__':
    main()
