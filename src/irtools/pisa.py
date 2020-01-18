import subprocess
from more_itertools import pairwise
import numpy as np
from collections import OrderedDict


class PisaBmw:
    def __init__(self, path, index):
        if not path:
            path = 'evaluate_queries'

        self._args = [
            path, '-t', 'opt', '-a', 'block_max_wand', '-i',
            index + '.index.opt', '-w', index + '.wand',
            '--documents', index + '.doclex', '--terms',
            index + '.termlex'
        ]

    def run_batch(self, qnos, queries, extra=None):
        '''Take an dict of {qno: query}. Return a dict {qno: [doc1, doc2]}'''
        if not isinstance(queries, str):
            formatted = ''.join(
                ['{}:{}\n'.format(no, q) for no, q in zip(qnos, queries)])

        args = self._args
        if extra is not None:
            extra = [str(_) for _ in extra]
            args = args + extra

        proc = subprocess.run(
            args,
            input=formatted.encode(),
            stdout=subprocess.PIPE)

        output = proc.stdout.decode('utf-8')
        output = output.splitlines()
        zeroes = [i for i, line in enumerate(output) if line.split()[3] == '0']
        extents = pairwise(zeroes + [None])
        buffers = ['\n'.join(output[begin:end]) + '\n' for begin, end in extents]
        return buffers


class PisaCorpora:
    def __init__(self, collection_name):
        self.corpora = np.memmap(collection_name, dtype=np.uint32, mode='r')
        doc_start = self.build_start_index()

        with open(collection_name + '.documents', 'r') as f:
            docnos = f.read().split()
            self.documents = OrderedDict(zip(docnos, doc_start))

        with open(collection_name + '.terms', 'r') as f:
            self.terms = np.array(f.read().split())

    def _doc_text(self, docno):
        start = self.documents[docno]
        size = self.corpora[start]
        seq = self.corpora[start + 1:start + 1 + size]
        return ' '.join(self.terms[seq])

    def doc_text(self, docnos):
        return list(map(self._doc_text, docnos))

    def batch_iter(self, batch_size, random):
        pass

    def build_start_index(self):
        i = 2
        starting = []
        while i < len(self.corpora):
            size = self.corpora[i]
            starting.append(i)
            i += size + 1
        return starting

    def __len__(self):
        return self.corpora[1]

    def __iter__(self):
        for docno in self.documents:
            yield (docno, self.doc_text(docno))

    def __next__(self):
        return self
