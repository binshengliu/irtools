from concurrent.futures import ProcessPoolExecutor as Pool
import subprocess
import sys
from dask.distributed import Client
from more_itertools import unzip
import argparse
from pathlib import Path
from lxml import etree
import tempfile
import os
import string


def eprint(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr, flush=True)


class IndriRunQuery:
    def __init__(self, path, index, scheduler=None):
        self._path = path if path else 'IndriRunQuery'
        self._index = index.strip()
        self._scheduler = scheduler

    def format_xml(self, qno, query, working_set=[], extra=[]):
        root = etree.Element('parameters')

        node_format = etree.SubElement(root, 'trecFormat')
        node_format.text = 'true'

        node_index = etree.SubElement(root, 'index')
        node_index.text = self._index

        for el in extra:
            etree.SubElement(root, el[0]).text = str(el[1])

        node_query = etree.SubElement(root, 'query')

        etree.SubElement(node_query, 'number').text = qno
        etree.SubElement(node_query, 'text').text = query

        for docno in working_set:
            etree.SubElement(node_query, 'workingSetDocno').text = docno

        return etree.tostring(root, pretty_print=True).decode('ascii')

    def run_file(self, qno, query, working_set, extra=[]):
        string = self.format_xml(qno, query, working_set, extra)

        fp = tempfile.NamedTemporaryFile(mode='w')
        fp.write(string)
        fp.flush()

        indri_args = [self._path, fp.name]

        proc = subprocess.Popen(indri_args,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL,
                                encoding='ascii',
                                errors='ignore')
        output = []
        for line in proc.stdout:
            if 'EXCEPTION' in line:
                eprint('EXCEPTION: {} {} {}'.format(qno, query, line))
                raise Exception('EXCEPTION: {} {} {}'.format(qno, query, line))
            output.append(line)

        fp.close()

        output = ''.join(output)
        return output

    def run_cmd(self, qno, query, extra=[]):
        indri_args = [
            self._path, '-index=' + self._index, '-trecFormat=True',
            '-queryOffset={}'.format(qno), '-query={}'.format(query)
        ]

        if extra:
            extra = ['-{}={}'.format(*el) for el in extra]
            indri_args.extend(extra)

        proc = subprocess.Popen(indri_args,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL,
                                encoding='ascii',
                                errors='ignore')
        output = []
        for line in proc.stdout:
            if 'EXCEPTION' in line:
                eprint('EXCEPTION: {} {} {}'.format(qno, query, line))
                raise Exception('EXCEPTION: {} {} {}'.format(qno, query, line))
            output.append(line)

        output = ''.join(output)
        return output

    def run_single(self, qno, query, working_set=[], extra=[]):
        if working_set:
            output = self.run_file(qno, query, working_set, extra)
        else:
            output = self.run_cmd(qno, query, extra)

        return output

    def run_batch(self, qnos, queries, working_set=None, extra=None):
        if not working_set:
            working_set = [[]] * len(qnos)

        if not extra:
            extra = [[]] * len(qnos)

        with Pool(40) as pool:
            output = list(
                pool.map(self.run_single, qnos, queries, working_set, extra))
        return output

    def run_distributed(self, qnos, queries, working_set=None, extra=None):
        '''Set up a cluster first:
        dask-scheduler
        env PYTHONPATH=/research/remote/petabyte/users/binsheng/trec_tools/ dask-worker segsresap10:8786 --nprocs 50 --nthreads 1 --memory-limit 0 --name segsresap10
        env PYTHONPATH=/research/remote/petabyte/users/binsheng/trec_tools/ dask-worker segsresap10:8786 --nprocs 50 --nthreads 1 --memory-limit 0 --name segsresap09
        '''
        if not working_set:
            working_set = [[]] * len(qnos)

        if not extra:
            extra = [[]] * len(qnos)

        client = Client(self._scheduler)
        futures = client.map(self.run_single, qnos, queries, working_set,
                             extra)
        output = [f.result() for f in futures]
        return output


def parse_arguments():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument(
        '--index',
        required=True,
        type=Path,
        help='Index',
    )

    parser.add_argument(
        '--query',
        required=True,
        help='Query: csv file, or \'-\' for stdin. Format: qno,query',
    )

    parser.add_argument(
        '--sep',
        required=True,
        choices=[',', 'space', 'none'],
        help='Separator between qno and query',
    )

    parser.add_argument(
        '--scheduler',
        required=True,
        help='Index',
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    if os.path.exists(args.query):
        content = Path(args.query).read_text().splitlines()
    elif args.query == '-':
        content = sys.stdin.read().splitlines()
    else:
        eprint('Incorrect query')
        return

    if args.sep != 'none':
        sep = None if args.sep == 'space' else args.sep
        qnos, queries = unzip(line.split(sep, maxsplit=1) for line in content)
    else:
        queries = content
        qnos = list(map(str, range(len(queries))))

    trans = str.maketrans('', '', string.punctuation)
    queries = [s.translate(trans) for s in queries]

    qnos = list(qnos)
    queries = list(queries)

    indri = IndriRunQuery(None, str(args.index), args.scheduler)

    output = indri.run_distributed(qnos,
                                   queries,
                                   working_set=None,
                                   extra=[[['count', 5]]] * len(qnos))
    sys.stdout.writelines(output)


if __name__ == '__main__':
    main()
