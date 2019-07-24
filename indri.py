from concurrent.futures import ProcessPoolExecutor as Pool
import subprocess
import sys
from dask.distributed import Client


def eprint(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr, flush=True)


class IndriRunQuery:
    def __init__(self, path, index, scheduler=None):
        if not path:
            path = 'IndriRunQuery'
        self._args = [path, '-index=' + index.strip(), '-trecFormat=True']
        self._scheduler = scheduler

    def _run_single(self, qno, query, extra=None):
        indri_args = self._args + [
            '-trecFormat=True', '-queryOffset={}'.format(qno),
            '-query={}'.format(query)
        ]

        if extra:
            indri_args.extend(extra)

        proc = subprocess.Popen(
            indri_args,
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

    def run_batch(self, qnos, queries, extra=None):
        with Pool() as pool:
            output = list(pool.map(self._run_single, qnos, queries, extra))
        return output

    def run_distributed(self, qnos, queries, extra=None):
        '''Set up a cluster first:
        dask-scheduler
        env PYTHONPATH=/research/remote/petabyte/users/binsheng/trec_tools/ dask-worker segsresap10:8786 --nprocs 50 --nthreads 1 --memory-limit 0 --name segsresap10
        env PYTHONPATH=/research/remote/petabyte/users/binsheng/trec_tools/ dask-worker segsresap10:8786 --nprocs 50 --nthreads 1 --memory-limit 0 --name segsresap09
        '''
        client = Client(self._scheduler)
        futures = client.map(self._run_single, qnos, queries, extra)
        output = [f.result() for f in futures]
        return output


def main():
    import numpy as np
    import itertools
    indri = IndriRunQuery(
        None, '/research/remote/petabyte/users/indri-indexes/ROBUST04/ROB04',
        'segsresap10:8786')
    fbDocs = list(range(10, 201, 10))
    fbTerms = list(range(10, 201, 10))
    fbOrigs = list(np.linspace(0.0, 1.0, 11))
    extra = []
    filenames = []
    for d, t, o in itertools.product(fbDocs, fbTerms, fbOrigs):
        o = round(o, 1)
        extra.append([
            '-count=1000', '-fbDocs={}'.format(d), '-fbTerms={}'.format(t),
            '-fbOrigWeight={}'.format(o)
        ])
        filenames.append('d_{}_t_{}_w_{}.run'.format(d, t, o))
    print('Total: {}'.format(len(extra)))
    qnos = ['301'] * len(extra)
    queries = ['international organized crime'] * len(extra)
    output = indri.run_distributed(qnos, queries, extra)
    for one, fn in zip(output, filenames):
        with open(fn, 'w') as f:
            f.write(one)


if __name__ == '__main__':
    main()
