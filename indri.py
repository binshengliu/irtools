from concurrent.futures import ProcessPoolExecutor as Pool
import subprocess
import sys


def eprint(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr, flush=True)


class IndriRunQuery:
    def __init__(self, path, index):
        if not path:
            path = 'IndriRunQuery'
        self._args = [path, '-index=' + index.strip(), '-trecFormat=True']

    def _run_single(self, qno, query, extra=None):
        indri_args = self._args + [
            '-trecFormat=True', '-queryOffset={}'.format(qno),
            '-query={}'.format(query)
        ]

        if extra:
            indri_args.extend(extra)

        proc = subprocess.Popen(
            indri_args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        output = []
        for line in proc.stdout:
            line = line.decode('utf-8')
            if 'EXCEPTION' in line:
                eprint('EXCEPTION: {} {} {}'.format(qno, query, line))
                raise Exception('EXCEPTION: {} {} {}'.format(qno, query, line))
            output.append(line)

        output = ''.join(output)
        return output

    def run_batch(self, qnos, queries, extra=None):
        count = len(qnos)
        with Pool() as pool:
            output = list(
                pool.map(self._run_single, qnos, queries, [extra] * count))
        return output
