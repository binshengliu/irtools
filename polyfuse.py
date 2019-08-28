import tempfile
import subprocess
import os
from pathlib import Path
from itertools import filterfalse


class Polyfuse:
    def __init__(self, path):
        "docstring"
        self._path = path

    def _write_tmp(self, *doclists):
        fps = [tempfile.NamedTemporaryFile(mode='w') for _ in doclists]
        for fp, doclist in zip(fps, doclists):
            buffer = []
            for rank, doc in enumerate(doclist):
                line = '1 Q0 {} {} {} N\n'.format(doc, rank, -rank)
                buffer.append(line)
            fp.writelines(buffer)
        return fps

    def _write_buffers(self, *buffers):
        fps = [tempfile.NamedTemporaryFile(mode='w') for _ in buffers]
        for fp, buffer in zip(fps, buffers):
            fp.write(buffer)
        return fps

    def rrf_files(self, files, weights=None):
        nonexist = list(filterfalse(os.path.exists, files))
        if nonexist:
            raise Exception('Invalid paths {}'.format(' '.join(nonexist)))

        if weights and len(files) != len(weights):
            raise Exception('Files and weights should match {} != {}'.format(
                len(files), len(weights)))

        if len(files) == 1:
            return Path(files[0]).read_text()

        args = [self._path, 'rrf', *files]
        if weights:
            args.insert(2, '-w {}'.format(','.join(map(str, weights))))

        proc = subprocess.run(args, stdout=subprocess.PIPE)
        output = proc.stdout.decode('utf-8')
        return output

    def rrf_buffers(self, buffers, weights=None):
        fps = self._write_buffers(*buffers)
        output = self.rrf_files([fp.name for fp in fps], weights)
        for fp in fps:
            fp.close()

        return output
