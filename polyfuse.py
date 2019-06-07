import tempfile
import subprocess


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

    def rrf_files(self, *runfiles):
        args = [self._path, 'rrf', *runfiles]
        proc = subprocess.run(args, stdout=subprocess.PIPE)
        output = proc.stdout.decode('utf-8')
        return output

    def rrf_buffers(self, *buffers):
        # In some cases, if the user uses zip_longest to parallel
        # fusion, there may be some None values.
        buffers = [_ for _ in buffers if _]
        fps = self._write_buffers(*buffers)
        output = self.rrf_files(*[fp.name for fp in fps])
        for fp in fps:
            fp.close()

        return output
