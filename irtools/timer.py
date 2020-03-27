import timeit
from irtools.eprint import eprint


class Timer(object):
    def __init__(self, end_str=None, start_str=None):
        self.start_str = start_str
        self.end_str = end_str

    def __enter__(self):
        if self.start_str is not None:
            eprint(self.start_str)
        self.start = timeit.default_timer()
        return self

    def __exit__(self, *args):
        self.end = timeit.default_timer()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs
        if self.end_str is not None:
            eprint(f'{self.end_str}. time: {self.secs:.1f}s')
