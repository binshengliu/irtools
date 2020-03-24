import timeit


class Timer(object):
    def __init__(self, desc=None):
        self.desc = desc

    def __enter__(self):
        if self.desc is not None:
            print(f'# start {self.desc}.')
        self.start = timeit.default_timer()
        return self

    def __exit__(self, *args):
        self.end = timeit.default_timer()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs
        if self.desc is not None:
            print(f'# end {self.desc}. time: {self.secs}s')
