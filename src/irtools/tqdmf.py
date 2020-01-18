from tqdm import tqdm
import os


def tqdmf(path, *args, **kwargs):
    with tqdm(
            total=os.stat(path).st_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            *args,
            **kwargs) as bar:
        with open(path, 'r') as f:
            for line in f:
                bar.update(len(line))
                yield line
