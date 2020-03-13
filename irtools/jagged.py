import numpy as np


def pad_jagged(jagged, pad_value=0, length=None, dtype=None):
    if length is None:
        lengths = np.array([len(x) for x in jagged])
        length = max(lengths)

    padded = np.full((len(jagged), length), pad_value, dtype=dtype)
    for idx, row in enumerate(jagged):
        padded[idx, :len(row)] = row

    return padded
