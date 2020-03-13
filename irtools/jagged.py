import numpy as np


def pad_jagged(jagged, pad_value=0, return_length=False):
    lengths = np.array([len(x) for x in jagged])
    maxlen = max(lengths)

    padded = np.full((len(jagged), maxlen), pad_value)
    for idx, row in enumerate(jagged):
        padded[idx, :len(row)] = row

    if return_length:
        return padded, lengths

    return padded
