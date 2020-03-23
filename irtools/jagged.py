import numpy as np


def pad_jagged(jagged, pad_value=0, max_len=None, dtype=None):
    act_max_len = max(len(x) for x in jagged)
    if max_len is not None:
        max_len = min(act_max_len, max_len)
    else:
        max_len = act_max_len

    padded = np.full((len(jagged), max_len), pad_value, dtype=dtype)
    for idx, row in enumerate(jagged):
        act_len = min(len(row), max_len)
        padded[idx, :act_len] = row[:act_len]

    return padded
