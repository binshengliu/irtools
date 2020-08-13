from typing import Optional

import numpy as np


# Copied from https://stackoverflow.com/a/56175538/955952
def indep_roll(arr: np.ndarray, shifts: np.ndarray, axis: int = 1) -> np.ndarray:
    """Apply an independent roll for each dimensions of a single axis.

    Parameters
    ----------
    arr : np.ndarray
        Array of any shape.

    shifts : np.ndarray
        How many shifting to use for each dimension. Shape: `(arr.shape[axis],)`.

    axis : int
        Axis along which elements are shifted.
    """
    arr = np.swapaxes(arr, axis, -1)
    all_idcs = np.ogrid[[slice(0, n) for n in arr.shape]]

    # Convert to a positive shift
    shifts[shifts < 0] += arr.shape[-1]
    all_idcs[-1] = all_idcs[-1] - shifts[:, np.newaxis]

    result = arr[tuple(all_idcs)]
    arr = np.swapaxes(result, -1, axis)
    return arr


def len_to_mask(lens: np.ndarray, seq_len: Optional[int] = None) -> np.ndarray:
    if seq_len is None:
        seq_len = max(lens)
    return np.arange(seq_len)[None, :] < lens[:, None]
