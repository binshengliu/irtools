from typing import Any, Callable, List, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix


# Copied from https://stackoverflow.com/a/56175538/955952
def indep_roll(arr: npt.NDArray[Any], shifts: npt.NDArray[Any], axis: int = 1) -> npt.NDArray[Any]:
    """Apply an independent roll for each dimensions of a single axis.

    Parameters
    ----------
    arr : npt.NDArray[Any]
        Array of any shape.

    shifts : npt.NDArray[Any]
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


def len_to_mask(lens: npt.NDArray[Any], seq_len: Optional[int] = None) -> npt.NDArray[Any]:
    if seq_len is None:
        seq_len = max(lens)
    return np.arange(seq_len)[None, :] < lens[:, None]


def str_join(
    sep: Union[str, Callable[[Sequence[str]], str]],
    arrays: Sequence[npt.NDArray[Any]],
    axis: int = -1,
) -> npt.NDArray[Any]:
    def join_func(x: Sequence[str]) -> List[str]:
        if isinstance(sep, str):
            return [sep.join(x)]
        elif callable(sep):
            return [sep(x)]
        assert False

    tmp = np.concatenate(arrays, axis=axis)
    out = np.apply_along_axis(join_func, axis=axis, arr=tmp)
    return out


def pad_ragged(ragged_arrays: List[npt.NDArray[Any]]) -> npt.NDArray[Any]:
    assert all(x.ndim == 1 for x in ragged_arrays)
    count = np.array([x.size for x in ragged_arrays])
    max_count = count.max()
    pad_size = max_count - count
    pad_loc = np.cumsum(count) - 1

    arr: npt.NDArray[Any] = np.concatenate(ragged_arrays)
    repeats = np.ones_like(arr, dtype=int)
    repeats[pad_loc] = pad_size + 1
    arr = np.repeat(arr, repeats)
    arr = arr.reshape(len(ragged_arrays), max_count)
    return arr


def tile_csr(X: csr_matrix, reps: int) -> csr_matrix:
    rows = len(X.indptr) - 1
    indptr = np.tile(X.indptr[None, 1:], (reps, 1))
    indptr += (X.indptr[-1] * np.arange(reps))[:, None]
    indptr = np.concatenate([[0], indptr.reshape(-1)])
    assert rows * reps == len(indptr) - 1
    X = csr_matrix((np.tile(X.data, reps), np.tile(X.indices, reps), indptr))
    return X
