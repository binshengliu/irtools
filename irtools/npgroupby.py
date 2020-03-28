import typing
import numpy as np


def npgroupby(data: typing.Union[list, np.ndarray],
              by: typing.Union[int, list, tuple, np.ndarray],
              return_index: bool = False):
    if isinstance(by, int):
        by = [by]

    data = np.asarray(data)
    assert data.ndim == 2

    idx = np.lexsort([data[:, i] for i in by[::-1]])
    ordered = data[idx]
    split_idx = (ordered[1:, by] != ordered[:-1, by]).nonzero()[0] + 1
    groups = np.split(ordered, split_idx)
    if return_index:
        idx_groups = np.split(idx, split_idx)
        return groups, idx_groups
    else:
        return groups


def test_groupby():
    data = [[4, 4], [1, 2], [2, 3], [1, 3], [1, 4], [2, 4], [3, 4], [3, 5]]
    groups = npgroupby(data, by=0)
    assert len(groups) == 4
    assert np.array_equal(groups[0], [[1, 2], [1, 3], [1, 4]])
    assert np.array_equal(groups[1], [[2, 3], [2, 4]])
    assert np.array_equal(groups[2], [[3, 4], [3, 5]])


def test_return_index():
    data = [[4, 4], [1, 2], [2, 3], [1, 3], [1, 4], [2, 4], [3, 4], [3, 5]]
    _, idx = npgroupby(data, by=0, return_index=True)
    assert len(idx) == 4
    assert np.array_equal(idx[0], [1, 3, 4])
    assert np.array_equal(idx[1], [2, 5])
    assert np.array_equal(idx[2], [6, 7])


def test_groupby_multi():
    data = [[1, 2, 1], [2, 3, 1], [1, 2, 3], [1, 3, 2]]
    groups = npgroupby(data, by=[0, 1])
    assert len(groups) == 3
    assert np.array_equal(groups[0], [[1, 2, 1], [1, 2, 3]])
    assert np.array_equal(groups[1], [[1, 3, 2]])
    assert np.array_equal(groups[2], [[2, 3, 1]])
