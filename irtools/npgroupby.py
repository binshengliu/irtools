import typing
import numpy as np


def npgroupby(data: typing.Union[list, np.ndarray],
              by: typing.Union[int, list, tuple, np.ndarray]):
    if isinstance(by, int):
        by = [by]

    data = np.asarray(data)
    assert data.ndim == 2

    data = data[np.lexsort([data[:, i] for i in by[::-1]])]
    split_idx = np.where(data[1:, 0] != data[:-1, 0])[0] + 1
    groups = np.split(data, split_idx)
    return groups


def test_groupby():
    data = [[4, 4], [1, 2], [2, 3], [1, 3], [1, 4], [2, 4], [3, 4], [3, 5]]
    groups = npgroupby(data, by=0)
    assert len(groups) == 4
    assert groups[0][0, 0] == 1
