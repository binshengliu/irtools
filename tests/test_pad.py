from typing import Callable, Sequence, Union

import numpy as np
import pytest
import torch

from irtools.pad import pad_batch, pad_to


@pytest.mark.parametrize(  # type: ignore
    "cls", [np.array, torch.tensor]
)
@pytest.mark.parametrize(  # type: ignore
    "data,target,pad,label",
    [
        ([1, 2, 3], [4], 0, [1, 2, 3, 0]),
        ([1, 2, 3], [4], -1, [1, 2, 3, -1]),
        ([[1, 2, 3]], [2, 4], 0, [[1, 2, 3, 0], [0, 0, 0, 0]]),
        ([[1, 2, 3]], [2, 4], -1, [[1, 2, 3, -1], [-1, -1, -1, -1]]),
    ],
)
def test_pad_to(
    cls: Callable[..., Union[np.ndarray, torch.Tensor]],
    data: Union[Sequence[int], Sequence[Sequence[int]]],
    target: Sequence[int],
    pad: int,
    label: Union[Sequence[int], Sequence[Sequence[int]]],
) -> None:
    array = pad_to(cls(data), target, pad)
    assert np.array_equal(array.tolist(), label)


@pytest.mark.parametrize(  # type: ignore
    "cls", [np.array, torch.Tensor]
)
@pytest.mark.parametrize(  # type: ignore
    "data,pad,label",
    [
        ([[1, 2, 3], [1, 2], [1]], 0, [[1, 2, 3], [1, 2, 0], [1, 0, 0]]),
        ([[1, 2, 3], [1, 2], [1]], -1, [[1, 2, 3], [1, 2, -1], [1, -1, -1]]),
        (
            [[[1, 2], [3, 4], [5, 6]], [[1, 2, 3], [1, 2, 3]]],
            0,
            [[[1, 2, 0], [3, 4, 0], [5, 6, 0]], [[1, 2, 3], [1, 2, 3], [0, 0, 0]]],
        ),
        (
            [[[1, 2], [3, 4], [5, 6]], [[1, 2, 3], [1, 2, 3]]],
            -1,
            [
                [[1, 2, -1], [3, 4, -1], [5, 6, -1]],
                [[1, 2, 3], [1, 2, 3], [-1, -1, -1]],
            ],
        ),
    ],
)
def test_pad_batch(
    cls: Callable[..., Union[np.ndarray, torch.Tensor]],
    data: Union[Sequence[int], Sequence[Sequence[int]]],
    pad: int,
    label: Union[Sequence[int], Sequence[Sequence[int]]],
) -> None:
    array = pad_batch([cls(x) for x in data], pad)
    assert np.array_equal([x.tolist() for x in array], label)
