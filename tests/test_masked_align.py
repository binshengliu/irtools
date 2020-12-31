from typing import List

import pytest
import torch
from irtools.pytorch_recipes import masked_align


@pytest.mark.parametrize(  # type: ignore
    "input,mask,target",
    [
        (
            [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
            [[1, 1, 0], [1, 0, 1], [0, 1, 0]],
            [[1, 2, 3], [1, 3, 2], [2, 1, 3]],
        ),
        (
            [
                [[1, 2], [2, 3], [3, 4]],
                [[1, 1], [2, 2], [3, 3]],
                [[1, 1], [2, 2], [3, 3]],
            ],
            [[1, 1, 0], [1, 0, 1], [0, 1, 0]],
            [
                [[1, 2], [2, 3], [3, 4]],
                [[1, 1], [3, 3], [2, 2]],
                [[2, 2], [1, 1], [3, 3]],
            ],
        ),
    ],
)
def test_masked_align(
    input: List[List[int]], mask: List[List[int]], target: List[List[int]],
) -> None:
    result = masked_align(
        torch.tensor(input), torch.tensor(mask), keep_shape=True  # type: ignore
    )
    assert torch.equal(result, torch.tensor(target))  # type: ignore


@pytest.mark.parametrize(  # type: ignore
    "input,mask,target",
    [
        (
            [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
            [[1, 1, 0], [1, 0, 1], [0, 1, 0]],
            [[1, 2], [1, 3], [2, 1]],
        ),
        (
            [
                [[1, 2], [2, 3], [3, 4]],
                [[1, 1], [2, 2], [3, 3]],
                [[1, 1], [2, 2], [3, 3]],
            ],
            [[1, 1, 0], [1, 0, 1], [0, 1, 0]],
            [[[1, 2], [2, 3]], [[1, 1], [3, 3]], [[2, 2], [1, 1]]],
        ),
    ],
)
def test_masked_align_strip(
    input: List[List[int]], mask: List[List[int]], target: List[List[int]],
) -> None:
    result = masked_align(
        torch.tensor(input), torch.tensor(mask)  # type: ignore
    )
    assert torch.equal(result, torch.tensor(target))  # type: ignore
