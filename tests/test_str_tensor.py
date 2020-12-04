from typing import List

import pytest
from irtools.pytorch_recipes import byte_tensor_to_str, str_to_byte_tensor


@pytest.mark.parametrize(  # type: ignore
    "inputs", [["hello", "world", "byte_tensor_to_str, str_to_byte_tensor"]],
)
def test_str_tensor(inputs: List[str],) -> None:
    assert inputs == byte_tensor_to_str(str_to_byte_tensor(inputs, 100))
