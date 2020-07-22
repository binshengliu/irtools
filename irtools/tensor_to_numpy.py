from typing import Dict, List, Sequence, Union, Mapping, Any

from collections import abc
import numpy as np
import torch

T = Union[torch.Tensor, Sequence[torch.Tensor], Mapping[Any, torch.Tensor]]
N = Union[np.ndarray, List[np.ndarray], Dict[Any, np.ndarray]]


def tensor_to_numpy(
    data: Union[torch.Tensor, Sequence[T], Mapping[Any, T]]
) -> Union[np.ndarray, List[N], Dict[Any, N]]:
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    elif isinstance(data, abc.Mapping):
        return {k: tensor_to_numpy(v) for k, v in data.items()}
    elif isinstance(data, abc.Sequence):
        return [tensor_to_numpy(x) for x in data]
    else:
        raise TypeError(f"Unknown data type {type(data)}")
