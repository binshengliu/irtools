from collections import abc
from typing import Any, Dict, List, Mapping, Sequence, Union

import numpy as np
import torch

# Keep an eye on https://github.com/python/mypy/issues/731 when mypy supports recursive
# types.
T = Union[torch.Tensor, Sequence[torch.Tensor], Mapping[Any, torch.Tensor]]
N = Union[np.ndarray, List[np.ndarray], Dict[Any, np.ndarray]]


def tensor_to_numpy(
    data: Union[torch.Tensor, Sequence[T], Mapping[Any, T]]
) -> Union[np.ndarray, List[N], Dict[Any, N]]:
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    elif isinstance(data, abc.Mapping):
        return {k: tensor_to_numpy(v) for k, v in data.items()}
    elif isinstance(data, abc.Sequence) and not isinstance(data, (bytes, str)):
        return [tensor_to_numpy(x) for x in data]
    else:
        return data


def tensor_to_primitive(data: Union[torch.Tensor, Sequence[T], Mapping[Any, T]]) -> Any:
    if isinstance(data, torch.Tensor):
        return data.cpu().tolist()
    elif isinstance(data, abc.Mapping):
        return {k: tensor_to_primitive(v) for k, v in data.items()}
    elif isinstance(data, abc.Sequence) and not isinstance(data, (bytes, str)):
        return [tensor_to_primitive(x) for x in data]
    else:
        return data


def masked_align(
    tensor: torch.Tensor, mask: torch.Tensor, pad: Any = 0, keep_shape: bool = False
) -> torch.Tensor:
    in_mask = mask.bool()
    length = in_mask.int().sum(dim=1)
    out_mask: torch.Tensor = torch.arange(  # type: ignore
        length.max(), device=tensor.device
    )
    out_mask = out_mask[None, :] < length[:, None]
    if keep_shape:
        new_shape = (*tensor.shape,)
    else:
        new_shape = (tensor.shape[0], length.max().item(), *tensor.shape[2:])
    out = tensor.new_full(new_shape, pad)
    out[out_mask] = tensor[in_mask]
    return out
