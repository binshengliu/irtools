import os
from collections import abc
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

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
    elif type(data).__module__ == "numpy":
        return data.tolist()
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
        tensor.shape[1], device=tensor.device
    )
    out_mask = out_mask[None, :] < length[:, None]
    out = tensor.new_zeros(tensor.shape)
    out[out_mask] = tensor[in_mask]
    out[~out_mask] = tensor[~in_mask]

    if not keep_shape:
        out = out[:, : length.max()]  # type: ignore
    return out


def str_to_byte_tensor(
    list_s: Sequence[str], max_len: int = -1, device: Any = None
) -> torch.ByteTensor:
    max_len = max(max(len(x) for x in list_s), max_len)
    output: torch.ByteTensor = torch.zeros(
        (len(list_s), max_len), dtype=torch.uint8, device=device
    )
    for i in range(len(list_s)):
        tensor = torch.tensor(list(bytes(list_s[i], "ascii")))
        output[i][: len(tensor)] = tensor

    return output


def byte_tensor_to_str(input: torch.Tensor) -> List[str]:
    shape = input.shape
    input = input.reshape(-1, shape[-1])
    output = [""] * len(input)
    lengths = (input != 0).sum(dim=-1).tolist()
    for i, one in enumerate(input[input != 0].cpu().split(lengths)):
        one = one[one != 0]
        string = bytes(one.tolist()).decode("ascii")
        output[i] = string
    output = np.array(output).reshape(*shape[:-1]).tolist()
    return output


def all_gather_str(
    v: Sequence[str], device: Any, world_size: Optional[int] = None
) -> np.ndarray:
    world_size = world_size or torch.distributed.get_world_size()  # type: ignore
    t = str_to_byte_tensor(v, 32, device)
    target = [t.new_zeros(t.shape) for _ in range(world_size)]
    torch.distributed.all_gather(target, t)  # type: ignore
    tensor = torch.cat(target)  # type: ignore
    strings = np.array(byte_tensor_to_str(tensor))
    return strings


def all_gather(v: torch.Tensor, world_size: Optional[int] = None) -> torch.Tensor:
    world_size = world_size or torch.distributed.get_world_size()  # type: ignore
    target = [v.new_zeros(v.shape) for _ in range(world_size)]
    torch.distributed.all_gather(target, v)  # type: ignore
    return torch.cat(target)  # type: ignore
