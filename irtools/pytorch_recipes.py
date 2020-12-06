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


def str_to_byte_tensor(
    list_s: Sequence[str], max_len: int = -1, device: Any = None
) -> torch.ByteTensor:
    tensors = [
        torch.tensor(  # type: ignore
            list(bytes(s, "utf-8")),
            dtype=torch.uint8,  # type: ignore
            device=device,
        )
        for s in list_s
    ]
    max_len = max(max(x.shape[0] for x in tensors), max_len)
    output: torch.ByteTensor = torch.zeros(  # type: ignore
        (len(tensors), max_len), dtype=torch.uint8, device=device  # type: ignore
    )
    for i in range(output.shape[0]):
        output[i][: tensors[i].shape[0]] = tensors[i]
    return output


def byte_tensor_to_str(input: torch.ByteTensor) -> List[str]:
    output = []
    for one in input:
        one = one[one != 0]
        string = bytes(one.tolist()).decode("utf-8")
        output.append(string)
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
