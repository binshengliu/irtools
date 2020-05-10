from typing import List, Sequence, TypeVar

import numpy as np
import torch
import torch.nn.functional as F

T = TypeVar("T", np.ndarray, torch.Tensor)


def pad_to(data: T, target_size: Sequence[int], value: int) -> T:
    if isinstance(data, torch.Tensor):
        diffs = torch.tensor(target_size) - torch.tensor(data.size())
        if diffs.sum() == 0:
            return data
        pt_pad_size = torch.tensor(
            [[0, diff] for diff in diffs.tolist()[::-1]]
        ).reshape(-1)
        return F.pad(data, pt_pad_size.tolist(), value=value)
    elif isinstance(data, np.ndarray):
        diffs = np.array(target_size) - np.array(data.shape)
        if diffs.sum() == 0:
            return data
        np_pad_size = [(0, diff) for diff in diffs]
        return np.pad(data, pad_width=np_pad_size, constant_values=value)
    else:
        assert False


def pad_batch(batch: Sequence[T], value: int) -> List[T]:
    if isinstance(batch[0], torch.Tensor):
        target_size = torch.tensor([x.size() for x in batch]).max(dim=0)[0]
        assert isinstance(target_size, torch.Tensor)
    elif isinstance(batch[0], np.ndarray):
        target_size = np.array([x.shape for x in batch]).max(axis=0)
        assert isinstance(target_size, np.ndarray)
    else:
        assert False

    batch = [pad_to(x, target_size.tolist(), value) for x in batch]
    return batch
