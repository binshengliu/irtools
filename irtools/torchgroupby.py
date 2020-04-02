import typing
import torch


def torchgroupby(data: typing.Union[list, torch.Tensor],
                 by: int,
                 return_index: bool = False):
    data = torch.as_tensor(data)
    assert data.ndim == 2

    idx = torch.argsort(data[:, by])
    ordered = data[idx]
    split_idx = (ordered[1:, by] != ordered[:-1, by]).nonzero(
        as_tuple=True)[0] + 1

    arr1 = torch.cat((split_idx, split_idx.new_tensor([len(ordered)])))
    arr2 = torch.cat((split_idx.new_tensor([0]), split_idx))

    splits = arr1 - arr2
    groups = torch.split(ordered, splits.tolist())
    if return_index:
        idx_groups = torch.split(idx, splits.tolist())
        return groups, idx_groups
    else:
        return groups


def test_groupby():
    data = [[4, 4], [1, 2], [2, 3], [1, 3], [1, 4], [2, 4], [3, 4], [3, 5]]
    groups = torchgroupby(data, by=0)
    assert len(groups) == 4
    assert torch.equal(groups[0], torch.tensor([[1, 2], [1, 3], [1, 4]]))
    assert torch.equal(groups[1], torch.tensor([[2, 3], [2, 4]]))
    assert torch.equal(groups[2], torch.tensor([[3, 4], [3, 5]]))


def test_return_index():
    data = [[4, 4], [1, 2], [2, 3], [1, 3], [1, 4], [2, 4], [3, 4], [3, 5]]
    _, idx = torchgroupby(data, by=0, return_index=True)
    assert len(idx) == 4
    assert torch.equal(idx[0], torch.tensor([1, 3, 4]))
    assert torch.equal(idx[1], torch.tensor([2, 5]))
    assert torch.equal(idx[2], torch.tensor([6, 7]))
