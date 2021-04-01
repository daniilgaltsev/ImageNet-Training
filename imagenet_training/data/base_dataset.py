"""Base dataset class."""

from typing import Any, Optional, Callable, Union, Tuple, Sequence


import torch


class BaseDataset(torch.utils.data.Dataset):
    """Base dataset class that processes data with optional transforms.

    Args:
        data: A tensor or a sequence containing data to store.
        targets: A tensor or a sequence containing targets for the data.
        transform (optional): Optional transformation to apply to the data.
        transform_target (optional): Optional transformation to apply to the targets.
    """

    def __init__(
        self,
        data: Union[Sequence[Any], torch.Tensor],
        targets: Union[Sequence[Any], torch.Tensor],
        transform: Optional[Callable[..., Any]] = None,
        transform_target: Optional[Callable[..., Any]] = None,
    ):
        if len(data) != len(targets):
            raise ValueError(f"Data and targets are not of equal lengths: {len(data)} != {len(targets)}")
        self.data = data
        self.targets = targets
        self.transform = transform
        self.transform_target = transform_target

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        item = self.data[idx]
        target = self.targets[idx]

        if self.transform is not None:
            item = self.transform(item)

        if self.transform_target is not None:
            target = self.transform_target(target)

        return item, target
