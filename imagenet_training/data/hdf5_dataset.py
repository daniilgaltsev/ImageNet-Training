"""HDF5 dataset class."""

from pathlib import Path
import random
from typing import Any, Callable, Generator, Union, Optional, Tuple, Sequence

import h5py
import torch
from torch.utils.data import IterableDataset, get_worker_info


class HDF5Dataset(IterableDataset):
    """HDF5 dataset class that processes data with optional transforms.

    Args:
        filename: Path to the hdf5 file.
        data_dataset_name: A name of a dataset in hdf5 with data.
        targets_dataset_name: A name of a dataset in hdf5 with targets for the data.
        worker_buffer_size: A size of a buffer used in each worker, which is preloaded and shuffled.
        transform (optional): Optional transformation to apply to the data.
        transform_target (optional): Optional transformation to apply to the targets.
    """

    def __init__(
        self,
        filename: Union[Path, str],
        data_dataset_name: Union[Sequence[Any], torch.Tensor],
        targets_dataset_name: Union[Sequence[Any], torch.Tensor],
        worker_buffer_size: int,
        transform: Optional[Callable[..., Any]] = None,
        transform_target: Optional[Callable[..., Any]] = None
    ):
        self.filename = filename
        self.data_dataset_name = data_dataset_name
        self.targets_dataset_name = targets_dataset_name
        self.worker_buffer_size = worker_buffer_size
        self.transform = transform
        self.transform_target = transform_target
        self.file = None
        self.buffer_data = []
        self.buffer_targets = []

        with h5py.File(filename, "r") as f:
            self.data_len = len(f[self.data_dataset_name])
            targets_len = len(f[self.targets_dataset_name])

        if self.data_len != targets_len:
            raise ValueError(f"Data and targets are not of equal lengths: {self.data_len} != {targets_len}")

        self.buffer_points = [
            (
                i,
                min(i + self.worker_buffer_size, self.data_len)
            ) for i in range(0, self.data_len, self.worker_buffer_size)
        ]

    def _open_file(self) -> None:
        """Opens the HDF5 file and specified datasets in it."""
        self.file = h5py.File(self.filename, "r")
        self.data = self.file[self.data_dataset_name]
        self.targets = self.file[self.targets_dataset_name]

    def _close_file(self) -> None:
        """Closes the HDF5 file if it's opened."""
        if self.file is not None:
            self.file.close()
            self.file = None

    def __len__(self) -> int:
        """Returns the length of the data."""
        return self.data_len

    def _load_buffer(self, buffer_point_idx: int) -> bool:
        """Loads and shuffles a buffer given its index.

        Args:
            buffer_point_ids: An index of a buffer to load corresponding to self.buffer_points.

        Returns:
            A boolean, which is True if reached end of data.
        """
        if buffer_point_idx >= len(self.buffer_points):
            self.buffer_data = []
            self.buffer_targets = []
            return True
        if self.file is None:
            self._open_file()
        start, end = self.buffer_points[buffer_point_idx]
        loaded = list(zip(self.data[start:end], self.targets[start:end]))
        random.shuffle(loaded)
        self.buffer_data, self.buffer_targets = zip(*loaded)
        return False

    def __iter__(self) -> Generator[Tuple[Any, Any], None, None]:
        """Iterates through the data with buffer loading and shuffling based on the id of a worker."""
        worker_info = get_worker_info()
        if worker_info is None:
            num_workers = 1
            worker_id = 0
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
        cur_buffer_segment = 0
        cur_buffer_idx = 0

        while True:
            if cur_buffer_idx == len(self.buffer_data):
                done = self._load_buffer(cur_buffer_segment + worker_id)
                if done:
                    break
                cur_buffer_idx = 0
                cur_buffer_segment += num_workers
            item = self.buffer_data[cur_buffer_idx]
            target = self.buffer_targets[cur_buffer_idx]
            if self.transform is not None:
                item = self.transform(item)
            if self.transform_target is not None:
                target = self.transform_target(target)
            yield item, target
            cur_buffer_idx += 1
        self._close_file()
