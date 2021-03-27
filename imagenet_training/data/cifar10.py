"""CIFAR-10 data module that downloads and saves data as .npz files if not already present."""

import argparse
import json
import os
from pathlib import Path
import pickle
import tarfile
from typing import Optional

import h5py
import numpy as np
from sklearn.model_selection import train_test_split
import toml
import torch
from torchvision import transforms

from imagenet_training.data.base_data_module import BaseDataModule
from imagenet_training.data.base_dataset import BaseDataset
from imagenet_training.data.utils import _download_raw_dataset


TRAIN_SPLIT = 0.8
SEED = 0
IMAGE_SHIFT = 0.0

RAW_DATA_DIRNAME = BaseDataModule.data_dirname() / "raw" / "cifar10"
METADATA_FILENAME = RAW_DATA_DIRNAME / "metadata.toml"
DL_DATA_DIRNAME = BaseDataModule.data_dirname() / "downloaded" / "cifar10"
PROCESSED_DATA_DIRNAME = BaseDataModule.data_dirname() / "processed" / "cifar10"
PROCESSED_DATA_FILENAME = PROCESSED_DATA_DIRNAME / "processed.h5"
ESSENTIALS_FILENAME = Path(__file__).parents[0].resolve() / "cifar10_essentials.json"


class CIFAR10(BaseDataModule):
    """CIFAR-10 data module that downloads and saves data as .npz files if not already present."""

    def __init__(self, args: Optional[argparse.Namespace] = None):
        super().__init__(args)

        if args is None:
            self.args = {}
        else:
            self.args = vars(args)
        self.seed = self.args.get("seed", SEED)

        if not os.path.exists(ESSENTIALS_FILENAME):
            _download_and_process_cifar10()
        with open(ESSENTIALS_FILENAME) as f:
            essentials = json.load(f)

        self.mapping = essentials['mapping']
        self.inverse_mapping = {v: k for k, v in self.mapping.items()}
        self.data_train = None
        self.data_val = None
        self.data_test = None
        transforms_list = [
            transforms.ToTensor(),
            # Calculated using imagenet_trianing.data.utils.calculate_mean_and_std
            transforms.Normalize([0.4918, 0.4824, 0.4467], [0.2470, 0.2434, 0.2617])
        ]
        self.transform_val = transforms.Compose(transforms_list[:])
        image_shift = self.args.get("image_shift", IMAGE_SHIFT)
        if image_shift != 0.0:
            transforms_list[1:1] = [transforms.RandomAffine(degrees=0, translate=(image_shift, image_shift))]
        self.transform_train = transforms.Compose(transforms_list)
        self.dims = (1, essentials['input_shape'][2], *essentials['input_shape'][:2])
        self.output_dims = (len(self.mapping),)

    def prepare_data(self) -> None:
        """Prepares data for the node globally."""
        if not os.path.exists(PROCESSED_DATA_FILENAME):
            _download_and_process_cifar10()
        with open(ESSENTIALS_FILENAME, "r") as f:
            self.essentials = json.load(f)

    def setup(self, stage: Optional[str] = None) -> None:
        """Prepares data for each process given stage."""
        if stage == 'fit' or stage is None:
            with h5py.File(PROCESSED_DATA_FILENAME, "r") as f:
                x_trainval = f['x_train'][:]
                y_trainval = f['y_train'][:].astype(np.int64)

            self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
                x_trainval, y_trainval,
                train_size=TRAIN_SPLIT, shuffle=True, random_state=self.seed, stratify=y_trainval
            )
            self.data_train = BaseDataset(self.x_train, self.y_train, transform=self.transform_train)
            self.data_val = BaseDataset(self.x_val, self.y_val, transform=self.transform_val)

        if stage == 'test' or stage is None:
            with h5py.File(PROCESSED_DATA_FILENAME, "r") as f:
                self.x_test = f['x_test'][:]
                self.y_test = f['y_test'][:].astype(np.int64)
            self.data_test = BaseDataset(self.x_test, self.y_test, transform=self.transform_val)

    @staticmethod
    def add_to_argparse(
        parser: argparse.ArgumentParser,
        main_parser: argparse.ArgumentParser
    ) -> argparse.ArgumentParser:
        """Adds arguments to parser required for the CIFAR10."""
        parser = BaseDataModule.add_to_argparse(parser, main_parser)
        parser.add_argument(
            "--image_shift", type=float, default=IMAGE_SHIFT, help="If not zero, will shift images by this fraction."
        )
        return parser


def _download_and_process_cifar10() -> None:
    """Downloads and processes CIFAR-10 dataset from a metadata file."""
    metadata = toml.load(METADATA_FILENAME)
    _download_raw_dataset(metadata, DL_DATA_DIRNAME)
    _process_raw_dataset(metadata["filename"], DL_DATA_DIRNAME)


def _process_raw_dataset(filename: str, dirname: Path) -> None:
    """Processes raw dataset at dirname/filename and saves .h5 files in processed."""
    path = dirname / filename

    print('Reading tar file contents.')
    contents = {}
    with tarfile.open(path, "r") as tar:
        for entry in tar:
            if entry.size == 0:
                continue
            f = tar.extractfile(entry)
            if entry.name.find('readme.html') == -1:
                contents[entry.name.split('/')[-1]] = pickle.load(f, encoding='bytes')  # nosec

    x_train, y_train = [], []
    x_test, y_test = None, None
    mapping = {}
    for key in contents:
        if key == 'batches.meta':
            for i, class_name in enumerate(contents[key][b'label_names']):
                mapping[i] = class_name.decode()
        else:
            labels = np.array(contents[key][b'labels'])
            data = contents[key][b'data'].reshape(-1, 3, 32, 32)
            data = np.transpose(data, (0, 2, 3, 1))

            if key.find('data_') == -1:
                x_test = data
                y_test = labels
            else:
                x_train.append(data)
                y_train.append(labels)
    x_train, y_train = np.concatenate(x_train), np.concatenate(y_train)

    print('Saving to HDF5 in a compressed format.')
    PROCESSED_DATA_DIRNAME.mkdir(exist_ok=True, parents=True)
    with h5py.File(PROCESSED_DATA_FILENAME, 'w') as f:
        f.create_dataset("x_train", data=x_train, dtype="u1", compression="lzf")
        f.create_dataset("y_train", data=y_train, dtype="u1", compression="lzf")
        f.create_dataset("x_test", data=x_test, dtype="u1", compression="lzf")
        f.create_dataset("y_test", data=y_test, dtype="u1", compression="lzf")

    print('Saving essential dataset parameters to imagenet_training/data')
    essentials = {"mapping": mapping, "input_shape": x_train.shape[1:]}
    with open(ESSENTIALS_FILENAME, "w") as f:
        json.dump(essentials, f)
