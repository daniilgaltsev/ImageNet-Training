"""Utility functions for data module."""

from pathlib import Path
from typing import Dict

import pytorch_lightning as pl
import torch
import tqdm

import imagenet_training.utils


def _download_raw_dataset(metadata: Dict[str, str], dl_path: Path) -> Path:
    """Downloads data based on metadata to dl_path."""
    dl_path.mkdir(parents=True, exist_ok=True)
    filename = dl_path / metadata["filename"]
    if filename.exists():
        return filename

    print("Downloading data from {} to {}.".format(metadata["url"], filename))
    imagenet_training.utils.download_url(metadata["url"], filename)
    print("Computing SHA256.")
    sha256 = imagenet_training.utils.compute_sha256(filename)
    print("Downloaded data sha256={}".format(sha256))
    if sha256 != metadata["sha256"]:
        raise ValueError("Downloaded data checksum does not match the one specified in metadata.")

    return filename


def calculate_mean_and_std(data_module: pl.LightningDataModule) -> torch.Tensor:
    """Calculates mean and std of an image dataset.

    Adopted from http://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html.

    Args:
        data_module: an image data module for which to calculate mean and std.

    Returns:
        A tensor of mean and std (shape=(2, 3)).
    """
    data_module.prepare_data()
    data_module.setup("fit")
    stats = torch.zeros((2, 3))
    n_samples = 0
    for batch in tqdm.tqdm(data_module.train_dataloader()):
        x = batch[0]
        batch_size = x.size(0)
        new_n_samples = n_samples + batch_size

        mean = x.mean((0, 2, 3))
        std = x.std((0, 2, 3))
        tmp = stats[0].clone()

        stats[0] = n_samples / new_n_samples * tmp + batch_size / new_n_samples * mean
        stats[1] = n_samples / new_n_samples * stats[1] ** 2 + batch_size / new_n_samples * std ** 2 + \
            n_samples * batch_size / new_n_samples ** 2 * (tmp - mean) ** 2
        stats[1] = torch.sqrt(stats[1])

        n_samples = new_n_samples

    return stats
