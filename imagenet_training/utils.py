"""Utility functions for imagenet_training module"""

import hashlib
from pathlib import Path
from tqdm import tqdm
from typing import Union
from urllib.request import urlretrieve


def compute_sha256(filename: Union[Path, str]) -> str:
    """
    Computes sha256 checksum of a file.

    Args:
        filename: path to the file.
    """

    with open(filename, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def download_url(url: str, filename: Union[Path, str]) -> None:
    """
    Downloads data from url.

    Args:
        url: url to the data to download.
        filename: path to the download location.
    """

    with TqdmUpTo(unit="B", unit_scale=True, unit_divisor=1024, miniters=1) as t:
        urlretrieve(url, filename, reporthook=t.update_to)


class TqdmUpTo(tqdm):
    """From https://github.com/tqdm/tqdm/blob/master/examples/tqdm_wget.py"""

    def update_to(self, blocks=1, bsize=1, tsize=None):
        """
        Args:
            blocks (optional): Number of blocks transferred so far [default: 1].
            bsize (optional): Size of each block (in tqdm units) [default: 1].
            tsize (optional): Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize  # pylint: disable=attribute-defined-outside-init
        self.update(blocks * bsize - self.n)  # will also set self.n = b * bsize