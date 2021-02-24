"""Utility functions for data module"""

from pathlib import Path
from typing import Dict

import imagenet_training.utils


def _download_raw_dataset(metadata: Dict[str, str], dl_path: Path) -> Path:
    """Downloads data based on metadata to dl_path"""

    dl_path.mkdir(parents=True, exist_ok=True)
    filename = dl_path / metadata["filename"]
    if filename.exists():
        return

    print("Downloading data from {} to {}.".format(metadata["url"], filename))
    imagenet_training.utils.download_url(metadata["url"], filename)
    print("Computing SHA256.")
    sha256 = imagenet_training.utils.compute_sha256(filename)
    print("Downloaded data sha256={}".format(sha256))
    if sha256 != metadata["sha256"]:
        raise ValueError("Downloaded data checksum does not match the one specified in metadata.")

    return filename

