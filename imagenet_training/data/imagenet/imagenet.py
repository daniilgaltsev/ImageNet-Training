"""ImageNet data module."""


import argparse
import json
import multiprocessing as mp
import os
from time import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

from imagenet_training.data.base_data_module import BaseDataModule
from imagenet_training.data.hdf5_dataset import HDF5Dataset
from .download_urls import parse_synset_mapping, download_image_urls, get_synset_stats
from .download_images import download_subsampled_images
from .process_downloaded import DownloadedProcessor


SEED = 0
USE_KAGGLE = False
SUBSAMPLE_CLASSES = 500
IMAGES_PER_CLASS = 100
REWRITE = False
PROCESSING_WORKERS = mp.cpu_count() - 1
PROCESSING_BATCH_SIZE = 5000
TRAINVAL_SPLIT = 0.9
TRAIN_SPLIT = 0.9
RESIZE_SIZE = (256, 256)

CROP_SIZE = 224
# Mean and std from https://pytorch.org/vision/0.8/models.html
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
WORKER_BUFFER_SIZE = 1024

RAW_DATA_DIRNAME = BaseDataModule.data_dirname() / "raw" / "imagenet"
METADATA_FILENAME = RAW_DATA_DIRNAME / "metadata.toml"
SYNSET_MAPPING_FILENAME = RAW_DATA_DIRNAME / "LOC_synset_mapping.txt"
URLS_FILENAME = RAW_DATA_DIRNAME / "urls.json"
DL_DATA_DIRNAME = BaseDataModule.data_dirname() / "downloaded" / "imagenet"
DATASET_KAGGLE_FILENAME = DL_DATA_DIRNAME / "imagenet_object_localization_patched2019.tar.gz"
IMAGES_DATA_DIRNAME = DL_DATA_DIRNAME / "images"
PROCESSED_DATA_DIRNAME = BaseDataModule.data_dirname() / "processed" / "imagenet"
PROCESSED_DATA_FILENAME = PROCESSED_DATA_DIRNAME / "processed.h5"
ESSENTIALS_FILENAME = PROCESSED_DATA_DIRNAME / "imagenet_essentials.json"


class ImageNet(BaseDataModule):
    """ImageNet data module."""

    def __init__(self, args: Optional[argparse.Namespace] = None):
        super().__init__(args)

        if args is None:
            self.args = {}
        else:
            self.args = vars(args)

        self.subsample_classes = self.args.get("subsample_classes", SUBSAMPLE_CLASSES)
        self.images_per_class = self.args.get("images_per_class", IMAGES_PER_CLASS)
        self.rewrite = self.args.get("rewrite", REWRITE)
        self.use_kaggle = self.args.get("use_kaggle", USE_KAGGLE)
        self.processing_workers = self.args.get("processing_workers", PROCESSING_WORKERS)
        self.worker_buffer_size = self.args.get("worker_buffer_size", WORKER_BUFFER_SIZE)

        if not os.path.exists(ESSENTIALS_FILENAME):
            _process_imagenet(
                classes=self.subsample_classes,
                images_per_class=self.images_per_class,
                use_kaggle=self.use_kaggle,
                rewrite=self.rewrite,
                processing_workers=self.processing_workers
            )
        with open(ESSENTIALS_FILENAME) as f:
            essentials = json.load(f)

        self.synset_to_class = essentials["synset_to_class"]
        self.synset_to_name = essentials["synset_to_name"]
        self.mapping = {
            self.synset_to_class[synset]: self.synset_to_name[synset] for synset in self.synset_to_class
        }
        self.dims = (1, essentials["input_size"][2], *essentials["input_size"][:2])
        self.output_dims = (len(self.synset_to_class),)
        self.data_train = None
        self.data_val = None
        self.data_test = None
        self.train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(CROP_SIZE),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])
        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(CROP_SIZE),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])

    @staticmethod
    def add_to_argparse(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Adds arguments to parser required for the ImageNet."""
        parser = BaseDataModule.add_to_argparse(parser)
        parser.add_argument(
            "--subsample_classes",
            type=int,
            default=SUBSAMPLE_CLASSES,
            help="A number of classes to use."
        )
        parser.add_argument(
            "--images_classes",
            type=int,
            default=IMAGES_PER_CLASS,
            help="A number of images per class to use."
        )
        parser.add_argument(
            "--rewrite",
            default=False,
            action="store_true",
            help="If True, will rewrite existing processing file."
        )
        parser.add_argument(
            "--use_kaggle",
            default=False,
            action="store_true",
            help="If True, will process kaggle's .tar.gz file."
        )
        parser.add_argument(
            "--processing_workers",
            type=int,
            default=PROCESSING_WORKERS,
            help="A number of subprocesses to use during processing."
        )
        parser.add_argument(
            "--worker_buffer_size",
            type=int,
            default=WORKER_BUFFER_SIZE,
            help="A size of the buffer of each dataloader worker."
        )
        return parser

    def prepare_data(self) -> None:
        """Prepares data for the node globally."""
        if not os.path.exists(PROCESSED_DATA_FILENAME):
            _process_imagenet(
                classes=self.subsample_classes,
                images_per_class=self.images_per_class,
                use_kaggle=self.use_kaggle,
                rewrite=self.rewrite,
                processing_workers=self.processing_workers
            )
        with open(ESSENTIALS_FILENAME, "r") as f:
            self.essentials = json.load(f)

    def setup(self, stage: Optional[str] = None) -> None:
        """Prepares data for each process given a stage."""
        if stage == 'fit' or stage is None:
            self.data_train = HDF5Dataset(
                filename=PROCESSED_DATA_FILENAME,
                data_dataset_name="x_train",
                targets_dataset_name="y_train",
                worker_buffer_size=self.worker_buffer_size,
                transform=self.train_transform,
                transform_target=_change_target_dtype
            )
            self.data_val = HDF5Dataset(
                filename=PROCESSED_DATA_FILENAME,
                data_dataset_name="x_val",
                targets_dataset_name="y_val",
                worker_buffer_size=self.worker_buffer_size,
                transform=self.val_transform,
                transform_target=_change_target_dtype
            )

        if stage == 'test' or stage is None:
            self.data_test = HDF5Dataset(
                filename=PROCESSED_DATA_FILENAME,
                data_dataset_name="x_test",
                targets_dataset_name="y_test",
                worker_buffer_size=self.worker_buffer_size,
                transform=self.val_transform,
                transform_target=_change_target_dtype
            )

    def train_dataloader(self) -> DataLoader:
        """Returns prepared train dataloader."""
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=False
        )


def _change_target_dtype(target: np.ndarray) -> np.ndarray:
    """Change dtype of the array to np.int64 (long). Used for targets during loading."""
    return target.astype(dtype=np.int64)


def _process_kaggle_imagenet(classes: int, images_per_class: int, processing_workers: int) -> None:
    """Processes downloaded imagenet data from kaggle.

    Args:
        classes: A number of classes to process.
        images_per_class: A number of images per class to process.
        processing_workers: A number of subprocesses to use for reading images.
    """
    raise NotImplementedError


def _process_imagenet(
    classes: int,
    images_per_class: int,
    use_kaggle: bool,
    rewrite: bool,
    processing_workers: int
) -> None:
    """Processes downloaded imagenet data.

    Args:
        classes: A number of classes to process.
        images_per_class: A number of images per class to process.
        use_kaggle: If true, will processes the kaggle .tar.gz file, otherwise downloaded images in images folder.
        rewrite: It true, will processes raw data even if a processed file already exists.
        processing_workers: a number of subprocesses to use for reading images.
    """
    if not rewrite and os.path.exists(PROCESSED_DATA_FILENAME):
        print("Dataset already processed and rewrite=False.")
        return
    if use_kaggle:
        print("Using kaggle dataset.")
        if not os.path.exists(DATASET_KAGGLE_FILENAME):
            dataset_path_str = str(DATASET_KAGGLE_FILENAME)
            print(f"No file found at {dataset_path_str}")
            return
        _process_kaggle_imagenet(classes, images_per_class, processing_workers)
    else:
        processor = DownloadedProcessor(
            images_dirname=IMAGES_DATA_DIRNAME,
            synset_mapping_filename=SYNSET_MAPPING_FILENAME,
            processed_data_filename=PROCESSED_DATA_FILENAME,
            essentials_filename=ESSENTIALS_FILENAME,
            classes=classes,
            images_per_class=images_per_class,
            resize_size=RESIZE_SIZE,
            trainval_split=TRAINVAL_SPLIT,
            train_split=TRAIN_SPLIT,
            processing_workers=PROCESSING_WORKERS,
            processing_batch_size=PROCESSING_BATCH_SIZE,
            seed=SEED
        )
        processor.process_imagenet()


def _get_synset_urls() -> Tuple[Dict[str, Optional[List[str]]], Dict[str, str]]:
    """Downloads a list of url for each synset based on a mapping.

    Returns:
        A tuple of a dict from synsets to their corresponding urls
        and a dict of synsets to their names
    """
    synsets, synset_mapping = parse_synset_mapping(SYNSET_MAPPING_FILENAME)
    synsets_to_urls = download_image_urls(URLS_FILENAME, synsets)
    return synsets_to_urls, synset_mapping


def _download_subsampled(
    synsets_to_urls: Dict[str, List[str]],
    classes: int,
    images_per_class: int,
    max_concurrent: int = 16,
    timeout: float = 15.0
) -> List[int]:
    """Downloads a subset of imagenet images.

    Args:
        synsets_to_urls: A map of synsets to lists of all corresponding urls.
        classes: A number of classes to download.
        images_per_class: A number of images per class to download.
        max_concurrent (optional): A number of concurrent download-writes.
        timeout (optional): Time until abandoning a download.
    """
    synset_stats = get_synset_stats(synsets_to_urls)
    synset_flickr_sort = []
    for synset in synset_stats:
        synset_flickr_sort.append([synset_stats[synset]["n_flickr_urls"], synset_stats[synset]["n_urls"], synset])
    synset_flickr_sort.sort(reverse=True)
    synset_subset = [synset for _, _, synset in synset_flickr_sort[:classes]]

    downloaded_images = download_subsampled_images(
        images_dirname=IMAGES_DATA_DIRNAME,
        synsets=synset_subset,
        synsets_to_urls=synsets_to_urls,
        images_per_class=images_per_class,
        max_concurrent=max_concurrent,
        timeout=timeout
    )
    return downloaded_images


def _parse_args() -> argparse.Namespace:
    """Parses args for downloading images."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--classes", type=int, default=SUBSAMPLE_CLASSES, help="Number of classes to download")
    parser.add_argument("--images_per_class", type=int, default=IMAGES_PER_CLASS, help="Number of images per class")
    parser.add_argument("--max_concurrent", type=int, default=8, help="Maximum number of concurrent download-writes.")
    parser.add_argument("--timeout", type=float, default=15.0, help="Time before abandoning download.")

    args = parser.parse_args()
    return args


def download_images() -> None:
    """Downloads a subset of imagenet images based on args and prints download count."""
    args = _parse_args()
    synsets_to_urls, synset_mapping = _get_synset_urls()  # pylint: disable=unused-variable
    start_t = time()
    res = _download_subsampled(synsets_to_urls, args.classes, args.images_per_class, args.max_concurrent, args.timeout)
    print("Downloaded images per class:", res)
    print("Time: {:.2f}".format(time() - start_t))


if __name__ == "__main__":
    download_images()
