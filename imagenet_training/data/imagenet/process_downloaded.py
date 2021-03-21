"""Module for processing downloaded imagenet images."""

from collections import defaultdict
import json
import multiprocessing as mp
from pathlib import Path
from time import time
from typing import Dict, List, Tuple

import cv2
import h5py
import numpy as np
from sklearn.model_selection import train_test_split

from .download_urls import parse_synset_mapping


class DownloadedProcessor:
    """Processes downloaded imagenet images.

    Args:
        images_dirname: A path to the directory containing images.
        synset_mapping_filename: A path to the synset mapping file.
        processed_data_filename: A path where to save the processed data.
        essentials_filename: A path where to save the essentials file.
        classes: A number of classes to process.
        images_per_class: A number of images per class to process.
        resize_size: A size at which to save images.
        trainval_split: A size of trainval with respect to test.
        train_split: A size of train with respect to val.
        processing_workers: A number of workers to use for image processing.
        processing_workers: A number of images to process at a time.
        seed: A number used for seeding.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        images_dirname: Path,
        synset_mapping_filename: Path,
        processed_data_filename: Path,
        essentials_filename: Path,
        classes: int,
        images_per_class: int,
        resize_size: Tuple[int, int],
        trainval_split: float,
        train_split: float,
        processing_workers: int,
        processing_batch_size: int,
        seed: int
    ):
        self.images_dirname = images_dirname
        self.synset_mapping_filename = synset_mapping_filename
        self.processed_data_filename = processed_data_filename
        self.essentials_filename = essentials_filename
        self.classes = classes
        self.images_per_class = images_per_class
        self.resize_size = resize_size
        self.trainval_split = trainval_split
        self.train_split = train_split
        self.processing_workers = processing_workers
        self.processing_batch_size = processing_batch_size
        self.seed = seed

    def _process_split(
        self,
        pool: mp.pool.Pool,
        image_paths: List[Path],
        dataset: h5py.Dataset,
        start_t: float,
        name: str
    ) -> None:
        """Processes a split (e.g. train) of data.

        Args:
            pool: A pool of workers to use for reading images.
            image_paths: A list of paths to images.
            dataset: A h5py dataset where to write data.
            start_t: Time from which to generate log.
            name: Name of the split (e.g. "train").
        """
        n_batches = (len(image_paths) - 1) // self.processing_batch_size + 1
        for i in range(n_batches):
            print("Processing {} batch {}/{}. {:.2f}".format(name, i + 1, n_batches, time() - start_t))
            start = i * self.processing_batch_size
            end = (i + 1) * self.processing_batch_size
            images = pool.starmap(
                _read_image,
                zip(
                    image_paths[start:end],
                    [self.resize_size for _ in range(start, end)]
                )
            )
            dataset[start:end] = images

    def _get_splits(
        self,
        y: List[int],
        image_paths: List[Path]
    ) -> Tuple[List[Path], List[Path], List[Path], np.ndarray, np.ndarray, np.ndarray]:
        """Splits data into train/val/split.

        Args:
            y: A list of classes.
            image_paths: A list of paths to images.
        """
        y_trainval, y_test, image_paths_trainval, image_paths_test = train_test_split(
            y, image_paths,
            train_size=self.trainval_split, random_state=self.seed, shuffle=True, stratify=y
        )
        y_train, y_val, image_paths_train, image_paths_val = train_test_split(
            y_trainval, image_paths_trainval,
            train_size=self.train_split, random_state=self.seed, shuffle=True, stratify=y_trainval
        )
        y_train, y_val, y_test = np.array(y_train), np.array(y_val), np.array(y_test)
        return image_paths_train, image_paths_val, image_paths_test, y_train, y_val, y_test

    def _create_hdf5(
        self,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
        image_paths_train: List[Path],
        image_paths_val: List[Path],
        image_paths_test: List[Path],
        start_t: float
    ) -> None:
        """Processes images and create a HDF5 file.

        Args:
            y_train: A list of classes for train split.
            y_val: A list of classes for val split.
            y_test: A list of classes for test split.
            image_paths_train: A list of paths to images for train split.
            image_paths_val: A list of paths to images for test split.
            image_paths_test: A list of paths to images for val split.
            start_t: Time from which to generate log.
        """
        image_shape = (*self.resize_size, 3)
        chunk_shape = (1, *image_shape)
        x_train_shape = (len(y_train), *image_shape)
        x_val_shape = (len(y_val), *image_shape)
        x_test_shape = (len(y_test), *image_shape)
        self.processed_data_filename.parent.mkdir(exist_ok=True, parents=True)
        with h5py.File(self.processed_data_filename, 'w') as file:
            dataset_train = file.create_dataset("x_train", shape=x_train_shape, chunks=chunk_shape, dtype=np.uint8)
            file.create_dataset("y_train", data=y_train, dtype=np.int32)
            dataset_val = file.create_dataset("x_val", shape=x_val_shape, chunks=chunk_shape, dtype=np.uint8)
            file.create_dataset("y_val", data=y_val, dtype=np.int32)
            dataset_test = file.create_dataset("x_test", shape=x_test_shape, chunks=chunk_shape, dtype=np.uint8)
            file.create_dataset("y_test", data=y_test, dtype=np.int32)

            with mp.Pool(self.processing_workers) as pool:
                self._process_split(
                    pool=pool, image_paths=image_paths_train, dataset=dataset_train, start_t=start_t, name="train"
                )
                self._process_split(
                    pool=pool, image_paths=image_paths_val, dataset=dataset_val, start_t=start_t, name="val"
                )
                self._process_split(
                    pool=pool, image_paths=image_paths_test, dataset=dataset_test, start_t=start_t, name="test"
                )

    def _create_essentials(
        self,
        synset_to_class: Dict[str, int],
        synset_mapping: Dict[str, str],
        image_shape: Tuple[int, int, int]
    ) -> None:
        """Creates essentials file.

        Args:
            synset_to_class: A mapping between a synset and its class.
            synset_mapping: A mapping between a synset and its name.
            image_shape: A shape of the images in the processes dataset.
        """
        essentials = {
            "synset_to_class": synset_to_class,
            "synset_to_name": synset_mapping,
            "input_size": image_shape
        }
        with open(self.essentials_filename, "w") as f:
            json.dump(essentials, f)

    def _get_images_to_parse(self) -> Tuple[List[Path], List[int], Dict[str, str]]:
        """Reads the image names and finds which image to use.

        Returns:
            A list of paths to images, a list of classes of these images and a dict from synset to a class.
        """
        image_paths = [*self.images_dirname.glob("*.jpg")]

        image_synsets = []
        for path in image_paths:
            image_synsets.append(path.name.split('_')[0])

        added_class_count = defaultdict(int)
        synset_to_class = {}
        y = []
        indicies = []
        for idx, synset in enumerate(image_synsets):
            if synset not in added_class_count and len(added_class_count) == self.classes:
                continue
            if added_class_count[synset] < self.images_per_class:
                if synset not in synset_to_class:
                    synset_to_class[synset] = len(synset_to_class)
                added_class_count[synset] += 1
                y.append(synset_to_class[synset])
                indicies.append(idx)
        image_paths = [image_paths[idx] for idx in indicies]
        return image_paths, y, synset_to_class

    def process_imagenet(self) -> None:
        """Processes downloaded imagenet data from urls (using python -m imagenet_training.data.imagenet.imagenet)."""
        start_t = time()

        print("Getting downloaded images stats. {:.2f}".format(time() - start_t))
        _, synset_mapping = parse_synset_mapping(self.synset_mapping_filename)
        image_paths, y, synset_to_class = self._get_images_to_parse()
        image_paths_train, image_paths_val, image_paths_test, y_train, y_val, y_test = self._get_splits(
            y, image_paths
        )

        print("Will save train={}, val={}, test={}".format(len(y_train), len(y_val), len(y_test)))
        print("Creating HDF5 file. {:.2f}".format(time() - start_t))
        self._create_hdf5(y_train, y_val, y_test, image_paths_train, image_paths_val, image_paths_test, start_t)

        print("Saving essential dataset parameters to imagenet_training/data. {:.2f}".format(time() - start_t))
        image_shape = (*self.resize_size, 3)
        self._create_essentials(synset_to_class, synset_mapping, image_shape)

        print("Done processing. {:.2f}".format(time() - start_t))


def _read_image(filename: Path, resize_size: Tuple[int, int]) -> np.ndarray:
    """Reads and resizes an image given its path.

    Args:
        filename: path to the image to read.
        resize_size: a size of output image.
    """
    image = cv2.imread(str(filename))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, resize_size)
    return image
