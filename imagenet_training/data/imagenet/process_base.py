"""A base class for ImageNet dataset processing."""

import json
import multiprocessing as mp
from pathlib import Path
from time import time
from typing import List, Dict, Tuple

import cv2
import h5py
import numpy as np


class BaseProcessor:
    """Processes imagenet images.

    Args:
        processed_data_filename: A path where to save the processed data.
        essentials_filename: A path where to save the essentials file.
        resize_size: A size at which to save images.
        processing_workers: A number of workers to use for image processing.
        processing_batch_size: A number of images to process at a time.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        processed_data_filename: Path,
        essentials_filename: Path,
        resize_size: Tuple[int, int],
        processing_workers: int,
        processing_batch_size: int,
    ):
        self.processed_data_filename = processed_data_filename
        self.essentials_filename = essentials_filename
        self.resize_size = resize_size
        self.processing_workers = processing_workers
        self.processing_batch_size = processing_batch_size

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
            start = i * self.processing_batch_size
            end = min((i + 1) * self.processing_batch_size, len(image_paths))
            print(f"Processing {name} batch {i + 1}/{n_batches} of size {end - start}. {time() - start_t:.2f}")
            images = pool.starmap(
                _read_image,
                zip(
                    image_paths[start:end],
                    (self.resize_size for _ in range(start, end))
                )
            )
            dataset[start:end] = images

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

    def process_imagenet(
        self,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
        image_paths_train: List[Path],
        image_paths_val: List[Path],
        image_paths_test: List[Path],
        synset_to_class: Dict[str, int],
        synset_mapping: Dict[str, str],
        start_t: float
    ) -> None:
        """Creates HDF5 and essentials files given data.

        Args:
            y_train: An array of train labels.
            y_val: An array of val labels.
            y_test: An array of test labels.
            image_paths_train: A list of paths to train images.
            image_paths_val: A list of paths to val images.
            image_paths_test: A list of paths to test images.
            synset_to_class: A mapping between a synset and the corresponding class.
            synset_mapping: A mapping between a synset and its name.
            start_t: Time at which the processing started.
        """
        print(f"Will save train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")
        print(f"Creating HDF5 file. {time() - start_t:.2f}")
        self._create_hdf5(y_train, y_val, y_test, image_paths_train, image_paths_val, image_paths_test, start_t)

        print(f"Saving essential dataset parameters to imagenet_training/data. {time() - start_t:.2f}")
        image_shape = (*self.resize_size, 3)
        self._create_essentials(synset_to_class, synset_mapping, image_shape)

        print(f"Done processing. {time() - start_t:.2f}")


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
