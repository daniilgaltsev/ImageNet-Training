"""Module for processing downloaded imagenet images."""

from collections import defaultdict
from pathlib import Path
from time import time
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from .download_urls import parse_synset_mapping
from .process_base import BaseProcessor


class DownloadedProcessor(BaseProcessor):
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
        processing_batch_size: A number of images to process at a time.
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
        super().__init__(
            processed_data_filename=processed_data_filename,
            essentials_filename=essentials_filename,
            resize_size=resize_size,
            processing_workers=processing_workers,
            processing_batch_size=processing_batch_size
        )
        self.images_dirname = images_dirname
        self.synset_mapping_filename = synset_mapping_filename
        self.classes = classes
        self.images_per_class = images_per_class
        self.trainval_split = trainval_split
        self.train_split = train_split
        self.seed = seed

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

    def process_imagenet(self) -> None:  # pylint: disable=arguments-differ
        """Processes downloaded imagenet data from urls (using python -m imagenet_training.data.imagenet.imagenet)."""
        start_t = time()

        print("Getting downloaded images stats. {:.2f}".format(time() - start_t))
        _, synset_mapping = parse_synset_mapping(self.synset_mapping_filename)
        image_paths, y, synset_to_class = self._get_images_to_parse()
        image_paths_train, image_paths_val, image_paths_test, y_train, y_val, y_test = self._get_splits(
            y, image_paths
        )

        super().process_imagenet(
            y_train, y_val, y_test,
            image_paths_train, image_paths_val, image_paths_test,
            synset_to_class, synset_mapping, start_t
        )
