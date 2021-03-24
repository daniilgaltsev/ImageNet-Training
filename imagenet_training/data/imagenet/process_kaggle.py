"""Module for processing .tar.gz imagenet file from kaggle."""


from pathlib import Path
from time import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .download_urls import parse_synset_mapping
from .process_base import BaseProcessor

VAL_SOLUTION_NAME = "LOC_val_solution.csv.zip"
KAGGLE_DATA_SUBPATH = Path("ILSVRC/Data/CLS-LOC")
TRAIN_DATA_SUBPATH = KAGGLE_DATA_SUBPATH / "train"
VAL_DATA_SUBPATH = KAGGLE_DATA_SUBPATH / "val"

MAX_IMAGES_PER_CLASS = 1300


class KaggleProcessor(BaseProcessor):
    """Processes kaggle imagenet images.

    Args:
        dl_data_dir: A path to the download directory containing LOC_val_solution.csv.zip.
        kaggle_dataset_dirname: A path to the directory containing kaggle images.
        synset_mapping_filename: A path to the synset mapping file.
        processed_data_filename: A path where to save the processed data.
        essentials_filename: A path where to save the essentials file.
        classes: A number of classes to process.
        images_per_class: A number of images per class to process.
        resize_size: A size at which to save images.
        train_split: A size of train with respect to val.
        processing_workers: A number of workers to use for image processing.
        processing_batch_size: A number of images to process at a time.
        seed: A number used for seeding.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        dl_data_dir: Path,
        kaggle_dataset_dirname: Path,
        synset_mapping_filename: Path,
        processed_data_filename: Path,
        essentials_filename: Path,
        classes: int,
        images_per_class: int,
        resize_size: Tuple[int, int],
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
        self.dl_data_dir = dl_data_dir
        self.synset_mapping_filename = synset_mapping_filename
        self.classes = classes
        self.images_per_class = min(images_per_class, MAX_IMAGES_PER_CLASS)
        self.train_split = train_split
        self.seed = seed
        self.train_data_dir = kaggle_dataset_dirname / TRAIN_DATA_SUBPATH
        self.val_data_dir = kaggle_dataset_dirname / VAL_DATA_SUBPATH

    def _get_val_sol(self) -> pd.DataFrame:
        """Reads and returns dataframe containing predictions for a validation split."""
        val_sol_filename = self.dl_data_dir / VAL_SOLUTION_NAME
        val_sol = pd.read_csv(val_sol_filename)
        val_sol["synset"] = val_sol["PredictionString"].str.split().str[0]
        return val_sol

    def _get_train_and_val_splits(
        self,
        synsets: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, List[Path], List[Path], Dict[str, int]]:
        """Reads and returns labels and paths for train and val splits.

        Args:
            synsets: A list of synsets to use.
        """
        synset_to_class = {}
        image_paths_trainval = []
        y_trainval = []
        for idx, synset in enumerate(synsets):
            synset_to_class[synset] = idx
            image_paths_trainval.extend(list((self.train_data_dir / synset).glob("*"))[:self.images_per_class])
            y_trainval.extend([idx] * self.images_per_class)
        y_train, y_val, image_paths_train, image_paths_val = train_test_split(
            y_trainval, image_paths_trainval,
            train_size=self.train_split, random_state=self.seed, shuffle=True, stratify=y_trainval
        )
        y_train = np.array(y_train)
        y_val = np.array(y_val)
        return y_train, y_val, image_paths_train, image_paths_val, synset_to_class

    def _get_test_split(self, synsets: List[str], synset_to_class: Dict[str, int]) -> Tuple[np.ndarray, List[Path]]:
        """Reads and returns labels and paths for a test split.

        Args:
            synsets: A list of synsets to use.
            synset_to_class: A mapping between a synset and its class.
        """
        val_sol = self._get_val_sol()
        val_sol = val_sol[val_sol["synset"].isin(synsets)]
        y_test = []
        image_paths_test = []
        for _, row in val_sol.iterrows():
            y_test.append(synset_to_class[row["synset"]])
            image_paths_test.append(self.val_data_dir / (row["ImageId"] + ".JPEG"))
        y_test = np.array(y_test)
        return y_test, image_paths_test

    def process_imagenet(self) -> None:  # pylint: disable=arguments-differ
        """Processes kaggle imagenet data."""
        start_t = time()

        print(f"Getting images stats. {time() - start_t:.2f}")
        _, synset_mapping = parse_synset_mapping(self.synset_mapping_filename)
        synsets = [path.name for path in self.train_data_dir.glob("*")][:self.classes]
        y_train, y_val, image_paths_train, image_paths_val, synset_to_class = self._get_train_and_val_splits(synsets)
        y_test, image_paths_test = self._get_test_split(synsets, synset_to_class)

        super().process_imagenet(
            y_train, y_val, y_test,
            image_paths_train, image_paths_val, image_paths_test,
            synset_to_class, synset_mapping, start_t
        )
