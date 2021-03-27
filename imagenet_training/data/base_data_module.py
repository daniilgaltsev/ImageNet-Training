"""Base DataModule class."""

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader


BATCH_SIZE = 32
NUM_WORKERS = 0


class BaseDataModule(pl.LightningDataModule):
    """Base data module class."""

    def __init__(self, args: Optional[argparse.Namespace]):
        super().__init__()

        if args is None:
            self.args = {}
        else:
            self.args = vars(args)
        self.batch_size = self.args.get("batch_size", BATCH_SIZE)
        self.num_workers = self.args.get("num_workers", NUM_WORKERS)

        self.dims = None
        self.output_dims = None
        self.mapping = None

        self.data_train = None
        self.data_val = None
        self.data_test = None

    @classmethod
    def data_dirname(cls) -> Path:
        """Returns the path to the data folder."""
        return Path(__file__).resolve().parents[2] / "data"

    @staticmethod
    def add_to_argparse(
        parser: argparse.ArgumentParser,
        main_parser: argparse.ArgumentParser  # pylint: disable=unused-argument
    ) -> argparse.ArgumentParser:
        """Adds arguments to parser required for the BaseDataModule."""
        parser.add_argument(
            "--batch_size", type=int, default=BATCH_SIZE, help="Number of examples per batch."
        )
        parser.add_argument(
            "--num_workers", type=int, default=NUM_WORKERS, help="Number of subprocesses for data loading."
        )

        return parser

    def config(self) -> Dict[str, Any]:
        """Returns a dict of options used to init a model."""
        return {"input_dims": self.dims, "output_dims": self.output_dims, "mapping": self.mapping}

    def prepare_data(self) -> None:  # pylint: disable=arguments-differ
        """Prepares data for the node globally."""
        return

    def setup(self, stage: Optional[str]) -> None:  # pylint: disable=signature-differs
        """Prepares data for each process given stage."""
        return

    def train_dataloader(self) -> DataLoader:  # pylint: disable=arguments-differ
        """Returns prepared train dataloader."""
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=True
        )

    def val_dataloader(self) -> DataLoader:  # pylint: disable=arguments-differ
        """Returns prepared validation dataloader."""
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            shuffle=False
        )

    def test_dataloader(self) -> DataLoader:  # pylint: disable=arguments-differ
        """Returns prepared test dataloader."""
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            shuffle=False
        )
