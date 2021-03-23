"""Base LightningModule class."""

import argparse
from typing import Any, Optional

import pytorch_lightning as pl
import torch
import torch.optim
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn as nn


OPTIMIZER = "Adam"
LR = 1e-3
LR_SCHEDULER = None
PCT_START = 0.3
LOSS = "cross_entropy"


class BaseLitModel(pl.LightningModule):
    """Base LightningModule class.

    Args:
        model: a torch module to use as backbone.
        args (optional): parsed args from argparse.
    """

    def __init__(
        self,
        model: nn.Module,
        args: Optional[argparse.Namespace] = None
    ):
        super().__init__()

        self.model = model
        if args is None:
            self.args = {}
        else:
            self.args = vars(args)

        optimizer = self.args.get("optimizer", OPTIMIZER)
        self.optimizer_class = getattr(torch.optim, optimizer)
        self.scheduler = self.args.get("lr_scheduler", LR_SCHEDULER)
        self.pct_start = self.args.get("pct_start", PCT_START)

        self.lr = self.args.get("lr", LR)

        loss = self.args.get("loss", LOSS)
        self.loss_fn = getattr(nn.functional, loss)

        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()

    @staticmethod
    def add_to_argparse(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Adds possible args to the given parser."""
        parser.add_argument("--optimizer", type=str, default=OPTIMIZER, help="Name of optimizer from torch.optim.")
        parser.add_argument("--lr", type=float, default=LR, help="Base learning rate.")
        parser.add_argument("--lr_scheduler", type=str, default=None, help="LR scheduler to use.")
        parser.add_argument(
            "--pct_start", type=float, default=PCT_START,
            help="The percentage of the cycle spent increasing the learning rate. Used with OneCycleLR."
        )
        parser.add_argument("--loss", type=str, default=LOSS, help="Name of loss function from torch.nn.functional.")
        return parser

    def configure_optimizers(self) -> Any:
        """Inits and returns optimizer to use."""
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)

        if self.scheduler == "OneCycleLR":
            num_gpus = self.trainer.num_gpus
            num_gpus += (num_gpus == 0)
            steps_per_epoch = len(self.train_dataloader()) // num_gpus
            epochs = self.trainer.max_epochs
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, self.lr,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                verbose=True
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

        return optimizer

    def forward(self, x: Any) -> Any:  # pylint: disable=arguments-differ
        """Performs a forward operation."""
        return self.model(x)

    def training_step(self, batch: Any, batch_idx: int):  # pylint: disable=unused-argument,arguments-differ
        """Performs a training step on a given batch."""
        x, y = batch

        logits = self(x)

        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)

        self.train_acc(logits, y)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument,arguments-differ
        """Performs a validation step on a given batch."""
        x, y = batch

        logits = self(x)

        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss, prog_bar=True)

        self.train_acc(logits, y)
        self.log("val_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int) -> None:  # pylint: disable=unused-argument,arguments-differ
        """Performs a test step on a given batch."""
        x, y = batch

        logits = self(x)

        loss = self.loss_fn(logits, y)
        self.log("test_loss", loss)

        self.train_acc(logits, y)
        self.log("test_acc", self.train_acc, on_step=False, on_epoch=True)
