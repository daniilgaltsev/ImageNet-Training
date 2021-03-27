"""Base LightningModule class."""

import argparse
import inspect
from typing import Any, Optional

import pytorch_lightning as pl
import torch
import torch.optim
import torch.nn as nn


OPTIMIZER = "Adam"
LR = 1e-3
WEIGHT_DECAY = 0.0
LR_SCHEDULER = None
REDUCELRONPLATEAU_MODE = 'min'
REDUCELRONPLATEAU_FACTOR = 0.1
REDUCELRONPLATEAU_PATIENCE = 5
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
        if self.scheduler is not None:
            self.scheduler_class = getattr(torch.optim.lr_scheduler, self.scheduler)
            self.mode = self.args.get("mode", REDUCELRONPLATEAU_MODE)
            self.factor = self.args.get("factor", REDUCELRONPLATEAU_FACTOR)
            self.patience = self.args.get("patience", REDUCELRONPLATEAU_PATIENCE)
            self.pct_start = self.args.get("pct_start", PCT_START)

        self.lr = self.args.get("lr", LR)
        self.weight_decay = self.args.get("weight_decay", WEIGHT_DECAY)

        loss = self.args.get("loss", LOSS)
        self.loss_fn = getattr(nn.functional, loss)

        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()

    @staticmethod
    def add_to_argparse(
        parser: argparse.ArgumentParser,
        main_parser: argparse.ArgumentParser
    ) -> argparse.ArgumentParser:
        """Adds possible args to the given parser."""
        parser.add_argument("--optimizer", type=str, default=OPTIMIZER, help="Name of optimizer from torch.optim.")
        parser.add_argument("--lr_scheduler", type=str, default=None, help="LR scheduler to use from torch.optim.")

        args, _ = main_parser.parse_known_args()
        parser = _add_optimizer_args(parser, args)
        parser = _add_lr_scheduler_args(parser, args)

        parser.add_argument("--loss", type=str, default=LOSS, help="Name of loss function from torch.nn.functional.")
        return parser

    def configure_optimizers(self) -> Any:
        """Inits and returns optimizer to use."""
        optimizer_args = inspect.getfullargspec(self.optimizer_class).args
        optimizer_input = {"lr": self.lr}
        if "weight_decay" in optimizer_args:
            optimizer_input["weight_decay"] = self.weight_decay
        optimizer = self.optimizer_class(self.parameters(), **optimizer_input)

        if self.scheduler is None:
            return optimizer

        if self.scheduler == "OneCycleLR":
            num_gpus = self.trainer.num_gpus
            num_gpus += (num_gpus == 0)
            steps_per_epoch = len(self.train_dataloader()) // num_gpus
            epochs = self.trainer.max_epochs
            scheduler = self.scheduler_class(
                optimizer, self.lr,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=self.pct_start
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        if self.scheduler == "ReduceLROnPlateau":
            scheduler = self.scheduler_class(
                optimizer, mode=self.mode, factor=self.factor, patience=self.patience, verbose=True
            )
            return [optimizer], [{"scheduler": scheduler, "monitor": "val_loss"}]

        scheduler = self.scheduler_class(optimizer)
        return [optimizer], [{"scheduler": scheduler}]

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


def _add_optimizer_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace
) -> argparse.ArgumentParser:
    """Adds arguments for parsing used for optimizer.

    Args:
        parser: Parser for adding args.
        args: Partially parsed arguments containing optimizer arg.
    """
    opt_cls = getattr(torch.optim, args.optimizer)
    opt_args = inspect.getfullargspec(opt_cls).args

    parser.add_argument("--lr", type=float, default=LR, help="Base learning rate.")
    if "weight_decay" in opt_args:
        parser.add_argument(
            "--weight_decay", type=float, default=WEIGHT_DECAY, help="Weight decay to use in optimizer."
        )
    return parser


def _add_lr_scheduler_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace
) -> argparse.ArgumentParser:
    """Adds arguments for parsing used for lr scheduler.

    Args:
        parser: Parser for adding args.
        args: Partially parsed arguments containing lr_scheduler arg.
    """
    if args.lr_scheduler == "OneCycleLR":
        parser.add_argument(
            "--pct_start", type=float, default=PCT_START,
            help="The percentage of the cycle spent increasing the learning rate."
        )
    elif args.lr_scheduler == "ReduceLROnPlateau":
        parser.add_argument(
            "--mode", type=str, default=REDUCELRONPLATEAU_MODE,
            help="One of min, max. Used in ReduceLROnPlateau."
        )
        parser.add_argument(
            "--factor", type=float, default=REDUCELRONPLATEAU_FACTOR,
            help="Factor bt which lr is reduced. Used in ReduceLROnPlateau."
        )
        parser.add_argument(
            "--patience", type=int, default=REDUCELRONPLATEAU_PATIENCE,
            help="Number of epochs with no improvement to tolerated. Used in ReduceLROnPlateau."
        )

    return parser
